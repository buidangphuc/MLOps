"""The tuning loop (Layer 2): train+eval every variant in experiments.yaml and rank
them by the objective metric.

Each variant is a sparse override deep-merged onto params.yaml, run through the SAME
Layer-1 pipeline, and logged as its own MLflow run (tagged variant=<name>) so they
compare side by side. The variant's model + metrics + resolved config are snapshotted
under artifacts/tune_runs/<variant>/ so it can be promoted later. Nothing is promoted
automatically — read the ranking, then `python main.py promote <variant>`.
"""

import json
import shutil

import yaml

from datascience import log
from datascience.config.configuration import ConfigurationManager
from datascience.constants import paths
from datascience.pipeline import stages
from datascience.tune.variants import load_experiments, resolve_variant


def run_tuning():
    exp = load_experiments()
    experiment_name = exp.get("experiment_name", "tuning")
    objective = yaml.safe_load(paths.PARAMS_FILE.read_text())["metrics"]["objective"]

    results = []
    for variant in exp["variants"]:
        name = variant["name"]
        log.info(f">>>>>> tuning variant '{name}' <<<<<<")

        # Same resolution path as `train --variant <name>`, plus the shared experiment
        # name so all variants land in one MLflow experiment for side-by-side comparison.
        config_path = resolve_variant(name, experiment_name=experiment_name, clean=True)
        run_dir = config_path.parent

        # fresh checkpoints per variant (else auto-resume picks up the previous variant's)
        shutil.rmtree(paths.CKPT_DIR, ignore_errors=True)

        cm = ConfigurationManager(params_filepath=config_path)
        paths.TRAIN_STATS.unlink(missing_ok=True)
        try:
            run_id = _train_eval_one(cm, name)
        except Exception as e:   # one broken variant must not cost the others their results
            log.error(f"variant '{name}' failed: {e!r}")
            results.append({"variant": name, "run_id": None, "passed": False, "objective": None,
                            "metrics": {}, "dir": str(run_dir), "error": repr(e)})
            continue

        metrics = json.loads(paths.METRICS_FILE.read_text())
        scalar = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        if paths.TRAIN_STATS.exists():
            scalar.update({k: v for k, v in json.loads(paths.TRAIN_STATS.read_text()).items() if v is not None})
        passed = _gate_passed(cm, metrics)

        # snapshot model + metrics so this variant can be promoted after the loop
        shutil.copytree(paths.MODEL_DIR, run_dir / "model")
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        if paths.DATA_VERSION.exists():
            shutil.copy2(paths.DATA_VERSION, run_dir / "data_version.json")

        results.append({
            "variant": name,
            "run_id": run_id,
            "passed": passed,
            "objective": scalar.get(objective),
            "metrics": scalar,
            "dir": str(run_dir),
        })

    # don't leave the last variant's checkpoint behind — it would poison a later plain `train`
    # (auto-resume could load a mismatched optimizer state).
    shutil.rmtree(paths.CKPT_DIR, ignore_errors=True)

    results.sort(key=lambda r: (r["objective"] is not None, r["objective"] or 0), reverse=True)
    paths.TUNE_RESULTS.write_text(json.dumps(
        {"experiment": experiment_name, "objective": objective, "runs": results}, indent=2
    ))
    _print_ranking(objective, results)
    failed = [r["variant"] for r in results if "error" in r]
    if failed:
        raise SystemExit(f"variants failed: {failed} (see the log; the others are ranked in {paths.TUNE_RESULTS})")
    return results


def _train_eval_one(cm, variant):
    """Run the pack's stages up to `evaluate` for one variant (no registration); return the
    training MLflow run_id (None if tracking off). Stages after `evaluate` only warn."""
    config_path = paths.TUNE_DIR / variant / "config.yaml"
    all_names = stages.names()
    cut = all_names.index("evaluate") + 1
    paths.TRAIN_RUN_FILE.unlink(missing_ok=True)
    for name in all_names[:cut]:
        stages.run_stage(name, config_path, register=False)

    run_id = json.loads(paths.TRAIN_RUN_FILE.read_text())["run_id"] if paths.TRAIN_RUN_FILE.exists() else None
    if cm.params.tracking.backend == "mlflow" and run_id:
        import mlflow

        mlflow.set_tracking_uri(cm.tracking_uri())
        metrics = json.loads(paths.METRICS_FILE.read_text())
        scalar = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tag("variant", variant)
            mlflow.log_artifact(str(config_path))
            mlflow.log_metrics({f"test_{k}": v for k, v in scalar.items()})

    for name in all_names[cut:]:
        try:
            stages.run_stage(name, config_path, register=False)
        except RuntimeError as e:
            log.warning(f"[tune] {variant}: {name} failed, keeping the variant: {e}")
    return run_id


def _gate_passed(cm, metrics) -> bool:
    # In the loop a failing gate must NOT abort the other variants, so swallow GateFailed.
    from datascience.gate import GateFailed, QualityGate

    try:
        return QualityGate(cm.gate()).check(metrics)
    except GateFailed:
        return False


def _print_ranking(objective, results):
    log.info(f"===== tuning ranking (by {objective}) =====")
    for i, r in enumerate(results, 1):
        val = f"{r['objective']:.4f}" if r["objective"] is not None else "n/a"
        gate = "ERROR" if "error" in r else "PASS" if r["passed"] else "FAIL"
        log.info(f"  {i}. {r['variant']:<20} {objective}={val}  gate={gate}")
    log.info("promote the winner with:  python main.py promote <variant>")
