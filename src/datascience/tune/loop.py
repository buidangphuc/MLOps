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
from datascience.components.model_evaluation import ModelEvaluation
from datascience.components.model_trainer import ModelTrainer
from datascience.config.configuration import ConfigurationManager
from datascience.constants import paths


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively overlay `override` onto `base` (variant overrides win)."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def run_tuning():
    exp = yaml.safe_load(paths.EXPERIMENTS_FILE.read_text())
    experiment_name = exp.get("experiment_name", "tuning")
    base = yaml.safe_load(paths.PARAMS_FILE.read_text())
    objective = base["metrics"]["objective"]

    results = []
    for variant in exp["variants"]:
        variant = dict(variant)
        name = variant.pop("name")
        log.info(f">>>>>> tuning variant '{name}' <<<<<<")

        # resolved config = base + sparse override; group all variants in one experiment
        merged = _deep_merge(base, variant)
        merged.setdefault("tracking", {})["experiment_name"] = experiment_name

        run_dir = paths.TUNE_DIR / name
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(merged, sort_keys=False))

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
    """Train + eval one variant; return its MLflow run_id (None if tracking off).

    HF auto-logs its TrainingArguments to the run; we additionally tag the run and
    attach the full resolved config + test metrics so the experiment is fully versioned.
    """
    use_mlflow = cm.trainer().tracking_backend == "mlflow"
    if use_mlflow:
        import mlflow

        mlflow.set_tracking_uri(cm.tracking_uri())

    ModelTrainer(cm.trainer()).train()
    run_id = None
    if use_mlflow:
        last = mlflow.last_active_run()
        run_id = last.info.run_id if last else None

    ModelEvaluation(cm.evaluation()).evaluate()

    if use_mlflow and run_id:
        metrics = json.loads(paths.METRICS_FILE.read_text())
        scalar = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tag("variant", variant)
            mlflow.log_artifact(str(paths.TUNE_DIR / variant / "config.yaml"))
            mlflow.log_metrics({f"test_{k}": v for k, v in scalar.items()})
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
