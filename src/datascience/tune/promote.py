"""Promote a tuning variant into the registry — manual, you pick the winner.

Reads the loop's ranking (artifacts/tune_runs/results.json), re-checks the gate on the
chosen variant, then versions its snapshotted model with full LINEAGE: run_id + the
resolved config (hyperparam snapshot) + metrics. This is the link that ties a registry
version back to the exact experiment that produced it.
"""

import json

import yaml

from datascience import log, registry
from datascience.config.configuration import ConfigurationManager
from datascience.constants import paths
from datascience.gate import QualityGate
from datascience.utils.io import read_data_version


def promote(variant: str):
    if not paths.TUNE_RESULTS.exists():
        raise SystemExit("no tuning results — run `python main.py tune` first")

    results = json.loads(paths.TUNE_RESULTS.read_text())
    entry = next((r for r in results["runs"] if r["variant"] == variant), None)
    if entry is None:
        choices = [r["variant"] for r in results["runs"]]
        raise SystemExit(f"variant '{variant}' not found; choices: {choices}")
    if "error" in entry:
        raise SystemExit(f"variant '{variant}' failed during tuning: {entry['error']}")

    run_dir = paths.TUNE_DIR / variant
    metrics = json.loads((run_dir / "metrics.json").read_text())
    config = yaml.safe_load((run_dir / "config.yaml").read_text())

    cm = ConfigurationManager()  # base config -> registry name + gate rules
    if not QualityGate(cm.gate()).check(metrics):  # raises GateFailed if on_fail=error
        return

    scalar = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    metadata = {
        "variant": variant,
        "run_id": entry.get("run_id"),
        "experiment": results.get("experiment"),
        "base_model": config["model"]["base_model"],
        "data_version": read_data_version(),
        "config": config,  # full hyperparam snapshot -> lineage
    }
    result = registry.register(run_dir / "model", scalar, metadata, cm.registry())
    log.info(
        f"promoted '{variant}' -> registry {cm.registry().name} "
        f"v{result['local_version']} (run_id={entry.get('run_id')})"
    )
    return result
