"""Named configs from experiments.yaml — the saved list of experiment settings.

A variant is a sparse override deep-merged onto params.yaml. This module is the ONE
place that resolves a name into a config file, so `train --variant X` and the same
variant inside `tune` can never drift apart.

Resolution writes the merged config to artifacts/tune_runs/<name>/config.yaml and
returns that path; ConfigurationManager(params_filepath=...) reads it from there.
"""

import shutil

from pathlib import Path

import yaml

from datascience.constants import paths
from datascience.utils.io import atomic_write_text


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively overlay `override` onto `base` (variant overrides win)."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_experiments() -> dict:
    if not paths.EXPERIMENTS_FILE.exists():
        raise FileNotFoundError(f"experiments file not found: {paths.EXPERIMENTS_FILE}")
    return yaml.safe_load(paths.EXPERIMENTS_FILE.read_text())


def list_variants() -> list:
    """Names of the saved configs, in file order."""
    return [v["name"] for v in load_experiments().get("variants", [])]


def resolve_variant(name: str, experiment_name: str = None, clean: bool = False) -> Path:
    """Merge variant `name` onto params.yaml and materialise it.

    `experiment_name` overrides tracking.experiment_name so the tuning loop can group
    all its variants under one MLflow experiment; a standalone run keeps params.yaml's.

    `clean` wipes the run dir first. Only the tuning loop wants that (each variant gets
    a fresh dir); resolving a variant to run it must NOT delete the model/metrics
    snapshot a previous `tune` left there for `promote` to read.
    """
    exp = load_experiments()
    variants = {v["name"]: v for v in exp.get("variants", [])}
    if name not in variants:
        raise ValueError(f"unknown variant '{name}' (choices: {sorted(variants)})")

    override = {k: v for k, v in variants[name].items() if k != "name"}
    merged = _deep_merge(yaml.safe_load(paths.PARAMS_FILE.read_text()), override)
    if experiment_name:
        merged.setdefault("tracking", {})["experiment_name"] = experiment_name

    run_dir = paths.TUNE_DIR / name
    if clean and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    # Atomic write: under DDP all ranks call this and would otherwise race on the same
    # path (a reader could see a half-written file). Each rank writes identical content,
    # so an atomic temp+rename makes concurrent writes safe.
    atomic_write_text(config_path, yaml.safe_dump(merged, sort_keys=False, allow_unicode=True))
    return config_path
