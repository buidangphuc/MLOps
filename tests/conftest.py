"""Shared fixture: redirect the fixed artifact paths into a tmp tree.

Tests that write artifacts monkeypatch `datascience.constants.paths`
attributes; this fixture centralizes the common ones so individual tests
stop repeating the boilerplate.
"""

import pytest

from datascience.constants import paths


@pytest.fixture()
def tmp_artifacts(tmp_path, monkeypatch):
    """Point the artifact tree at a tmp dir; returns the tmp root."""
    mapping = {
        "ARTIFACTS": tmp_path,
        "REGISTRY_DIR": tmp_path / "registry",
        "REGISTRY_CACHE": tmp_path / "registry_cache",
        "EVAL_DIR": tmp_path / "model_evaluation",
        "METRICS_FILE": tmp_path / "model_evaluation" / "metrics.json",
        "PERF_DIR": tmp_path / "perf",
        "TUNE_DIR": tmp_path / "tune_runs",
        "TUNE_RESULTS": tmp_path / "tune_runs" / "results.json",
        "SPLIT_DIR": tmp_path / "data_transformation",
        "TRAIN_FILE": tmp_path / "data_transformation" / "train.jsonl",
        "VAL_FILE": tmp_path / "data_transformation" / "val.jsonl",
        "TEST_FILE": tmp_path / "data_transformation" / "test.jsonl",
        "LABEL_MAP": tmp_path / "data_transformation" / "label_map.json",
        "MODEL_DIR": tmp_path / "model_trainer" / "model",
        "CKPT_DIR": tmp_path / "model_trainer" / "checkpoints",
        "TRAIN_STATS": tmp_path / "model_trainer" / "train_stats.json",
        "TRAIN_RUN_FILE": tmp_path / "model_trainer" / "mlflow_run.json",
        "DATA_VERSION": tmp_path / "data_transformation" / "data_version.json",
        "RUN_STATUS": tmp_path / "run_status.json",
    }
    for name, value in mapping.items():
        monkeypatch.setattr(paths, name, value, raising=False)
    for d in ("EVAL_DIR", "SPLIT_DIR", "MODEL_DIR"):
        mapping[d].mkdir(parents=True, exist_ok=True)
    return tmp_path


TOY_PARAMS = """\
pack: toy
model: {base_model: toy-base}
toy: {rows: 6, score: 0.7}
metrics: {compute: [score], objective: score}
gate: {enabled: true, on_fail: warn, rules: [{metric: score, min: 0.5}]}
tracking: {backend: mlflow, experiment_name: toy}
registry: {name: ToyModel, remote: {backend: mlflow, mlflow: {alias: champion}}}
"""

TOY_SERVING = """\
sagemaker:
  region: us-east-1
  models: [{name: alpha}, {name: beta}]
  release: {build_store: s3://example-bucket/builds/toy}
"""


@pytest.fixture()
def toy_model(tmp_path, monkeypatch):
    """A self-contained model `toy` (pack tests/toy) rooted in a tmp config + artifacts tree,
    tracked in a tmp sqlite MLflow. Returns its ModelPaths."""
    import importlib.util
    import sys
    import types
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("packs.toy.pack", Path(__file__).parent / "toy" / "pack.py")
    toy_pack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toy_pack)

    cfg = tmp_path / "config" / "toy"
    cfg.mkdir(parents=True)
    (cfg / "params.yaml").write_text(TOY_PARAMS)
    (cfg / "serving.yaml").write_text(TOY_SERVING)
    monkeypatch.setattr(paths, "CONFIG_ROOT", tmp_path / "config")
    monkeypatch.setattr(paths, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(paths, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setitem(sys.modules, "packs.toy", types.ModuleType("packs.toy"))
    monkeypatch.setitem(sys.modules, "packs.toy.pack", toy_pack)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(paths, "_active", None)
    return paths.use("toy")
