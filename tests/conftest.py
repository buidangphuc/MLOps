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
        "RUN_STATUS": tmp_path / "run_status.json",
    }
    for name, value in mapping.items():
        monkeypatch.setattr(paths, name, value, raising=False)
    for d in ("EVAL_DIR", "SPLIT_DIR", "MODEL_DIR"):
        mapping[d].mkdir(parents=True, exist_ok=True)
    return tmp_path
