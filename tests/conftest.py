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
        "DATA_DIR": tmp_path / "data_ingestion",
        "RAW_DATA": tmp_path / "data_ingestion" / "data.jsonl",
        "SPLIT_DIR": tmp_path / "data_transformation",
        "TRAIN_FILE": tmp_path / "data_transformation" / "train.jsonl",
        "VAL_FILE": tmp_path / "data_transformation" / "val.jsonl",
        "TEST_FILE": tmp_path / "data_transformation" / "test.jsonl",
        "LABEL_MAP": tmp_path / "data_transformation" / "label_map.json",
        "DATA_VERSION": tmp_path / "data_transformation" / "data_version.json",
        "REGISTRY_DIR": tmp_path / "registry",
        "EVAL_DIR": tmp_path / "model_evaluation",
        "METRICS_FILE": tmp_path / "model_evaluation" / "metrics.json",
    }
    for name, value in mapping.items():
        monkeypatch.setattr(paths, name, value, raising=False)
    for d in (mapping["DATA_DIR"], mapping["SPLIT_DIR"], mapping["EVAL_DIR"]):
        d.mkdir(parents=True, exist_ok=True)
    return tmp_path
