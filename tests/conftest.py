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
    }
    for name, value in mapping.items():
        monkeypatch.setattr(paths, name, value, raising=False)
    mapping["EVAL_DIR"].mkdir(parents=True, exist_ok=True)
    return tmp_path
