"""Stage lifecycle: assign -> resolve -> rollback (+ failure modes)."""

import pytest

from datascience.registry.stages import (
    assign_stage,
    resolve_stage,
    rollback,
    stage_path,
)


def _make_versions(root, name, *versions):
    for v in versions:
        (root / name / v).mkdir(parents=True)


def test_assign_resolve_and_path(tmp_path):
    _make_versions(tmp_path, "M", "v1", "v2")
    root = str(tmp_path)

    assert assign_stage("M", "v1", "production", root) == "v1"
    assert resolve_stage("M", "production", root) == "v1"
    assert stage_path("M", "production", root).endswith("M/v1")
    # bare number normalizes
    assert assign_stage("M", "2", "production", root) == "v2"
    assert resolve_stage("M", "production", root) == "v2"
    # staging independent
    assert resolve_stage("M", "staging", root) is None


def test_rollback_restores_previous(tmp_path):
    _make_versions(tmp_path, "M", "v1", "v2", "v3")
    root = str(tmp_path)
    assign_stage("M", "v1", "production", root)
    assign_stage("M", "v2", "production", root)
    assign_stage("M", "v3", "production", root)

    assert rollback("M", "production", root) == "v2"
    assert resolve_stage("M", "production", root) == "v2"
    assert rollback("M", "production", root) == "v1"
    with pytest.raises(ValueError, match="no previous version"):
        rollback("M", "production", root)


def test_guards(tmp_path):
    _make_versions(tmp_path, "M", "v1")
    root = str(tmp_path)
    with pytest.raises(ValueError, match="no registered version"):
        assign_stage("M", "v9", "production", root)
    with pytest.raises(ValueError, match="unknown stage"):
        assign_stage("M", "v1", "canary", root)
    with pytest.raises(ValueError, match="nothing assigned"):
        stage_path("M", "staging", root)
