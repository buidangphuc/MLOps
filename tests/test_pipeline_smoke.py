"""Fast, offline smoke tests for the pure pipeline logic (no model download, no GPU).

The full HF fine-tune path is exercised by `python main.py run`; these cover the
logic that must stay correct: io, metrics, gate, split helpers, local registry.

Run: pytest tests/test_pipeline_smoke.py
"""

import json

import numpy as np
import pytest

from src.datascience.components.data_transformation import _can_stratify, _drop_cross_split_dups
from src.datascience.entity.config_entity import GateConfig
from src.datascience.gate import GateFailed, QualityGate
from src.datascience.metrics import (
    bootstrap_ci,
    compute_scalar_metrics,
    expected_calibration_error,
    softmax,
)
from src.datascience.registry.local import LocalRegistry
from src.datascience.utils.io import read_jsonl, write_jsonl


def test_jsonl_roundtrip_and_skip_bad(tmp_path):
    path = tmp_path / "d.jsonl"
    write_jsonl(path, [{"text": "a", "label": "x"}, {"text": "b", "label": "y"}])
    with open(path, "a") as f:
        f.write("{not valid json}\n")  # malformed line must be skipped, not fatal
    rows = read_jsonl(path, required_fields=["text", "label"])
    assert len(rows) == 2


def test_metrics_and_calibration():
    y = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])
    m = compute_scalar_metrics(["accuracy", "macro_f1", "micro_f1"], y, y_pred)
    assert set(m) == {"accuracy", "macro_f1", "micro_f1"}
    assert 0.0 <= m["accuracy"] <= 1.0

    probs = softmax(np.random.RandomState(0).randn(6, 3))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert 0.0 <= expected_calibration_error(probs, y) <= 1.0

    lo, hi = bootstrap_ci(lambda a, b: float((a == b).mean()), y, y_pred, n=200)
    assert lo <= hi


def test_gate_pass_fail_warn():
    rules = [{"metric": "macro_f1", "min": 0.5}]
    assert QualityGate(GateConfig(True, "error", rules)).check({"macro_f1": 0.8}) is True
    with pytest.raises(GateFailed):
        QualityGate(GateConfig(True, "error", rules)).check({"macro_f1": 0.3})
    assert QualityGate(GateConfig(True, "warn", rules)).check({"macro_f1": 0.3}) is False


def test_split_helpers():
    assert _can_stratify(["a", "a", "b", "b"]) is True
    assert _can_stratify(["a", "b"]) is False  # min class count 1
    train = [{"t": "x"}]
    val = [{"t": "x"}, {"t": "y"}]
    test = [{"t": "y"}, {"t": "z"}]
    _, val2, test2 = _drop_cross_split_dups(train, val, test, "t")
    assert all(r["t"] != "x" for r in val2)  # 'x' (in train) dropped from val
    assert all(r["t"] != "y" for r in test2)  # 'y' (now in val) dropped from test


def test_local_registry_versions(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    reg = LocalRegistry(root=str(tmp_path / "registry"), name="M")
    assert reg.register(str(model_dir), {"macro_f1": 0.90}, {"base": "x"}) == "1"
    assert reg.register(str(model_dir), {"macro_f1": 0.91}, {"base": "x"}) == "2"
    latest = json.loads((tmp_path / "registry" / "M" / "latest.json").read_text())
    assert latest["version"] == 2
    assert (tmp_path / "registry" / "M" / "v2" / "config.json").exists()
