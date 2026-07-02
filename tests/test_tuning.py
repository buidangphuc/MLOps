"""Offline tests for the Layer-2 tuning, lineage, and de-hardcode logic (no GPU/training).

The full tune/promote/analyze paths are exercised on real infra; these cover the pure
logic that must stay correct: config deep-merge, the train/objective metric wiring, the
data fingerprint, and the Model Analyzer config shape.

Run: pytest tests/test_tuning.py
"""

import types

import numpy as np

from src.datascience.config.configuration import ConfigurationManager
from src.datascience.metrics import compute_scalar_metrics
from src.datascience.serve_analyzer import _ma_config
from src.datascience.tune.loop import _deep_merge
from src.datascience.utils.io import data_fingerprint


def test_deep_merge_nested():
    base = {"training": {"epochs": 2, "lr": 5e-5}, "model": {"base_model": "distilbert"}}
    out = _deep_merge(base, {"training": {"epochs": 4}})
    assert out["training"]["epochs"] == 4
    assert out["training"]["lr"] == 5e-5        # sibling untouched
    assert out["model"]["base_model"] == "distilbert"
    assert base["training"]["epochs"] == 2      # base not mutated


def test_trainer_compute_includes_objective():
    # de-hardcode: the best-model metric must be among the metrics computed each eval
    t = ConfigurationManager().trainer()
    assert t.objective in t.compute


def test_tuning_defaults_off():
    tun = ConfigurationManager().tuning()
    assert tun.freeze_encoder_layers == 0
    assert tun.lora is False
    assert tun.differential_lr is False


def test_data_fingerprint(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"text":"a"}\n{"text":"b"}\n')
    fp = data_fingerprint(p)
    assert fp["n_rows"] == 2 and len(fp["sha256"]) == 12
    p.write_text('{"text":"a"}\n{"text":"c"}\n')
    assert data_fingerprint(p)["sha256"] != fp["sha256"]   # content change -> new version


def test_compute_scalar_metrics_non_default_objective():
    y, pred = np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])
    m = compute_scalar_metrics(["mcc", "macro_f1"], y, pred)
    assert "mcc" in m and "macro_f1" in m       # a non-default objective is still computed


def test_ma_config_shape(tmp_path):
    a = types.SimpleNamespace(max_instance_count=4, latency_budget_ms=50, max_batch_size=64)
    cfg = _ma_config(a, tmp_path / "repo", tmp_path / "traffic.json")
    assert cfg["run_config_search_max_instance_count"] == 4
    pm = cfg["profile_models"]["text_classifier"]
    assert pm["constraints"]["perf_latency_p99"]["max"] == 50
    assert pm["objectives"] == ["perf_throughput"]
    assert "input-data" in pm["perf_analyzer_flags"]
