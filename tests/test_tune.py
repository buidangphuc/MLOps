"""Tuning loop: a failing variant is recorded and the others still get ranked."""

import json

import pytest

from datascience.constants import paths
from datascience.tune import loop


def test_failed_variant_is_recorded_and_the_rest_ranked(tmp_artifacts, monkeypatch):
    exp = tmp_artifacts / "experiments.yaml"
    exp.write_text("experiment_name: t\nvariants:\n  - name: good\n  - name: bad\n  - name: also-good\n")
    monkeypatch.setattr(paths, "EXPERIMENTS_FILE", exp)

    def fake_train_eval(cm, name):
        if name == "bad":
            raise ValueError("Unexpected base config")
        paths.METRICS_FILE.write_text(json.dumps({"macro_f1": 0.9 if name == "good" else 0.8}))
        paths.TRAIN_STATS.write_text(json.dumps({"samples_per_s": 70.0, "peak_mem_gb": None}))
        (paths.MODEL_DIR / "config.json").write_text("{}")

    monkeypatch.setattr(loop, "_train_eval_one", fake_train_eval)
    with pytest.raises(SystemExit, match="bad"):
        loop.run_tuning()

    runs = json.loads(paths.TUNE_RESULTS.read_text())["runs"]
    assert [r["variant"] for r in runs] == ["good", "also-good", "bad"]
    assert "Unexpected base config" in runs[2]["error"] and runs[2]["objective"] is None
    assert runs[0]["metrics"]["samples_per_s"] == 70.0 and "peak_mem_gb" not in runs[0]["metrics"]
