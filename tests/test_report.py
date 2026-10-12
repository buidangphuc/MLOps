"""Quant-vs-quality report: rows from the stored infer sweep + tune runs, md/csv written."""

import csv
import json

import pytest

from datascience.constants import paths
from datascience.perf import report, store

FP = {"hw_id": "tesla-t4-1x-abc", "gpus": []}


def _trial(precision, thr, drop, feasible=True, reason=None, batch=8, **extra):
    m = {"throughput": thr, "p99_ms": 12.5, "peak_mem_gb": 1.5, "quality": 0.9 - drop,
         "quality_drop": drop, **extra}
    return {"config": {"backend": "torch", "precision": precision, "batch_size": batch},
            "metrics": m, "feasible": feasible, "reason": reason}


@pytest.fixture()
def stored(tmp_artifacts):
    trials = [_trial("fp32", 100.0, 0.0), _trial("int8-dyn", 180.0, 0.002, tokens_per_s=900.0),
              _trial("nf4", 300.0, 0.05, feasible=False, reason="quality_drop=0.05 > 0.01")]
    store.save("infer", "infer", FP, {"objective": "throughput", "mode": "max", "constraints": {},
                                      "grid": {}, "trials": trials, "chosen": trials[1]["config"],
                                      "chosen_metrics": trials[1]["metrics"]})
    return trials


def test_infer_results_only(stored):
    md = report.build_report(FP)
    out = paths.PERF_DIR / "infer" / FP["hw_id"]
    rows = list(csv.DictReader((out / "report.csv").open()))
    assert len(rows) == 3
    assert (out / "report.md").read_text() == md
    assert "Tune runs" not in md
    assert rows[0]["tokens_per_s"] == "" and rows[1]["tokens_per_s"] == "900.0"


def test_no_stored_results(tmp_artifacts):
    with pytest.raises(FileNotFoundError, match="perf tune infer"):
        report.build_report(FP)


def test_chosen_and_rejected_rows(stored):
    md = report.build_report(FP)
    lines = {ln.split("|")[3].strip(): ln for ln in md.splitlines() if "| torch |" in ln}
    assert lines["int8-dyn"].split("|")[1].strip() == "*"
    assert lines["fp32"].split("|")[1].strip() == ""
    assert "quality_drop=0.05 > 0.01" in lines["nf4"]


def test_tune_runs_section(stored):
    for name, cfg in [("base", "a: 1\n"), ("lora", "finetune: {method: lora}\n")]:
        d = paths.TUNE_DIR / name
        d.mkdir(parents=True)
        (d / "config.yaml").write_text(cfg)
    paths.TUNE_RESULTS.write_text(json.dumps({"objective": "f1", "runs": [
        {"variant": "lora", "objective": 0.91, "metrics": {"samples_per_s": 12.0, "peak_mem_gb": 6.1}},
        {"variant": "base", "objective": 0.88, "metrics": {}}]}))
    md = report.build_report(FP)
    out = paths.PERF_DIR / "infer" / FP["hw_id"]
    rows = list(csv.DictReader((out / "report_tune.csv").open()))
    assert [(r["variant"], r["method"], r["objective"]) for r in rows] == [
        ("lora", "lora", "0.91"), ("base", "full", "0.88")]
    assert rows[1]["samples_per_s"] == "" and rows[0]["peak_mem_gb"] == "6.1"
    assert "Tune runs" in md
