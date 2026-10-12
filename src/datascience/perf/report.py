"""Quant-vs-quality report: this box's stored infer sweep (+ tune runs) as one markdown/CSV table.

Pure formatting over `store.load("infer", "infer")` and `paths.TUNE_RESULTS` — nothing is measured
here. Files land next to the sweep: <PERF_DIR>/infer/<hw_id>/report.md, report.csv (and
report_tune.csv when tune runs exist).
"""

import csv
import io
import json

import yaml

from datascience.constants import paths
from datascience.perf import store
from datascience.perf.hw import fingerprint

INFER_COLS = ["chosen", "backend", "precision", "batch_size", "throughput", "tokens_per_s",
              "p99_ms", "peak_mem_gb", "quality", "quality_drop", "feasible", "reason"]
TUNE_COLS = ["variant", "method", "objective", "samples_per_s", "peak_mem_gb"]


def infer_rows(result: dict) -> list[dict]:
    rows = []
    for t in result["trials"]:
        c, m = t["config"], t["metrics"]
        rows.append({"chosen": "*" if c == result["chosen"] else "", "backend": c.get("backend"),
                     "precision": c.get("precision"), "batch_size": c.get("batch_size"),
                     **{k: m.get(k) for k in INFER_COLS[4:10]},
                     "feasible": t["feasible"], "reason": t["reason"]})
    return rows


def tune_rows() -> list[dict]:
    if not paths.TUNE_RESULTS.exists():
        return []
    rows = []
    for run in json.loads(paths.TUNE_RESULTS.read_text())["runs"]:
        cfg_path = paths.TUNE_DIR / run["variant"] / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        rows.append({"variant": run["variant"], "method": (cfg.get("finetune") or {}).get("method", "full"),
                     "objective": run["objective"], "samples_per_s": run["metrics"].get("samples_per_s"),
                     "peak_mem_gb": run["metrics"].get("peak_mem_gb")})
    return rows


def _cell(v) -> str:
    return "" if v is None else str(v)


def _csv(rows: list[dict], cols: list[str]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    w.writerows([_cell(r[c]) for c in cols] for r in rows)
    return buf.getvalue()


def _md(rows: list[dict], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_cell(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def build_report(fp: dict | None = None, track: bool = False) -> str:
    """Write report.md/report.csv for this box (+ log them to MLflow `perf`); return the markdown."""
    fp = fp or fingerprint()
    try:
        result = store.load("infer", "infer", fp)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e} — run `datascience perf tune infer` on this box first") from e
    rows, tune = infer_rows(result), tune_rows()
    md = (f"# Quant vs quality — {fp['hw_id']}\n\n"
          f"objective: {result['objective']} ({result['mode']}); `*` = chosen\n\n{_md(rows, INFER_COLS)}\n")
    if tune:
        md += f"\n## Tune runs\n\n{_md(tune, TUNE_COLS)}\n"
    out = paths.PERF_DIR / "infer" / fp["hw_id"]
    (out / "report.md").write_text(md)
    (out / "report.csv").write_text(_csv(rows, INFER_COLS))
    if tune:
        (out / "report_tune.csv").write_text(_csv(tune, TUNE_COLS))
    if track:
        import os

        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", paths.MLFLOW_URI))
        mlflow.set_experiment("perf")
        with mlflow.start_run(tags={"role": "infer", "hw_id": fp["hw_id"], "workload": "report"}):
            mlflow.log_artifacts(str(out), artifact_path="report")
    return md
