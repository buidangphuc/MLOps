"""Drift monitoring — warn-only, no labels needed (the standalone half of observability).

Compares a batch of REAL production inputs against the reference profile saved at eval
(predicted-label distribution + mean confidence + input length) and flags drift. It does NOT
act: closing the loop (live quality gate, retrain) needs ground-truth labels + lifecycle,
which are deferred. Run it on captured traffic, read the report, decide manually.
"""

import collections
import json

from pathlib import Path

from src.datascience import log
from src.datascience import metrics as M
from src.datascience.config.configuration import ConfigurationManager
from src.datascience.constants import paths
from src.datascience.utils.io import atomic_write_json, read_jsonl


def monitor():
    cm = ConfigurationManager()
    m = cm.serving().monitor
    if not paths.REFERENCE_PROFILE.exists():
        raise SystemExit("no reference profile — run `python main.py eval` first")
    if not m.get("traffic_file"):
        raise SystemExit("set serving.monitor.traffic_file — captured prod inputs (JSONL {text})")

    ref = json.loads(paths.REFERENCE_PROFILE.read_text())
    texts = [r["text"] for r in read_jsonl(Path(m.traffic_file), required_fields=["text"])]

    from src.datascience.infer.predict import Predictor

    preds = Predictor().predict_batch(texts)
    counts = collections.Counter(p["label"] for p in preds)
    cur = {
        "n": len(texts),
        "label_dist": {k: v / len(preds) for k, v in counts.items()},
        "mean_confidence": float(sum(p["score"] for p in preds) / len(preds)),
        "char_len_mean": float(sum(len(t) for t in texts) / len(texts)),
    }

    psi = M.population_stability_index(ref["label_dist"], cur["label_dist"])
    conf_drop = ref["mean_confidence"] - cur["mean_confidence"]
    len_change = abs(cur["char_len_mean"] - ref["char_len_mean"]) / max(ref["char_len_mean"], 1e-6)

    flags = []
    if psi > m.psi_warn:
        flags.append(f"label-distribution PSI={psi:.3f} > {m.psi_warn}")
    if conf_drop > m.confidence_drop_warn:
        flags.append(f"mean confidence dropped {conf_drop:.3f} > {m.confidence_drop_warn}")
    if len_change > m.length_change_warn:
        flags.append(f"input length changed {len_change:.0%} > {m.length_change_warn:.0%}")

    report = {
        "n_monitored": len(texts),
        "psi": psi,
        "confidence_drop": conf_drop,
        "length_change": len_change,
        "drift": bool(flags),
        "flags": flags,
        "reference": ref,
        "current": cur,
    }
    atomic_write_json(paths.MONITOR_REPORT, report)

    if flags:
        log.warning("DRIFT detected: " + "; ".join(flags))
    else:
        log.info(f"no drift (PSI={psi:.3f}, conf_drop={conf_drop:.3f}, len_change={len_change:.0%})")
    log.info(f"monitor report -> {paths.MONITOR_REPORT}")
    return report
