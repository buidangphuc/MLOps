"""OPTIMIZE — export GLiNER → ONNX + int8 for the cheap CPU serving path, GUARDED by a
parity gate so quantization can't silently degrade quality.

Flow:
  1. export the trained GLiNER to ONNX and int8-dynamic quantize it  (CPU speedup)
  2. PARITY GATE: score torch vs onnx-int8 on val with the SAME eval metric (per-entity F1)
     + measure latency. FAIL if F1 drops more than serving.optimize.max_f1_drop.

Constraint: GLiNER (encoder + LSTM + dynamic span decode) has NO built-in ONNX export and
full-model export is finicky — this step is validated on the GPU box, not here. The parity
gate is exactly the automatic catch: a broken/quantization-degraded export tanks F1 and the
gate fails, so a bad ONNX never reaches serving.
"""

import json
import time

from src.datascience import log
from src.datascience.config.configuration import ConfigurationManager
from src.datascience.constants import paths


def optimize():
    cm = ConfigurationManager()
    opt = cm.serving().optimize
    _export_onnx(quantize=opt.get("quantize", "int8-dynamic"))
    if opt.get("parity_check", True):
        parity_gate(cm, max_f1_drop=float(opt.get("max_f1_drop", 0.01)),
                    sample=int(opt.get("parity_sample", 2000)))
    log.info(f"optimized ONNX model ready -> {paths.OPTIMIZED_DIR}")


def _export_onnx(quantize: str):
    """Export trained GLiNER -> ONNX (+ int8). Best-effort; validate on the GPU box.

    GLiNER ships no export helper. This attempts a whole-model torch.onnx.export; if the
    LSTM / dynamic span decode won't trace, it raises with guidance (fall back to
    encoder-only export or GLiNER's onnx example script). The parity gate below is the
    safety net regardless of how the ONNX was produced.
    """
    paths.OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from src.datascience.model import export_gliner_onnx  # thin wrapper, see model.py

        export_gliner_onnx(paths.MODEL_DIR, paths.OPTIMIZED_DIR, quantize=quantize)
        log.info(f"exported ONNX ({quantize}) -> {paths.OPTIMIZED_DIR}")
    except Exception as e:
        raise RuntimeError(
            f"GLiNER ONNX export failed: {e}\n"
            "GLiNER has no export helper and full-model export is finicky (LSTM + dynamic "
            "spans). Run this on the GPU box / GLiNER onnx example, or export encoder-only. "
            "The parity gate will validate whatever ONNX is produced."
        ) from e


def parity_gate(cm=None, max_f1_drop=0.01, sample=2000):
    """Score torch vs onnx-int8 on a val sample with the eval metric; FAIL if F1 drops too far.

    Reuses model_evaluation's scoring so the parity number is the SAME metric the real eval
    reports. Writes artifacts/optimize/parity.json.
    """
    cm = cm or ConfigurationManager()
    from src.datascience.components.model_evaluation import ModelEvaluation
    from src.datascience.infer.predict import Predictor
    from src.datascience.model import load_labels
    from src.datascience.utils.io import read_jsonl

    ev = cm.evaluation()
    labels = load_labels(paths.LABEL_MAP)
    rows = read_jsonl(paths.VAL_FILE, required_fields=[ev.text_field, ev.label_field])[:sample]

    f1_torch, lat_torch = _score_predictor(Predictor(device="cpu"), rows, labels, ev, ModelEvaluation)
    f1_onnx, lat_onnx = _score_predictor(
        Predictor(model_dir=paths.OPTIMIZED_DIR, device="cpu", onnx=True), rows, labels, ev, ModelEvaluation
    )
    drop = f1_torch - f1_onnx
    result = {
        "f1_torch": round(f1_torch, 4), "f1_onnx": round(f1_onnx, 4), "f1_drop": round(drop, 4),
        "max_f1_drop": max_f1_drop, "latency_torch_ms": round(lat_torch, 1),
        "latency_onnx_ms": round(lat_onnx, 1), "speedup": round(lat_torch / lat_onnx, 2) if lat_onnx else None,
        "n": len(rows), "passed": drop <= max_f1_drop,
    }
    paths.PARITY_FILE.parent.mkdir(parents=True, exist_ok=True)
    paths.PARITY_FILE.write_text(json.dumps(result, indent=2))
    log.info(f"parity: f1 torch={result['f1_torch']} onnx={result['f1_onnx']} "
             f"drop={result['f1_drop']} (max {max_f1_drop}); speedup x{result['speedup']}")
    if not result["passed"]:
        raise RuntimeError(
            f"PARITY GATE FAILED: onnx-int8 F1 dropped {drop:.4f} > {max_f1_drop} — "
            "quantization degraded quality; do not deploy this ONNX. See artifacts/optimize/parity.json"
        )
    log.info("parity gate PASSED — onnx-int8 keeps quality")
    return result


def _score_predictor(predictor, rows, labels, ev, ModelEvaluation):
    """(per-entity F1 on `rows`, mean latency ms) for a Predictor, via eval's set-based scorer."""
    from src.datascience.components.model_evaluation import _score

    me = ModelEvaluation(ev)
    gold = me._gold_pairs(rows, labels)
    texts = [" ".join(r[ev.text_field]) for r in rows]
    t0 = time.time()
    preds = predictor.predict_batch(texts, batch_size=ev.eval_batch_size)
    latency_ms = (time.time() - t0) / max(1, len(rows)) * 1000
    pred_pairs = {(i, e["label"]) for i, p in enumerate(preds)
                  for e in p["entities"] if e["label"] in set(labels)}
    return _score(gold, pred_pairs, labels, per_class=False)[ev.objective], latency_ms
