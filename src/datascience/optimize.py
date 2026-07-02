"""OPTIMIZE — export the trained model to ONNX for serving (+ optional quantize).

Lean: export → optionally quantize → re-eval on test and print the metric so you
can eyeball the speed/quality trade-off. No hard gate (you decide).
"""

import numpy as np

from datascience import log
from datascience import metrics as M
from datascience.config.configuration import ConfigurationManager
from datascience.constants import paths
from datascience.model import load_label_map
from datascience.utils.io import read_jsonl


def optimize():
    cm = ConfigurationManager()
    quantize = cm.serving().optimize.get("quantize", "none")
    out = paths.OPTIMIZED_DIR

    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    log.info(f"exporting {paths.MODEL_DIR} -> ONNX at {out}")
    model = ORTModelForSequenceClassification.from_pretrained(paths.MODEL_DIR, export=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(paths.MODEL_DIR).save_pretrained(out)

    if quantize == "dynamic-int8":
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        log.info("applying dynamic INT8 quantization")
        quantizer = ORTQuantizer.from_pretrained(out)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        quantizer.quantize(save_dir=out, quantization_config=qconfig)

    _reeval(cm, out)
    log.info(f"optimized model ready -> {out}")


def _reeval(cm, onnx_dir):
    """Run the ONNX model on test and print the objective metric (informational)."""
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    ev = cm.evaluation()
    label2id, _ = load_label_map(paths.LABEL_MAP)
    model = ORTModelForSequenceClassification.from_pretrained(onnx_dir)
    tokenizer = AutoTokenizer.from_pretrained(onnx_dir)

    rows = [
        r
        for r in read_jsonl(paths.TEST_FILE, required_fields=[ev.text_field, ev.label_field])
        if r[ev.label_field] in label2id
    ]
    y_true = np.array([label2id[r[ev.label_field]] for r in rows])

    chunks = []
    for i in range(0, len(rows), ev.eval_batch_size):
        texts = [r[ev.text_field] for r in rows[i : i + ev.eval_batch_size]]
        enc = tokenizer(texts, truncation=True, max_length=ev.max_length, padding=True, return_tensors="pt")
        chunks.append(np.asarray(model(**enc).logits))
    probs = M.softmax(np.concatenate(chunks))

    scores = M.compute_scalar_metrics(ev.compute, y_true, probs.argmax(1), probs)
    log.info(f"ONNX re-eval (vs fp32 torch): { {k: round(v, 4) for k, v in scores.items()} }")
