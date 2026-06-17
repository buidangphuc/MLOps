"""Local inference endpoint: FastAPI + ONNX Runtime (torch-free).

Loads the packaged bundle (onnx + tokenizer + label_map + temperature) and
serves text -> {label, score}. This is the endpoint a business app calls.
"""

import json

import numpy as np

from src.datascience import log
from src.datascience.constants import paths


class _OnnxModel:
    def __init__(self, bundle=None):
        import onnxruntime as ort

        from transformers import AutoTokenizer

        bundle = bundle or paths.BUNDLE_DIR
        card = json.loads((bundle / "model_card.json").read_text())
        self.max_length = card.get("max_length", 128)

        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.session = ort.InferenceSession(str(bundle / "model.onnx"), providers=providers or None)
        self.tokenizer = AutoTokenizer.from_pretrained(bundle)
        lm = json.loads((bundle / "label_map.json").read_text())
        self.id2label = {int(k): v for k, v in lm["id2label"].items()}
        self.temperature = json.loads((bundle / "temperature.json").read_text()).get("temperature", 1.0)
        log.info(f"loaded ONNX model (providers={self.session.get_providers()})")

    def predict(self, texts):
        enc = self.tokenizer(
            texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="np"
        )
        logits = self.session.run(
            None,
            {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)},
        )[0]
        z = logits / self.temperature
        e = np.exp(z - z.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        idx = probs.argmax(axis=1)
        return [{"label": self.id2label[int(i)], "score": float(probs[k][i])} for k, i in enumerate(idx)]


class _TorchModel:
    """Serve the trained torch model directly — the default runtime, no ONNX conversion."""

    def __init__(self):
        from src.datascience.infer.predict import Predictor

        self.predictor = Predictor()
        log.info("loaded torch model (runtime=torch)")

    def predict(self, texts):
        return self.predictor.predict_batch(texts)


def _runtime() -> str:
    import yaml  # available via transformers; avoids pulling ConfigurationManager (box) on the ONNX path

    cfg = yaml.safe_load(paths.SERVING_FILE.read_text()) or {}
    return cfg.get("runtime", "torch")


def _load_model(bundle=None):
    """torch (default) = serve the trained model; onnx (opt-in) = serve the packaged bundle."""
    if _runtime() == "onnx":
        bundle = bundle or paths.BUNDLE_DIR
        if not (bundle / "model_card.json").exists():
            raise SystemExit(
                "runtime=onnx but no bundle — run `optimize` + `package` first, "
                "or set serving.runtime: torch"
            )
        return _OnnxModel(bundle)
    return _TorchModel()


def create_app(bundle=None):
    from fastapi import FastAPI
    from fastapi.responses import Response
    from prometheus_client import CONTENT_TYPE_LATEST, Histogram, generate_latest
    from pydantic import BaseModel

    model = _load_model(bundle)
    latency = Histogram("predict_seconds", "predict latency")
    app = FastAPI(title="text-classifier")

    class Req(BaseModel):
        text: str | None = None
        texts: list[str] | None = None

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict")
    def predict(req: Req):
        with latency.time():
            out = model.predict(req.texts if req.texts else [req.text])
        return out if req.texts is not None else out[0]

    return app


def serve(port=8080, bundle=None):
    import uvicorn

    log.info(f"serving on :{port} (FastAPI + ONNX Runtime)")
    uvicorn.run(create_app(bundle), host="0.0.0.0", port=port)
