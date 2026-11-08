"""SageMaker PyTorch inference handler. Request: {"text": "..."} or {"texts": [...]}.
Response: one {"label", "score"} per text, plus the release version from model_meta.json."""

import json
import os

from serving import ServingModel


def model_fn(model_dir):
    model = ServingModel(model_dir, device=os.environ.get("TORCH_DEVICE", "cpu"))
    with open(os.path.join(model_dir, "model_meta.json"), encoding="utf-8") as f:
        model.release_id = json.load(f)["release_id"]
    return model


def input_fn(body, content_type):
    if content_type.split(";")[0].strip() != "application/json":
        raise ValueError(f"unsupported content type: {content_type}")
    payload = json.loads(body)
    return payload["texts"] if "texts" in payload else [payload["text"]]


def predict_fn(texts, model):
    return {"predictions": model.predict_batch(texts), "model_version": model.release_id}


def output_fn(prediction, accept):
    return json.dumps(prediction, ensure_ascii=False), "application/json"
