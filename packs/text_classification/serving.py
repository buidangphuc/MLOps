"""Serving-side model: everything it needs sits in the model dir (`serving_meta.json` holds the
labels, the prompt and the calibration temperature), so the FastAPI app and the SageMaker
PyTorch container load it without params.yaml or the training package. Depends only on
torch, transformers and scoring.py (shipped next to this file in the DLC tar)."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    from packs.text_classification.scoring import score_labels
except ImportError:
    from scoring import score_labels

META_FILE = "serving_meta.json"


def serving_meta(labels, task, max_length, temperature=1.0) -> dict:
    return {"labels": list(labels), "system_prompt": task.system_prompt,
            "prompt_template": task.prompt_template, "max_length": max_length,
            "temperature": float(temperature)}


class ServingModel:
    def __init__(self, model_dir, device="cpu", batch_size=16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_dir = Path(model_dir)
        meta = json.loads((model_dir / META_FILE).read_text(encoding="utf-8"))
        self.labels = meta["labels"]
        self.task = SimpleNamespace(system_prompt=meta["system_prompt"], prompt_template=meta["prompt_template"])
        self.max_length = meta["max_length"]
        self.temperature = meta["temperature"]
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_dir).to(device).eval()

    def predict_batch(self, texts) -> list:
        scores = score_labels(self.model, self.tokenizer, self.task, list(texts), self.labels,
                              max_length=self.max_length, batch_size=self.batch_size) / self.temperature
        probs = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        return [{"label": self.labels[int(p.argmax())], "score": float(p.max())} for p in probs]

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]
