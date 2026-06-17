"""Inference: single text, batch, or batch-over-file (resumable).

Loads the saved model + tokenizer + label map, and applies the calibration
temperature (if present) so scores are trustworthy. Batched forward passes for
throughput; bad input rows are skipped, not fatal.
"""

import json

from pathlib import Path

from src.datascience import log
from src.datascience.config.configuration import ConfigurationManager
from src.datascience.constants import paths
from src.datascience.metrics import softmax
from src.datascience.model import load_label_map, load_trained
from src.datascience.utils.runtime import pick_device


class Predictor:

    def __init__(self, model_dir=None, device=None, max_length=None):
        ev = ConfigurationManager().evaluation()
        self.text_field = ev.text_field
        self.max_length = max_length or ev.max_length
        self.device = device or pick_device()

        model_dir = Path(model_dir or paths.MODEL_DIR)
        self.model, self.tokenizer = load_trained(model_dir, self.device)
        _, self.id2label = load_label_map(paths.LABEL_MAP)

        temp_file = model_dir / "temperature.json"
        self.temperature = (
            json.loads(temp_file.read_text())["temperature"] if temp_file.exists() else 1.0
        )

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts, batch_size: int = 32) -> list:
        import torch

        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = self.tokenizer(
                chunk, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits.cpu().numpy()
            probs = softmax(logits / self.temperature)
            for p in probs:
                idx = int(p.argmax())
                results.append({"label": self.id2label[idx], "score": float(p[idx])})
        return results

    def predict_file(self, input_file, output_file, batch_size: int = 32) -> int:
        """Predict over a JSONL file -> {..row.., label, score}. Resumable: if the
        output already has N lines, skip the first N inputs and append the rest."""
        input_file, output_file = Path(input_file), Path(output_file)
        rows = []
        bad = 0
        with open(input_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                if self.text_field not in obj:
                    bad += 1
                    continue
                rows.append(obj)
        if bad:
            log.warning(f"{input_file.name}: skipped {bad} bad/incomplete line(s)")

        done = sum(1 for _ in open(output_file, encoding="utf-8")) if output_file.exists() else 0
        if done:
            log.info(f"resuming batch infer: {done} rows already done")
        rows = rows[done:]

        output_file.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with open(output_file, "a", encoding="utf-8") as out:
            for i in range(0, len(rows), batch_size):
                chunk = rows[i : i + batch_size]
                preds = self.predict_batch([r[self.text_field] for r in chunk], batch_size)
                for r, pred in zip(chunk, preds):
                    out.write(json.dumps({**r, **pred}, ensure_ascii=False) + "\n")
                    written += 1
        log.info(f"batch infer: wrote {written} predictions -> {output_file}")
        return written
