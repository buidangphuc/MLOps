"""HF model + tokenizer factory.

One place that builds/loads the model, shared by trainer / eval / infer so the
architecture and label mapping can never drift. To use a different architecture,
change `base_model` in params.yaml — the code only assumes a HF
SequenceClassification model.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

from pathlib import Path


def load_label_map(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    label2id = obj["label2id"]
    id2label = {int(k): v for k, v in obj["id2label"].items()}
    return label2id, id2label


def build_for_training(base_model: str, label2id: dict, id2label: dict):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    return model, tokenizer


def load_trained(model_dir, device: str = "cpu"):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    return model, tokenizer


def save_pretrained_atomic(model, tokenizer, model_dir) -> None:
    """Save model+tokenizer to a temp dir then swap in — never a half-written model dir."""
    model_dir = Path(model_dir)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=model_dir.parent, prefix=".model_tmp_"))
    model.save_pretrained(tmp)
    tokenizer.save_pretrained(tmp)
    if model_dir.exists():
        shutil.rmtree(model_dir)
    os.replace(tmp, model_dir)
