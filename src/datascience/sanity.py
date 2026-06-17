"""Sanity pre-flight checks (opt-in debugging).

Two classic checks that catch model/data/loss bugs BEFORE a long run:
  1. loss@init should be ~ ln(num_classes) for balanced cross-entropy.
  2. the model must be able to overfit a single tiny batch (loss drops sharply);
     if it can't, something is wrong (bad labels, frozen weights, wrong loss).
"""

import math

from src.datascience import log
from src.datascience.constants import paths
from src.datascience.entity.config_entity import ModelTrainerConfig
from src.datascience.model import build_for_training, load_label_map
from src.datascience.utils.io import read_jsonl
from src.datascience.utils.runtime import pick_device, set_seed


def run_sanity(cfg: ModelTrainerConfig, n: int = 16, steps: int = 80, lr: float = 1e-4) -> bool:
    import torch

    set_seed(cfg.seed)
    device = pick_device()
    label2id, _ = load_label_map(paths.LABEL_MAP)

    rows = [
        r
        for r in read_jsonl(paths.TRAIN_FILE, required_fields=[cfg.text_field, cfg.label_field])
        if r[cfg.label_field] in label2id
    ][:n]

    model, tokenizer = build_for_training(cfg.base_model, label2id, {v: k for k, v in label2id.items()})
    model.to(device).train()
    enc = tokenizer(
        [r[cfg.text_field] for r in rows],
        truncation=True,
        max_length=cfg.max_length,
        padding=True,
        return_tensors="pt",
    ).to(device)
    labels = torch.tensor([label2id[r[cfg.label_field]] for r in rows]).to(device)

    # 1) loss @ init ~ ln(C)
    with torch.no_grad():
        loss0 = model(**enc, labels=labels).loss.item()
    expected = math.log(len(label2id))
    init_ok = abs(loss0 - expected) < 0.7
    log.info(f"sanity loss@init = {loss0:.3f} (expected ~ ln(C) = {expected:.3f}) -> {'OK' if init_ok else 'SUSPECT'}")

    # 2) overfit one batch
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    final = loss0
    for _ in range(steps):
        optimizer.zero_grad()
        out = model(**enc, labels=labels)
        out.loss.backward()
        optimizer.step()
        final = out.loss.item()
    overfit_ok = final < loss0 * 0.5
    log.info(
        f"sanity overfit-one-batch: loss {loss0:.3f} -> {final:.3f} after {steps} steps "
        f"-> {'OK' if overfit_ok else 'SUSPECT'}"
    )

    ok = init_ok and overfit_ok
    if ok:
        log.info("SANITY PASSED — model learns and init loss is sane")
    else:
        log.warning("SANITY SUSPECT — check data labels / model / loss before a full run")
    return ok
