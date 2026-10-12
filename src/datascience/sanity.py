"""Sanity pre-flight check (opt-in debugging).

The model must be able to overfit a single tiny batch (completion loss drops sharply); if it
can't, something is wrong (bad labels, frozen weights, a prompt that masks the whole example).
"""

from datascience import log
from datascience.constants import paths
from datascience.entity.config_entity import ModelTrainerConfig
from datascience.infer.scoring import encode_example, pad_batch
from datascience.model import build_for_training, load_label_map
from datascience.utils.io import read_jsonl
from datascience.utils.runtime import pick_device, set_seed


def run_sanity(cfg: ModelTrainerConfig, n: int = 16, steps: int = 40, lr: float = 1e-4) -> bool:
    import torch

    set_seed(cfg.seed)
    device = pick_device()
    label2id, _ = load_label_map(paths.LABEL_MAP)
    labels = list(label2id)

    rows = [
        r
        for r in read_jsonl(paths.TRAIN_FILE, required_fields=[cfg.text_field, cfg.label_field])
        if r[cfg.label_field] in label2id
    ][:n]

    model, tokenizer = build_for_training(cfg)
    model.to(device).train()
    batch = pad_batch(
        [encode_example(tokenizer, cfg.task, r[cfg.text_field], r[cfg.label_field], labels, cfg.max_length) for r in rows],
        tokenizer.pad_token_id,
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    first = final = None
    for _ in range(steps):
        optimizer.zero_grad()
        out = model(**batch)
        out.loss.backward()
        optimizer.step()
        first = out.loss.item() if first is None else first
        final = out.loss.item()
    ok = final < first * 0.5
    log.info(f"sanity overfit-one-batch: loss {first:.3f} -> {final:.3f} after {steps} steps -> {'OK' if ok else 'SUSPECT'}")
    if ok:
        log.info("SANITY PASSED — model learns")
    else:
        log.warning("SANITY SUSPECT — check data labels / model / prompt before a full run")
    return ok
