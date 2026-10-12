"""DAPT — domain-adaptive pretraining.

Continue causal-LM training of the backbone on your *unlabeled* domain text, then
fine-tune from the adapted backbone (point a variant's model.base_model at artifacts/dapt).
Helps most when your text is far from the backbone's pretraining (jargon, a new register).
Heavy → run on the GPU box. Off the default path; you invoke it explicitly.
"""

from pathlib import Path

from datascience import log
from datascience.config.configuration import ConfigurationManager
from datascience.constants import paths
from datascience.entity.config_entity import DaptConfig
from datascience.model import bf16_ok, save_pretrained_atomic
from datascience.utils.io import read_jsonl


def run_dapt(cfg: DaptConfig = None):
    cfg = cfg or ConfigurationManager().dapt()
    if not cfg.corpus:
        raise SystemExit("set dapt.corpus (a JSONL with a `text` field) before running dapt")

    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    rows = read_jsonl(Path(cfg.corpus), required_fields=["text"])
    log.info(f"DAPT: continuing causal-LM on {len(rows)} domain docs from {cfg.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model)
    ds = Dataset.from_dict({"text": [r["text"] for r in rows]}).map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=cfg.max_length),
        batched=True,
        remove_columns=["text"],
    )
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=str(paths.DAPT_DIR / "_ckpt"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        seed=cfg.seed,
        bf16=bf16_ok(),
        save_strategy="no",
        logging_steps=50,
        report_to=[],
    )
    Trainer(model=model, args=args, train_dataset=ds, data_collator=collator).train()
    save_pretrained_atomic(model, tokenizer, paths.DAPT_DIR)
    log.info(f"DAPT done -> {paths.DAPT_DIR}. Point a variant's model.base_model at this path.")
