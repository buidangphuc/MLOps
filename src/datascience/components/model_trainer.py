import math
import os

from src.datascience import log
from src.datascience.constants import paths
from src.datascience.entity.config_entity import ModelTrainerConfig
from src.datascience.model import build_for_training, load_label_map, save_pretrained_atomic
from src.datascience.utils.io import read_jsonl
from src.datascience.utils.runtime import is_oom_error, set_run_status, set_seed


def _empty_cache():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _last_checkpoint(ckpt_dir):
    from transformers.trainer_utils import get_last_checkpoint

    if os.path.isdir(ckpt_dir):
        return get_last_checkpoint(str(ckpt_dir))
    return None


def _nan_guard_callback():
    from transformers import TrainerCallback

    class NanGuard(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            loss = (logs or {}).get("loss")
            if loss is not None and (math.isnan(loss) or math.isinf(loss)):
                raise RuntimeError(f"loss is {loss} — stopping (NaN/inf guard)")

    return NanGuard()


def _histogram_callback(log_dir, every: int):
    """TB-only diagnostic: log weight + gradient histograms every `every` optimizer steps
    (spot vanishing/exploding gradients, weight drift). Off by default — opt-in via params."""
    from transformers import TrainerCallback

    class HistLogger(TrainerCallback):
        writer = None

        def on_train_begin(self, args, state, control, **kw):
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(log_dir))

        def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
            if self.writer is None or model is None or state.global_step % every != 0:
                return
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                self.writer.add_histogram(f"weights/{name}", p.detach().float().cpu(), state.global_step)
                if p.grad is not None:
                    self.writer.add_histogram(f"grads/{name}", p.grad.detach().float().cpu(), state.global_step)

        def on_train_end(self, args, state, control, **kw):
            if self.writer is not None:
                self.writer.close()

    return HistLogger()


def _bf16_ok() -> bool:
    # bf16 needs a CUDA GPU that supports it (Ampere sm_80+). Off on CPU/MPS.
    import torch

    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


# --- Tuning rungs applied to the model before training (off by default) ------------

def _apply_lora(model, t):
    """R5: replace full-weight training with small LoRA adapters."""
    from peft import LoraConfig, TaskType, get_peft_model

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=t.lora_r,
            lora_alpha=t.lora_alpha,
            lora_dropout=t.lora_dropout,
            # None -> PEFT auto-maps the attention projections for backbones it knows
            # (bert, roberta, gpt2, llama, ...). For ones it doesn't (e.g. distilbert),
            # set tuning.lora_target_modules, e.g. [q_lin, v_lin].
            target_modules=t.lora_target_modules,
        ),
    )
    model.print_trainable_parameters()
    return model


def _freeze_bottom_layers(model, n: int):
    """R3: freeze the embeddings + the bottom N transformer layers (HF naming: ...layer.<i>...)."""
    import re

    frozen = 0
    for name, p in model.named_parameters():
        m = re.search(r"\.layer\.(\d+)\.", name)
        if "embeddings" in name or (m and int(m.group(1)) < n):
            p.requires_grad = False
            frozen += 1
    log.info(f"R3 freeze: embeddings + bottom {n} layers ({frozen} tensors frozen)")


def _differential_optimizer(model, head_lr, encoder_lr, weight_decay):
    """R3: lower LR for the pretrained encoder, full LR for the freshly-init head."""
    import torch

    prefix = model.base_model_prefix  # e.g. "distilbert" / "bert" / "roberta"
    enc, head = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (enc if name.startswith(prefix) else head).append(p)
    log.info(f"R3 differential LR: encoder={encoder_lr} (prefix '{prefix}'), head={head_lr}")
    return torch.optim.AdamW(
        [{"params": enc, "lr": encoder_lr}, {"params": head, "lr": head_lr}],
        weight_decay=weight_decay,
    )


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        """Fine-tune via HF Trainer. Robust: auto-resume, NaN guard, OOM auto-retry,
        graceful interrupt (last checkpoint kept), run-status, always-save best model."""
        cfg = self.config
        set_seed(cfg.seed)
        set_run_status("train", "running")
        try:
            self._run(cfg.train_batch_size)
        except RuntimeError as e:
            if cfg.oom_auto_retry and is_oom_error(e):
                new_bs = max(1, cfg.train_batch_size // 2)
                log.warning(f"OOM at batch_size={cfg.train_batch_size} -> retry at {new_bs}")
                _empty_cache()
                try:
                    self._run(new_bs)
                except BaseException as e2:
                    set_run_status("train", "failed", error=str(e2))
                    raise
            else:
                set_run_status("train", "failed", error=str(e))
                raise
        except BaseException as e:
            set_run_status("train", "failed", error=str(e))
            raise
        set_run_status("train", "done")

    def _run(self, batch_size: int):
        cfg = self.config

        import numpy as np

        from datasets import Dataset
        from transformers import (
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )

        label2id, id2label = load_label_map(paths.LABEL_MAP)
        model, tokenizer = build_for_training(cfg.base_model, label2id, id2label)

        # Tuning rungs (R5 LoRA wrap first, then R3 freeze; differential LR -> custom optimizer)
        tun = cfg.tuning
        if tun.lora:
            model = _apply_lora(model, tun)
        if tun.freeze_encoder_layers > 0:
            _freeze_bottom_layers(model, tun.freeze_encoder_layers)
        optimizers = (None, None)
        if tun.differential_lr and not tun.lora:
            optimizers = (_differential_optimizer(model, cfg.lr, tun.encoder_lr, cfg.weight_decay), None)

        def to_ds(path):
            rows = read_jsonl(path, required_fields=[cfg.text_field, cfg.label_field])
            kept = [r for r in rows if r[cfg.label_field] in label2id]
            if len(kept) < len(rows):
                log.warning(f"{path.name}: dropped {len(rows) - len(kept)} rows with unseen label")
            ds = Dataset.from_dict(
                {
                    "text": [r[cfg.text_field] for r in kept],
                    "labels": [label2id[r[cfg.label_field]] for r in kept],
                }
            )
            return ds.map(
                lambda b: tokenizer(b["text"], truncation=True, max_length=cfg.max_length),
                batched=True,
                remove_columns=["text"],
            )

        train_ds, val_ds = to_ds(paths.TRAIN_FILE), to_ds(paths.VAL_FILE)
        collator = DataCollatorWithPadding(tokenizer)

        def compute_metrics(eval_pred):
            from src.datascience import metrics as M

            logits, labels = eval_pred
            probs = M.softmax(np.asarray(logits))
            preds = probs.argmax(axis=-1)
            # the declared metrics (incl. objective) — not a hardcoded set
            return M.compute_scalar_metrics(cfg.compute, labels, preds, probs)

        report_to = []
        if cfg.tracking_backend == "mlflow":
            os.environ["MLFLOW_TRACKING_URI"] = cfg.mlflow_uri
            os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg.experiment_name
            report_to.append("mlflow")
        # TB can be the primary backend OR run alongside mlflow (opt-in); histograms need it.
        if cfg.tracking_backend == "tensorboard" or cfg.tensorboard or cfg.log_histograms:
            report_to.append("tensorboard")

        args_kwargs = dict(
            output_dir=str(paths.CKPT_DIR),
            learning_rate=cfg.lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=cfg.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model=cfg.objective,
            greater_is_better=True,
            logging_steps=20,
            seed=cfg.seed,
            report_to=report_to,
            bf16=_bf16_ok(),
            logging_dir=str(paths.LOG_DIR / "tensorboard"),
        )
        if cfg.max_steps and cfg.max_steps > 0:
            args_kwargs["max_steps"] = cfg.max_steps
        else:
            args_kwargs["num_train_epochs"] = cfg.epochs

        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=cfg.patience),
            _nan_guard_callback(),
        ]
        if cfg.log_histograms:
            callbacks.append(_histogram_callback(paths.LOG_DIR / "tensorboard", every=20))

        args = TrainingArguments(**args_kwargs)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            compute_metrics=compute_metrics,
            optimizers=optimizers,
            callbacks=callbacks,
        )

        last_ckpt = _last_checkpoint(paths.CKPT_DIR)
        if last_ckpt:
            log.info(f"resuming from checkpoint: {last_ckpt}")
        try:
            trainer.train(resume_from_checkpoint=last_ckpt)
        except KeyboardInterrupt:
            log.warning("interrupted — last checkpoint kept for resume")
            raise

        # Trainer already restored the best model (load_best_model_at_end).
        save_pretrained_atomic(trainer.model, tokenizer, paths.MODEL_DIR)
        log.info(f"best model saved -> {paths.MODEL_DIR} (objective={cfg.objective})")
