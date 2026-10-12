import math
import os

from datascience import log
from datascience.constants import paths
from datascience.entity.config_entity import ModelTrainerConfig
from datascience.infer.scoring import encode_example, pad_batch
from datascience.model import bf16_ok, build_for_training, load_label_map, merge_adapters, save_pretrained_atomic
from datascience.utils.io import read_jsonl
from datascience.utils.runtime import is_oom_error, set_run_status, set_seed


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


def training_kwargs(cfg: ModelTrainerConfig, batch_size: int, report_to: list) -> dict:
    """TrainingArguments kwargs. With `perf.apply_tuned` the throughput config tuned for this
    box's train role is merged over them; a result tuned on another GPU raises HardwareMismatch."""
    kwargs = dict(
        output_dir=str(paths.CKPT_DIR),
        learning_rate=cfg.lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_ratio,   # a float in [0, 1) is a ratio of total steps
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=20,
        seed=cfg.seed,
        report_to=report_to,
        bf16=bf16_ok(),
        remove_unused_columns=False,   # the collator takes our pre-encoded columns as they are
    )
    if cfg.max_steps and cfg.max_steps > 0:
        kwargs["max_steps"] = cfg.max_steps
    else:
        kwargs["num_train_epochs"] = cfg.epochs
    if cfg.apply_tuned:
        from datascience.perf.store import load_tuned
        from datascience.perf.workloads.train import training_args

        kwargs.update(training_args(load_tuned("train", "train")))
    return kwargs


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
        if cfg.finetune.method == "qat":
            raise NotImplementedError("group 3")

        from datasets import Dataset
        from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

        label2id, _ = load_label_map(paths.LABEL_MAP)
        labels = list(label2id)   # candidate labels = TRAIN-only label map
        model, tokenizer = build_for_training(cfg)

        def to_ds(path):
            rows = read_jsonl(path, required_fields=[cfg.text_field, cfg.label_field])
            kept = [r for r in rows if r[cfg.label_field] in label2id]
            if len(kept) < len(rows):
                log.warning(f"{path.name}: dropped {len(rows) - len(kept)} rows with unseen label")
            return Dataset.from_list([
                encode_example(tokenizer, cfg.task, r[cfg.text_field], r[cfg.label_field], labels, cfg.max_length)
                for r in kept
            ])

        train_ds, val_ds = to_ds(paths.TRAIN_FILE), to_ds(paths.VAL_FILE)
        pad_id = tokenizer.pad_token_id

        report_to = []
        if cfg.tracking_backend == "mlflow":
            os.environ["MLFLOW_TRACKING_URI"] = cfg.mlflow_uri
            os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg.experiment_name
            report_to.append("mlflow")
        # TB can be the primary backend OR run alongside mlflow (opt-in); histograms need it.
        if cfg.tracking_backend == "tensorboard" or cfg.tensorboard or cfg.log_histograms:
            report_to.append("tensorboard")
            os.environ["TENSORBOARD_LOGGING_DIR"] = str(paths.LOG_DIR / "tensorboard")

        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=cfg.patience),
            _nan_guard_callback(),
        ]
        if cfg.log_histograms:
            callbacks.append(_histogram_callback(paths.LOG_DIR / "tensorboard", every=20))

        trainer = Trainer(
            model=model,
            args=TrainingArguments(**training_kwargs(cfg, batch_size, report_to)),
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=lambda features: pad_batch(features, pad_id),
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
        save_pretrained_atomic(merge_adapters(trainer.model, cfg), tokenizer, paths.MODEL_DIR)
        log.info(f"best model saved -> {paths.MODEL_DIR} (method={cfg.finetune.method})")
