"""Typed configs of the text-classification pack (knobs only — paths live in constants/paths.py).

The platform's ConfigurationManager validates the common sections; these functions read
the pack's own sections from `cm.params`.
"""

from dataclasses import dataclass
from typing import Optional

_FINETUNE_METHODS = ("full", "lora", "qlora", "qat")


@dataclass
class DataIngestionConfig:
    dataset_name: Optional[str]   # HF dataset to sample; None = use the JSONL already at RAW_DATA
    sample_size: Optional[int]    # cap on rows (None = all)


@dataclass
class DataValidationConfig:
    text_field: str
    label_field: str


@dataclass
class DataTransformationConfig:
    text_field: str
    label_field: str
    val_size: float
    test_size: float
    seed: int
    leakage_check: bool
    group_field: Optional[str]    # keep rows sharing this field in one split


@dataclass
class TaskConfig:
    """The classification task posed as generation (see scoring.py)."""
    system_prompt: str
    prompt_template: str          # must contain {text}; {labels} lists the candidate labels


@dataclass
class FinetuneConfig:
    method: str                   # full | lora | qlora | qat
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: Optional[list]
    qat_scheme: Optional[str]     # the scheme a qat model is trained for


@dataclass
class DaptConfig:
    """Domain-adaptive pretraining: continued causal-LM training on unlabeled text."""
    base_model: str
    corpus: Optional[str]
    epochs: int
    max_length: int
    train_batch_size: int
    seed: int


@dataclass
class ModelTrainerConfig:
    text_field: str
    label_field: str
    base_model: str
    max_length: int
    epochs: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    train_batch_size: int
    eval_batch_size: int
    save_total_limit: int
    seed: int
    max_steps: int
    oom_auto_retry: bool
    patience: int                 # early stopping: evals without eval_loss improvement
    task: TaskConfig
    finetune: FinetuneConfig
    apply_tuned: bool             # opt-in: merge the tuned train-throughput config for this box
    tracking_backend: str
    experiment_name: str
    mlflow_uri: str
    tensorboard: bool             # also write TB logs alongside the backend
    log_histograms: bool          # TB-only: weight + gradient histograms (implies tensorboard)


@dataclass
class ModelEvaluationConfig:
    text_field: str
    label_field: str
    max_length: int
    compute: list                 # metric names, see metrics.compute_scalar_metrics
    objective: str
    per_class: bool
    error_dump: bool
    bootstrap_ci: bool
    calibration: bool
    slice_field: Optional[str]
    eval_batch_size: int
    task: TaskConfig


def validate_config(params) -> None:
    from packs.text_classification.model import QAT_SCHEMES

    missing = [k for k in ("ingestion", "data", "task", "finetune", "training") if k not in params]
    if missing:
        raise ValueError(f"params.yaml missing sections: {missing}")

    d = params.data
    if not (0 < d.val_size < 1 and 0 < d.test_size < 1 and d.val_size + d.test_size < 1):
        raise ValueError("data.val_size and test_size in (0,1), and their sum < 1")

    method = params.finetune.method
    if method not in _FINETUNE_METHODS:
        raise ValueError(f"finetune.method '{method}' is not allowed; choose one of {list(_FINETUNE_METHODS)}")
    if method == "qat":
        scheme = params.finetune.get("qat", {}).get("scheme")
        if scheme not in QAT_SCHEMES:
            raise ValueError(f"finetune.method qat needs finetune.qat.scheme in {list(QAT_SCHEMES)}, got {scheme!r}")
    if "{text}" not in params.task.prompt_template:
        raise ValueError("task.prompt_template must contain {text}")


def ingestion(cm) -> DataIngestionConfig:
    c = cm.params.ingestion
    return DataIngestionConfig(dataset_name=c.get("dataset_name"), sample_size=c.get("sample_size"))


def validation(cm) -> DataValidationConfig:
    d = cm.params.data
    return DataValidationConfig(text_field=d.text_field, label_field=d.label_field)


def transformation(cm) -> DataTransformationConfig:
    d = cm.params.data
    return DataTransformationConfig(
        text_field=d.text_field,
        label_field=d.label_field,
        val_size=d.val_size,
        test_size=d.test_size,
        seed=d.seed,
        leakage_check=d.get("leakage_check", True),
        group_field=d.get("group_field"),
    )


def task(cm) -> TaskConfig:
    t = cm.params.task
    return TaskConfig(system_prompt=t.system_prompt, prompt_template=t.prompt_template)


def finetune(cm) -> FinetuneConfig:
    f = cm.params.finetune
    lora, qat = f.get("lora", {}), f.get("qat", {})
    return FinetuneConfig(
        method=f.method,
        lora_r=lora.get("r", 16),
        lora_alpha=lora.get("alpha", 32),
        lora_dropout=lora.get("dropout", 0.05),
        lora_target_modules=lora.get("target_modules"),
        qat_scheme=qat.get("scheme"),
    )


def dapt(cm) -> DaptConfig:
    d = cm.params.get("dapt", {})
    return DaptConfig(
        base_model=cm.params.model.base_model,
        corpus=d.get("corpus"),
        epochs=d.get("epochs", 1),
        max_length=cm.params.model.max_length,
        train_batch_size=cm.params.training.train_batch_size,
        seed=cm.params.training.seed,
    )


def trainer(cm) -> ModelTrainerConfig:
    d, m, t = cm.params.data, cm.params.model, cm.params.training
    return ModelTrainerConfig(
        text_field=d.text_field,
        label_field=d.label_field,
        base_model=m.base_model,
        max_length=m.max_length,
        epochs=t.epochs,
        lr=float(t.lr),
        weight_decay=float(t.weight_decay),
        warmup_ratio=t.warmup_ratio,
        train_batch_size=t.train_batch_size,
        eval_batch_size=t.eval_batch_size,
        save_total_limit=t.save_total_limit,
        seed=t.seed,
        max_steps=t.get("max_steps", -1),
        oom_auto_retry=t.get("oom_auto_retry", True),
        patience=int(t.patience),
        task=task(cm),
        finetune=finetune(cm),
        apply_tuned=bool(cm.params.get("perf", {}).get("apply_tuned", False)),
        tracking_backend=cm.params.tracking.backend,
        experiment_name=cm.params.tracking.experiment_name,
        mlflow_uri=cm.tracking_uri(),
        tensorboard=cm.params.tracking.get("tensorboard", False),
        log_histograms=cm.params.tracking.get("log_histograms", False),
    )


def evaluation(cm) -> ModelEvaluationConfig:
    d, m = cm.params.data, cm.params.metrics
    return ModelEvaluationConfig(
        text_field=d.text_field,
        label_field=d.label_field,
        max_length=cm.params.model.max_length,
        compute=list(m.compute),
        objective=m.objective,
        per_class=m.get("per_class", True),
        error_dump=m.get("error_dump", True),
        bootstrap_ci=m.get("bootstrap_ci", True),
        calibration=m.get("calibration", True),
        slice_field=m.get("slice_field"),
        eval_batch_size=cm.params.training.eval_batch_size,
        task=task(cm),
    )
