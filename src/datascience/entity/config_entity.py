"""Typed config objects (knobs only — artifact paths live in constants/paths.py)."""

from dataclasses import dataclass
from typing import Optional


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
    """The classification task posed as generation (see infer/scoring.py)."""
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


@dataclass
class GateConfig:
    enabled: bool
    on_fail: str                  # "error" | "warn"
    rules: list                   # [{metric, min}, ...]


@dataclass
class RegistryConfig:
    name: str
    remote_backend: str           # none | mlflow | sagemaker | s3
    remote_s3: dict = None        # {bucket, prefix, region} when remote_backend == s3
