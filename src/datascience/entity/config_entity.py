"""Typed config objects (knobs only — artifact paths live in constants/paths.py)."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DataIngestionConfig:
    dataset_name: Optional[str]
    sample_size: Optional[int]


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
    group_field: Optional[str]


@dataclass
class TuningConfig:
    """Tuning rungs applied inside the trainer (off by default = plain fine-tune)."""
    freeze_encoder_layers: int    # R3: freeze embeddings + bottom N layers (0 = none)
    differential_lr: bool         # R3: encoder_lr for the encoder, lr for the head
    encoder_lr: float
    lora: bool                    # R5: wrap the model with LoRA adapters
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: Optional[list]


@dataclass
class DaptConfig:
    """R4 domain-adaptive pretraining (a pre-stage that adapts the backbone)."""
    base_model: str
    corpus: Optional[str]
    epochs: int
    mlm_probability: float
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
    train_batch_size: int
    eval_batch_size: int
    weight_decay: float
    warmup_ratio: float
    patience: int
    save_total_limit: int
    seed: int
    max_steps: int
    oom_auto_retry: bool
    compute: list             # metrics to compute each eval (must include objective)
    objective: str            # metric_for_best_model
    tracking_backend: str
    experiment_name: str
    mlflow_uri: str
    tensorboard: bool          # also write TB logs alongside the backend
    log_histograms: bool       # TB-only: weight + gradient histograms (implies tensorboard)
    tuning: "TuningConfig"     # R3/R5 rungs (off by default)


@dataclass
class ModelEvaluationConfig:
    text_field: str
    label_field: str
    max_length: int
    eval_batch_size: int
    compute: list             # which metrics to compute
    objective: str
    per_class: bool
    error_dump: bool
    bootstrap_ci: bool
    calibration: bool
    slice_field: Optional[str]


@dataclass
class GateConfig:
    enabled: bool
    on_fail: str              # "error" | "warn"
    rules: list               # [{metric, min}, ...]


@dataclass
class RegistryConfig:
    name: str
    remote_backend: str       # none | mlflow | sagemaker
