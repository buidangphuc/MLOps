"""Typed config objects (knobs only — artifact paths live in constants/paths.py)."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IngestSource:
    """One input file, optionally filtered/sampled before merging."""
    file: str
    require_labels: Optional[list]   # keep only rows whose ner has ≥1 of these types (None = all)
    sample_size: Optional[int]       # cap rows taken from THIS source (after filtering); None = all


@dataclass
class DataIngestionConfig:
    sources: list                # list[IngestSource]; >1 = merge. Empty = skip (use existing data).
    sample_size: Optional[int]   # global cap on the merged pool (None = all)
    seed: int


@dataclass
class DataValidationConfig:
    text_field: str
    label_field: str
    expected_labels: Optional[list]   # None = accept whatever the data contains
    require_all_labels: bool          # fail when the data does not cover every declared type


@dataclass
class DataTransformationConfig:
    text_field: str
    label_field: str
    val_size: float
    test_size: float
    seed: int
    leakage_check: bool
    dedupe: bool
    expected_labels: Optional[list]   # if set, spans with other labels are dropped


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
class GlinerConfig:
    """GLiNER architecture + span/negative-sampling knobs.

    These have no equivalent in a text-classification schema — they're passed
    straight into GLiNERConfig when a model is built from scratch.
    """
    name: str
    max_width: int                # widest span (in tokens) the model will consider
    hidden_size: int
    dropout: float
    fine_tune: bool
    subtoken_pooling: str
    span_mode: str                # markerV0 | token_level
    words_splitter_type: str      # whitespace | bert_tokenizer (whitespace for pre-tokenized data)
    labels_encoder: Optional[str]
    fuse_layers: bool
    post_fusion_schema: Optional[str]
    # loss
    loss_alpha: float
    loss_gamma: float
    loss_prob_margin: float
    label_smoothing: float
    loss_reduction: str
    negatives: float
    masking: str
    # negative-type sampling
    size_sup: int
    max_types: int
    shuffle_types: bool
    random_drop: bool
    max_neg_type_ratio: int
    max_len: int
    freeze_token_rep: bool
    # encoder dropout overrides (applied to model.config after build)
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float


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
    text_field: str            # GLiNER: "tokenized_text"
    label_field: str           # GLiNER: "ner"
    base_model: str
    prev_path: Optional[str]   # continue from a trained checkpoint instead of base_model
    max_length: int
    epochs: int
    # GLiNER trains the pretrained encoder and the freshly-init span head at different
    # rates — that's native to its TrainingArguments, not an opt-in rung.
    lr: float                  # encoder LR (at lr_reference_batch_size)
    others_lr: float           # span head / scorer LR
    lr_scaling: str            # "sqrt" (Adam) | "linear" (SGD) | "none"
    lr_reference_batch_size: int   # the effective batch lr/others_lr were tuned at
    weight_decay: float        # encoder weight decay
    others_weight_decay: float
    scheduler_type: str
    max_grad_norm: float
    train_batch_size: int
    eval_batch_size: int
    warmup_ratio: float
    save_steps: int
    eval_steps: int
    logging_steps: int
    save_total_limit: int
    num_workers: int
    seed: int
    max_steps: int
    oom_auto_retry: bool
    early_stopping_patience: int      # stop after N evals with no eval_loss improvement; 0 = off
    early_stopping_threshold: float   # min eval_loss improvement to count as progress
    objective: str             # metric_for_best_model
    tracking_backend: str
    experiment_name: str
    mlflow_uri: str
    tensorboard: bool          # also write TB logs alongside the backend
    log_histograms: bool       # TB-only: weight + gradient histograms (implies tensorboard)
    gliner: "GlinerConfig"


@dataclass
class ModelEvaluationConfig:
    text_field: str
    label_field: str
    max_width: int
    compute: list             # which metrics to compute (subset of f1/precision/recall)
    objective: str
    per_class: bool           # per-entity-type P/R/F1
    error_dump: bool
    bootstrap_ci: bool
    flat_ner: bool            # non-overlapping spans when decoding
    threshold_sweep: list     # candidate decision thresholds, swept on VAL
    sweep_sample_size: Optional[int]   # cap val rows used for the sweep (None = all)
    per_entity_threshold: bool         # one threshold per entity type (matches serving)
    eval_batch_size: int
    load_in_8bit: bool
    test_file: Optional[str]           # external shared test set; None => internal TEST_FILE split


@dataclass
class GateConfig:
    enabled: bool
    on_fail: str              # "error" | "warn"
    rules: list               # [{metric, min}, ...]


@dataclass
class RegistryConfig:
    name: str
    remote_backend: str       # none | mlflow | sagemaker | s3
    remote_s3: dict = None    # {bucket, prefix, region} when remote_backend == s3
