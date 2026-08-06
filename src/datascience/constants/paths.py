"""Fixed artifact paths + MLflow URI — convention, not user-tweakable.

Everything is absolute (derived from the project root), so the pipeline behaves
identically no matter which directory you launch it from.
"""

from pathlib import Path

# .../src/datascience/constants/paths.py -> parents[3] = repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_DIR = PROJECT_ROOT / "config"
PARAMS_FILE = CONFIG_DIR / "params.yaml"            # base experiment config (train + tuning)
SERVING_FILE = CONFIG_DIR / "serving.yaml"          # serving lifecycle config (Phase 2)
EXPERIMENTS_FILE = CONFIG_DIR / "experiments.yaml"  # the tuning loop: named variants

ARTIFACTS = PROJECT_ROOT / "artifacts"
DATA_DIR = ARTIFACTS / "data_ingestion"
RAW_DATA = DATA_DIR / "data.jsonl"
VALID_DIR = ARTIFACTS / "data_validation"
STATUS_FILE = VALID_DIR / "status.txt"
SPLIT_DIR = ARTIFACTS / "data_transformation"
TRAIN_FILE = SPLIT_DIR / "train.jsonl"
VAL_FILE = SPLIT_DIR / "val.jsonl"
TEST_FILE = SPLIT_DIR / "test.jsonl"
LABEL_MAP = SPLIT_DIR / "label_map.json"      # GLiNER: the entity-type list, derived from TRAIN
DATA_VERSION = SPLIT_DIR / "data_version.json"   # content hash + row count -> lineage
TRAINER_DIR = ARTIFACTS / "model_trainer"
MODEL_DIR = TRAINER_DIR / "model"
CKPT_DIR = TRAINER_DIR / "checkpoints"
EVAL_DIR = ARTIFACTS / "model_evaluation"
METRICS_FILE = EVAL_DIR / "metrics.json"
SWEEP_FILE = EVAL_DIR / "threshold_sweep.json"   # per-threshold P/R/F1 on VAL (the sweep curve)
THRESHOLD_FILE = MODEL_DIR / "threshold.json"    # threshold picked on VAL; infer/eval reuse it
REGISTRY_DIR = ARTIFACTS / "registry"
REGISTRY_CACHE = ARTIFACTS / "registry_cache"  # serving pulls weights from the S3 registry here
RUN_STATUS = ARTIFACTS / "run_status.json"
LOG_DIR = PROJECT_ROOT / "logs"

# Tuning loop (Layer 2): one subdir per variant (model + metrics + config snapshot)
TUNE_DIR = ARTIFACTS / "tune_runs"
TUNE_RESULTS = TUNE_DIR / "results.json"       # ranked summary the `promote` command reads
DAPT_DIR = ARTIFACTS / "dapt"                  # R4 domain-adapted backbone

# Serving / inference (Phase 2)
OPTIMIZED_DIR = ARTIFACTS / "optimized"        # ONNX (+ quantized) export
PARITY_FILE = OPTIMIZED_DIR / "parity.json"    # torch-vs-onnx parity-gate result
BUNDLE_DIR = ARTIFACTS / "bundle"              # self-contained deployable bundle
TRITON_REPO = ARTIFACTS / "model_repository"   # Triton python-backend MME repo (per model)
MME_TARS = ARTIFACTS / "mme_tars"              # per-model .tar.gz for SageMaker MME upload

# Tracking: hardcoded, absolute sqlite. Override silently via MLFLOW_TRACKING_URI env.
MLFLOW_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


def ensure_dirs() -> None:
    for d in (DATA_DIR, VALID_DIR, SPLIT_DIR, TRAINER_DIR, CKPT_DIR, EVAL_DIR, REGISTRY_DIR, LOG_DIR, TUNE_DIR, OPTIMIZED_DIR, BUNDLE_DIR):
        d.mkdir(parents=True, exist_ok=True)
