"""Fixed artifact paths + MLflow URI — convention, not user-tweakable.

Everything is absolute (derived from the project root), so the pipeline behaves
identically no matter which directory you launch it from. Paths are per model:
`config/<model_id>/` + `artifacts/<model_id>/`; the CLI picks the model with `--model`
(env `DS_MODEL`) and `paths.X` resolves against it. Hardware/perf files are shared.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# .../src/datascience/constants/paths.py -> parents[3] = repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = PROJECT_ROOT / "config"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_MODEL = "text_classification"

HARDWARE_FILE = CONFIG_ROOT / "hardware.yaml"   # train/infer GPU targets (perf binding)
PERF_FILE = CONFIG_ROOT / "perf.yaml"           # perf sweep grids / objectives / constraints
# Perf platform: tuned results bound to hardware -> <PERF_DIR>/<role>/<hw_id>/<workload>.json
PERF_DIR = ARTIFACTS_ROOT / "perf"

# Tracking: hardcoded, absolute sqlite. Override silently via MLFLOW_TRACKING_URI env.
MLFLOW_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


@dataclass(frozen=True)
class ModelPaths:
    MODEL_ID: str
    CONFIG_DIR: Path = field(init=False)
    PARAMS_FILE: Path = field(init=False)
    SERVING_FILE: Path = field(init=False)
    EXPERIMENTS_FILE: Path = field(init=False)
    SAGEMAKER_FILE: Path = field(init=False)
    ARTIFACTS: Path = field(init=False)
    DATA_DIR: Path = field(init=False)
    RAW_DATA: Path = field(init=False)
    VALID_DIR: Path = field(init=False)
    STATUS_FILE: Path = field(init=False)
    SPLIT_DIR: Path = field(init=False)
    TRAIN_FILE: Path = field(init=False)
    VAL_FILE: Path = field(init=False)
    TEST_FILE: Path = field(init=False)
    LABEL_MAP: Path = field(init=False)
    DATA_VERSION: Path = field(init=False)
    TRAINER_DIR: Path = field(init=False)
    MODEL_DIR: Path = field(init=False)
    CKPT_DIR: Path = field(init=False)
    TRAIN_STATS: Path = field(init=False)
    TRAIN_RUN_FILE: Path = field(init=False)
    EVAL_DIR: Path = field(init=False)
    METRICS_FILE: Path = field(init=False)
    SWEEP_FILE: Path = field(init=False)
    THRESHOLD_FILE: Path = field(init=False)
    REGISTRY_DIR: Path = field(init=False)
    REGISTRY_CACHE: Path = field(init=False)
    RUN_STATUS: Path = field(init=False)
    TUNE_DIR: Path = field(init=False)
    TUNE_RESULTS: Path = field(init=False)
    DAPT_DIR: Path = field(init=False)
    TRITON_REPO: Path = field(init=False)
    MME_TARS: Path = field(init=False)

    def __post_init__(self):
        cfg = CONFIG_ROOT / self.MODEL_ID
        art = ARTIFACTS_ROOT / self.MODEL_ID
        split = art / "data_transformation"
        trainer = art / "model_trainer"
        model = trainer / "model"
        ev = art / "model_evaluation"
        tune = art / "tune_runs"
        values = {
            "CONFIG_DIR": cfg,
            "PARAMS_FILE": cfg / "params.yaml",
            "SERVING_FILE": cfg / "serving.yaml",
            "EXPERIMENTS_FILE": cfg / "experiments.yaml",
            "SAGEMAKER_FILE": cfg / "sagemaker.yaml",
            "ARTIFACTS": art,
            "DATA_DIR": art / "data_ingestion",
            "RAW_DATA": art / "data_ingestion" / "data.jsonl",
            "VALID_DIR": art / "data_validation",
            "STATUS_FILE": art / "data_validation" / "status.txt",
            "SPLIT_DIR": split,
            "TRAIN_FILE": split / "train.jsonl",
            "VAL_FILE": split / "val.jsonl",
            "TEST_FILE": split / "test.jsonl",
            "LABEL_MAP": split / "label_map.json",
            "DATA_VERSION": split / "data_version.json",
            "TRAINER_DIR": trainer,
            "MODEL_DIR": model,
            "CKPT_DIR": trainer / "checkpoints",
            "TRAIN_STATS": trainer / "train_stats.json",
            "TRAIN_RUN_FILE": trainer / "mlflow_run.json",
            "EVAL_DIR": ev,
            "METRICS_FILE": ev / "metrics.json",
            "SWEEP_FILE": ev / "threshold_sweep.json",
            "THRESHOLD_FILE": model / "threshold.json",
            "REGISTRY_DIR": art / "registry",
            "REGISTRY_CACHE": art / "registry_cache",
            "RUN_STATUS": art / "run_status.json",
            "TUNE_DIR": tune,
            "TUNE_RESULTS": tune / "results.json",
            "DAPT_DIR": art / "dapt",
            "TRITON_REPO": art / "model_repository",
            "MME_TARS": art / "mme_tars",
        }
        for k, v in values.items():
            object.__setattr__(self, k, v)

    def ensure_dirs(self) -> None:
        for d in (self.DATA_DIR, self.VALID_DIR, self.SPLIT_DIR, self.TRAINER_DIR, self.CKPT_DIR,
                  self.EVAL_DIR, self.REGISTRY_DIR, LOG_DIR, self.TUNE_DIR):
            d.mkdir(parents=True, exist_ok=True)


def for_model(model_id: str) -> ModelPaths:
    if not (CONFIG_ROOT / model_id).is_dir():
        known = sorted(p.name for p in CONFIG_ROOT.iterdir() if p.is_dir())
        raise SystemExit(f"unknown model '{model_id}': no config/{model_id}/ (known: {', '.join(known)})")
    return ModelPaths(model_id)


_active = None


def use(model_id: str) -> ModelPaths:
    global _active
    _active = for_model(model_id)
    return _active


def active() -> ModelPaths:
    return _active or use(os.getenv("DS_MODEL", DEFAULT_MODEL))


def ensure_dirs() -> None:
    active().ensure_dirs()


def __getattr__(name):
    if name in ModelPaths.__dataclass_fields__:
        return getattr(active(), name)
    raise AttributeError(name)
