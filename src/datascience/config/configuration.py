import os

from src.datascience.constants import paths
from src.datascience.entity.config_entity import (
    DaptConfig,
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    GateConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
    RegistryConfig,
    TuningConfig,
)
from src.datascience.utils.common import read_yaml

_REQUIRED_SECTIONS = ["ingestion", "data", "model", "training", "metrics", "gate", "tracking", "registry"]
_VALID_METRICS = {"accuracy", "macro_f1", "micro_f1", "pr_auc", "mcc"}


class ConfigurationManager:
    """Reads params.yaml, validates it fail-fast, and hands out typed configs.
    Paths are NOT here — they're fixed in constants/paths.py."""

    def __init__(self, params_filepath=None):
        params_filepath = params_filepath or paths.PARAMS_FILE
        if not os.path.exists(params_filepath):
            raise FileNotFoundError(f"params file not found: {params_filepath}")
        self.params = read_yaml(params_filepath)
        self._serving = None  # serving.yaml is loaded lazily (only the serving commands need it)
        self._validate()
        paths.ensure_dirs()

    def _validate(self):
        missing = [k for k in _REQUIRED_SECTIONS if k not in self.params]
        if missing:
            raise ValueError(f"params.yaml missing sections: {missing}")

        d = self.params.data
        if not (0 < d.val_size < 1 and 0 < d.test_size < 1 and d.val_size + d.test_size < 1):
            raise ValueError("data.val_size/test_size must each be in (0,1) and sum < 1")

        m = self.params.metrics
        unknown = set(m.compute) - _VALID_METRICS
        if unknown:
            raise ValueError(f"metrics.compute has unknown metrics {unknown}; valid: {_VALID_METRICS}")
        if m.objective not in m.compute:
            raise ValueError(f"metrics.objective '{m.objective}' must be one of metrics.compute {list(m.compute)}")

        if self.params.gate.on_fail not in ("error", "warn"):
            raise ValueError("gate.on_fail must be 'error' or 'warn'")
        for r in self.params.gate.rules:
            if "metric" not in r or "min" not in r:
                raise ValueError(f"each gate rule needs 'metric' and 'min': {dict(r)}")

    def tracking_uri(self) -> str:
        # Env wins so CI/prod can point elsewhere; default is the fixed sqlite.
        return os.getenv("MLFLOW_TRACKING_URI", paths.MLFLOW_URI)

    def ingestion(self) -> DataIngestionConfig:
        c = self.params.ingestion
        return DataIngestionConfig(dataset_name=c.get("dataset_name"), sample_size=c.get("sample_size"))

    def validation(self) -> DataValidationConfig:
        d = self.params.data
        return DataValidationConfig(text_field=d.text_field, label_field=d.label_field)

    def transformation(self) -> DataTransformationConfig:
        d = self.params.data
        return DataTransformationConfig(
            text_field=d.text_field,
            label_field=d.label_field,
            val_size=d.val_size,
            test_size=d.test_size,
            seed=d.seed,
            leakage_check=d.get("leakage_check", True),
            group_field=d.get("group_field"),
        )

    def tuning(self) -> TuningConfig:
        # The `tuning:` block is optional; absent => everything off (plain fine-tune).
        t = self.params.get("tuning", {})
        return TuningConfig(
            freeze_encoder_layers=t.get("freeze_encoder_layers", 0),
            differential_lr=t.get("differential_lr", False),
            encoder_lr=float(t.get("encoder_lr", 1e-5)),
            lora=t.get("lora", False),
            lora_r=t.get("lora_r", 8),
            lora_alpha=t.get("lora_alpha", 16),
            lora_dropout=t.get("lora_dropout", 0.05),
            lora_target_modules=t.get("lora_target_modules"),
        )

    def dapt(self) -> DaptConfig:
        d = self.params.get("tuning", {}).get("dapt", {})
        return DaptConfig(
            base_model=self.params.model.base_model,
            corpus=d.get("corpus"),
            epochs=d.get("epochs", 1),
            mlm_probability=d.get("mlm_probability", 0.15),
            max_length=self.params.model.max_length,
            train_batch_size=self.params.training.train_batch_size,
            seed=self.params.training.seed,
        )

    def trainer(self) -> ModelTrainerConfig:
        d, m, t = self.params.data, self.params.model, self.params.training
        return ModelTrainerConfig(
            text_field=d.text_field,
            label_field=d.label_field,
            base_model=m.base_model,
            max_length=m.max_length,
            epochs=t.epochs,
            lr=float(t.lr),
            train_batch_size=t.train_batch_size,
            eval_batch_size=t.eval_batch_size,
            weight_decay=t.weight_decay,
            warmup_ratio=t.warmup_ratio,
            patience=t.patience,
            save_total_limit=t.save_total_limit,
            seed=t.seed,
            max_steps=t.get("max_steps", -1),
            oom_auto_retry=t.get("oom_auto_retry", True),
            compute=list(self.params.metrics.compute),
            objective=self.params.metrics.objective,
            tracking_backend=self.params.tracking.backend,
            experiment_name=self.params.tracking.experiment_name,
            mlflow_uri=self.tracking_uri(),
            tensorboard=self.params.tracking.get("tensorboard", False),
            log_histograms=self.params.tracking.get("log_histograms", False),
            tuning=self.tuning(),
        )

    def evaluation(self) -> ModelEvaluationConfig:
        d, m = self.params.data, self.params.metrics
        return ModelEvaluationConfig(
            text_field=d.text_field,
            label_field=d.label_field,
            max_length=self.params.model.max_length,
            eval_batch_size=self.params.training.eval_batch_size,
            compute=list(m.compute),
            objective=m.objective,
            per_class=m.get("per_class", True),
            error_dump=m.get("error_dump", True),
            bootstrap_ci=m.get("bootstrap_ci", True),
            calibration=m.get("calibration", True),
            slice_field=m.get("slice_field"),
        )

    def gate(self) -> GateConfig:
        g = self.params.gate
        return GateConfig(enabled=g.enabled, on_fail=g.on_fail, rules=[dict(r) for r in g.rules])

    def registry(self) -> RegistryConfig:
        r = self.params.registry
        return RegistryConfig(name=r.name, remote_backend=r.remote.backend)

    def serving(self):
        """The serving.yaml ConfigBox (.optimize / .triton / .sagemaker / .local).
        Loaded lazily so the training pipeline never touches it."""
        if self._serving is None:
            if not os.path.exists(paths.SERVING_FILE):
                raise FileNotFoundError(f"serving config not found: {paths.SERVING_FILE}")
            self._serving = read_yaml(paths.SERVING_FILE)
        return self._serving
