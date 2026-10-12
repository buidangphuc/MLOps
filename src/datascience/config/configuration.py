import os

from datascience.constants import paths
from datascience.entity.config_entity import (
    DaptConfig,
    DataIngestionConfig,
    IngestSource,
    DataTransformationConfig,
    DataValidationConfig,
    GateConfig,
    GlinerConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
    RegistryConfig,
    TuningConfig,
)
from datascience.utils.common import read_yaml

_REQUIRED_SECTIONS = ["ingestion", "data", "model", "training", "metrics", "gate", "tracking", "registry", "gliner"]
# Span-level NER metrics (set-based over (example, entity_type) pairs), not classification ones.
_VALID_METRICS = {"f1", "precision", "recall"}


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
        # test_size may be 0 (test comes from a separate data.test_file); val_size must be > 0.
        if not (0 < d.val_size < 1 and 0 <= d.test_size < 1 and d.val_size + d.test_size < 1):
            raise ValueError("data.val_size in (0,1), test_size in [0,1), and their sum < 1")

        m = self.params.metrics
        unknown = set(m.compute) - _VALID_METRICS
        if unknown:
            raise ValueError(f"metrics.compute has unknown metrics {unknown}; valid: {_VALID_METRICS}")
        if m.objective not in m.compute:
            raise ValueError(f"metrics.objective '{m.objective}' must be one of metrics.compute {list(m.compute)}")

        if self.params.training.get("lr_scaling", "none") not in ("none", "sqrt", "linear"):
            raise ValueError("training.lr_scaling must be 'none', 'sqrt', or 'linear'")

        sweep = m.get("threshold_sweep", [0.5])
        if not sweep or any(not 0 < float(t) < 1 for t in sweep):
            raise ValueError("metrics.threshold_sweep must be a non-empty list of values in (0,1)")

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
        src = c.get("source_file")
        # Two accepted shapes, normalized to a list[IngestSource]:
        #   source_file: <path> | [<path>, ...]     — sugar, whole file(s), no per-source filter
        #   sources: [{file, require_labels?, sample_size?}, ...]  — rich, per-source filter/sample
        raw = c.get("sources")
        if raw is None:
            raw = [] if not src else ([src] if isinstance(src, str) else list(src))
            raw = [{"file": f} for f in raw]
        sources = [
            IngestSource(
                file=s["file"],
                require_labels=list(s["require_labels"]) if s.get("require_labels") else None,
                sample_size=s.get("sample_size"),
            )
            for s in raw
        ]
        return DataIngestionConfig(
            sources=sources,
            sample_size=c.get("sample_size"),
            seed=self.params.data.seed,
        )

    def validation(self) -> DataValidationConfig:
        d = self.params.data
        expected = d.get("expected_labels")
        return DataValidationConfig(
            text_field=d.text_field,
            label_field=d.label_field,
            expected_labels=list(expected) if expected else None,
            require_all_labels=d.get("require_all_labels", True),
        )

    def transformation(self) -> DataTransformationConfig:
        d = self.params.data
        return DataTransformationConfig(
            text_field=d.text_field,
            label_field=d.label_field,
            val_size=d.val_size,
            test_size=d.test_size,
            seed=d.seed,
            leakage_check=d.get("leakage_check", True),
            dedupe=d.get("dedupe", True),
            expected_labels=list(d.get("expected_labels") or []) or None,
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

    def gliner(self) -> GlinerConfig:
        """GLiNER architecture knobs. `max_len` comes from model.max_length so the
        sequence limit is stated once."""
        g, m = self.params.gliner, self.params.model
        return GlinerConfig(
            name=g.get("name", "span level gliner"),
            max_width=g.max_width,
            hidden_size=g.hidden_size,
            dropout=g.dropout,
            fine_tune=g.get("fine_tune", True),
            subtoken_pooling=g.get("subtoken_pooling", "first"),
            span_mode=g.span_mode,
            words_splitter_type=g.get("words_splitter_type", "whitespace"),
            labels_encoder=g.get("labels_encoder"),
            fuse_layers=g.get("fuse_layers", False),
            post_fusion_schema=g.get("post_fusion_schema"),
            loss_alpha=g.get("loss_alpha", -1),
            loss_gamma=g.get("loss_gamma", 0),
            loss_prob_margin=g.get("loss_prob_margin", 0),
            label_smoothing=g.get("label_smoothing", 0),
            loss_reduction=g.get("loss_reduction", "sum"),
            negatives=g.get("negatives", 1.0),
            masking=g.get("masking", "none"),
            size_sup=g.get("size_sup", -1),
            max_types=g.get("max_types", 12),
            shuffle_types=g.get("shuffle_types", True),
            random_drop=g.get("random_drop", True),
            max_neg_type_ratio=g.get("max_neg_type_ratio", 1),
            max_len=m.max_length,
            freeze_token_rep=g.get("freeze_token_rep", False),
            hidden_dropout_prob=g.get("hidden_dropout_prob", 0.15),
            attention_probs_dropout_prob=g.get("attention_probs_dropout_prob", 0.15),
        )

    def trainer(self) -> ModelTrainerConfig:
        d, m, t = self.params.data, self.params.model, self.params.training
        return ModelTrainerConfig(
            text_field=d.text_field,
            label_field=d.label_field,
            base_model=m.base_model,
            prev_path=m.get("prev_path"),
            max_length=m.max_length,
            epochs=t.epochs,
            lr=float(t.lr),
            others_lr=float(t.others_lr),
            lr_scaling=t.get("lr_scaling", "none"),
            lr_reference_batch_size=int(t.get("lr_reference_batch_size", t.train_batch_size)),
            weight_decay=float(t.weight_decay),
            others_weight_decay=float(t.get("others_weight_decay", t.weight_decay)),
            scheduler_type=t.get("scheduler_type", "cosine"),
            max_grad_norm=float(t.get("max_grad_norm", 1.0)),
            train_batch_size=t.train_batch_size,
            eval_batch_size=t.eval_batch_size,
            warmup_ratio=t.warmup_ratio,
            save_steps=t.get("save_steps", 1000),
            eval_steps=t.get("eval_steps", 1000),
            logging_steps=t.get("logging_steps", 100),
            save_total_limit=t.save_total_limit,
            num_workers=t.get("num_workers", 1),
            seed=t.seed,
            max_steps=t.get("max_steps", -1),
            oom_auto_retry=t.get("oom_auto_retry", True),
            early_stopping_patience=int(t.get("early_stopping_patience", 0)),
            early_stopping_threshold=float(t.get("early_stopping_threshold", 0.0)),
            objective=self.params.metrics.objective,
            tracking_backend=self.params.tracking.backend,
            experiment_name=self.params.tracking.experiment_name,
            mlflow_uri=self.tracking_uri(),
            tensorboard=self.params.tracking.get("tensorboard", False),
            log_histograms=self.params.tracking.get("log_histograms", False),
            gliner=self.gliner(),
        )

    def evaluation(self) -> ModelEvaluationConfig:
        d, m = self.params.data, self.params.metrics
        return ModelEvaluationConfig(
            text_field=d.text_field,
            label_field=d.label_field,
            max_width=self.params.gliner.max_width,
            compute=list(m.compute),
            objective=m.objective,
            per_class=m.get("per_class", True),
            error_dump=m.get("error_dump", True),
            bootstrap_ci=m.get("bootstrap_ci", True),
            flat_ner=m.get("flat_ner", True),
            threshold_sweep=[float(t) for t in m.get("threshold_sweep", [0.5])],
            sweep_sample_size=m.get("sweep_sample_size"),
            per_entity_threshold=m.get("per_entity_threshold", True),
            eval_batch_size=self.params.training.eval_batch_size,
            load_in_8bit=m.get("load_in_8bit", False),
            test_file=d.get("test_file"),   # external, shared test set; None => internal split
        )

    def gate(self) -> GateConfig:
        g = self.params.gate
        return GateConfig(enabled=g.enabled, on_fail=g.on_fail, rules=[dict(r) for r in g.rules])

    def registry(self) -> RegistryConfig:
        r = self.params.registry
        s3 = dict(r.remote.s3) if r.remote.get("s3") else None
        return RegistryConfig(name=r.name, remote_backend=r.remote.backend, remote_s3=s3)

    def serving(self):
        """The serving.yaml ConfigBox (.optimize / .triton / .sagemaker / .local).
        Loaded lazily so the training pipeline never touches it."""
        if self._serving is None:
            if not os.path.exists(paths.SERVING_FILE):
                raise FileNotFoundError(f"serving config not found: {paths.SERVING_FILE}")
            self._serving = read_yaml(paths.SERVING_FILE)
        return self._serving
