import os

from datascience import pack
from datascience.constants import paths
from datascience.entity.config_entity import GateConfig, RegistryConfig
from datascience.utils.common import read_yaml

_REQUIRED_SECTIONS = ["model", "metrics", "gate", "tracking", "registry"]


class ConfigurationManager:
    """Reads the model's params.yaml, validates it fail-fast, and hands out the common
    configs. Pack-specific knobs are read by the pack's own config module.
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

        valid = pack.load().METRICS
        m = self.params.metrics
        unknown = set(m.compute) - valid
        if unknown:
            raise ValueError(f"metrics.compute has unknown metrics {unknown}; valid: {sorted(valid)}")
        if m.objective not in m.compute:
            raise ValueError(f"metrics.objective '{m.objective}' must be one of metrics.compute {list(m.compute)}")

        if self.params.gate.on_fail not in ("error", "warn"):
            raise ValueError("gate.on_fail must be 'error' or 'warn'")
        for r in self.params.gate.rules:
            if "metric" not in r or "min" not in r:
                raise ValueError(f"each gate rule needs 'metric' and 'min': {dict(r)}")

        validate = getattr(pack.load(), "validate_config", None)
        if validate:
            validate(self.params)

    def tracking_uri(self) -> str:
        # Env wins so CI/prod can point elsewhere; default is the fixed sqlite.
        return os.getenv("MLFLOW_TRACKING_URI", paths.MLFLOW_URI)

    def gate(self) -> GateConfig:
        g = self.params.gate
        return GateConfig(enabled=g.enabled, on_fail=g.on_fail, rules=[dict(r) for r in g.rules])

    def registry(self) -> RegistryConfig:
        r = self.params.registry
        s3 = dict(r.remote.s3) if r.remote.get("s3") else None
        alias = r.remote.get("mlflow", {}).get("alias", "champion")
        return RegistryConfig(name=r.name, remote_backend=r.remote.backend, remote_s3=s3,
                              tracking_uri=self.tracking_uri(),
                              experiment_name=self.params.tracking.experiment_name, alias=alias)

    def serving(self):
        """The serving.yaml ConfigBox. Loaded lazily so the training pipeline never touches it."""
        if self._serving is None:
            if not os.path.exists(paths.SERVING_FILE):
                raise FileNotFoundError(f"serving config not found: {paths.SERVING_FILE}")
            self._serving = read_yaml(paths.SERVING_FILE)
        return self._serving
