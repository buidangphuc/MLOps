"""Model registry orchestration.

The trained model is ALWAYS on disk. On top of that, a model that passes the gate is
versioned in the LOCAL registry (always). A remote backend (mlflow / sagemaker) can
ADDITIONALLY receive it — pluggable seam, not connected yet.

To connect a remote: write a class with `.push(model_dir, metrics, metadata) -> str`
and register it in `_REMOTE_BACKENDS` below. Everything else stays the same.
"""

from typing import Protocol

from src.datascience import log
from src.datascience.constants import paths
from src.datascience.entity.config_entity import RegistryConfig
from src.datascience.registry.local import LocalRegistry


class RemoteRegistry(Protocol):
    def push(self, model_dir, metrics: dict, metadata: dict) -> str: ...


# Seam: backend name -> factory(cfg) -> RemoteRegistry. Empty = nothing connected yet.
_REMOTE_BACKENDS = {}  # e.g. {"mlflow": MlflowRegistry, "sagemaker": SageMakerRegistry}


def register(model_dir, metrics: dict, metadata: dict, cfg: RegistryConfig) -> dict:
    result = {}

    local = LocalRegistry(root=str(paths.REGISTRY_DIR), name=cfg.name)
    result["local_version"] = local.register(str(model_dir), metrics, metadata)

    backend = (cfg.remote_backend or "none").lower()
    if backend not in ("none", ""):
        factory = _REMOTE_BACKENDS.get(backend)
        if factory is None:
            log.warning(
                f"remote registry '{backend}' not connected — local copy saved only. "
                "Add a RemoteRegistry to registry.base._REMOTE_BACKENDS to connect it."
            )
        else:
            result["remote"] = factory(cfg).push(model_dir, metrics, metadata)

    return result
