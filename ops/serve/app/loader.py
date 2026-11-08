"""Model acquisition: local mount or MLflow registry pull.

Self-contained: NO import from datascience, no ConfigurationManager, no params.yaml. The
registry pull is a trimmed copy of datascience/registry/mlflow_registry.py::pull.
"""

from pathlib import Path

from loguru import logger

from app.config import Settings


def resolve_model_dir(s: Settings) -> str:
    if s.MODEL_SOURCE == "registry":
        return _mlflow_pull(s)
    logger.info(f"MODEL_SOURCE=local -> {s.MODEL_DIR}")
    return s.MODEL_DIR


def _mlflow_pull(s: Settings) -> str:
    import mlflow.artifacts
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=s.MLFLOW_TRACKING_URI, registry_uri=s.MLFLOW_TRACKING_URI)
    version = s.MODEL_VERSION
    if not version.isdigit():
        version = str(client.get_model_version_by_alias(s.MODEL_NAME, version).version)

    dest = Path(s.REGISTRY_CACHE) / s.MODEL_NAME / version
    cached = dest / "model"
    if cached.is_dir() and any(cached.iterdir()):
        logger.info(f"registry: {s.MODEL_NAME} v{version} already cached at {cached}")
        return str(cached)

    dest.mkdir(parents=True, exist_ok=True)
    path = mlflow.artifacts.download_artifacts(
        artifact_uri=client.get_model_version_download_uri(s.MODEL_NAME, version), dst_path=str(dest),
        tracking_uri=s.MLFLOW_TRACKING_URI,
    )
    logger.info(f"registry: pulled {s.MODEL_NAME} v{version} -> {path}")
    return path
