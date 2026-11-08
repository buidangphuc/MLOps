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

    version = s.MODEL_VERSION
    if not version.isdigit():
        client = MlflowClient(tracking_uri=s.MLFLOW_TRACKING_URI, registry_uri=s.MLFLOW_TRACKING_URI)
        version = str(client.get_model_version_by_alias(s.MODEL_NAME, version).version)

    dest = Path(s.REGISTRY_CACHE) / s.MODEL_NAME / version
    if dest.is_dir() and any(dest.iterdir()):
        logger.info(f"registry: {s.MODEL_NAME} v{version} already cached at {dest}")
        return str(dest)

    dest.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{s.MODEL_NAME}/{version}", dst_path=str(dest),
        tracking_uri=s.MLFLOW_TRACKING_URI,
    )
    logger.info(f"registry: pulled {s.MODEL_NAME} v{version} -> {dest}")
    return str(dest)
