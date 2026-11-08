"""App settings — env-driven (pydantic-settings). No training config, no params.yaml.

MODEL_SOURCE / MODEL_VERSION pick the weights; MLflow credentials come from env
MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", protected_namespaces=()
    )

    ENVIRONMENT: str = "test"
    PORT: int = 8080
    LOG_LEVEL: str = "INFO"

    # Flip MODEL_VERSION (env) to change the served model, no rebuild.
    MODEL_SOURCE: str = "local"          # local | registry
    MODEL_VERSION: str = "champion"
    MODEL_DIR: str = "/app/model"        # MODEL_SOURCE=local: the mounted model dir
    REGISTRY_CACHE: str = "/tmp/registry_cache"

    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MODEL_NAME: str = ""

    BATCH_SIZE: int = 32

    X_REQUEST_ID_HEADER: str = "X-Request-ID"


@lru_cache
def get_settings() -> Settings:
    return Settings()
