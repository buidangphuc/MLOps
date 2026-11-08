"""SageMaker settings: config/<model>/sagemaker.yaml, with account-specific values taken from
the environment (CI sets them from repository / environment variables), so no ARN or bucket
ever lands in git."""

import os

import yaml

from datascience.constants import paths

_ENV = {
    "image_uri": "SAGEMAKER_IMAGE_URI",
    "role_arn": "SAGEMAKER_ROLE_ARN",
    "bucket": "SAGEMAKER_BUCKET",
    "region": "AWS_REGION",
}


def load() -> dict:
    cfg = yaml.safe_load(paths.SAGEMAKER_FILE.read_text()) if paths.SAGEMAKER_FILE.exists() else {}
    for key, env in _ENV.items():
        if os.environ.get(env):
            cfg[key] = os.environ[env]
    mlflow = dict(cfg.get("mlflow") or {})
    mlflow["tracking_uri"] = os.environ.get("MLFLOW_TRACKING_URI") or mlflow.get("tracking_uri")
    mlflow["secret_arn"] = os.environ.get("MLFLOW_SECRET_ARN") or mlflow.get("secret_arn")
    cfg["mlflow"] = mlflow
    cfg.setdefault("prefix", f"mlops/{paths.active().MODEL_ID}")
    return cfg


def require(cfg: dict, *keys) -> None:
    missing = [k for k in keys if not cfg.get(k)]
    if missing:
        raise SystemExit(f"set {', '.join(_ENV.get(k, k) for k in missing)} "
                         f"(or {missing} in config/{paths.active().MODEL_ID}/sagemaker.yaml)")


def job_environment(cfg: dict) -> dict:
    env = {"DS_MODEL": paths.active().MODEL_ID}
    if cfg["mlflow"].get("tracking_uri"):
        env["MLFLOW_TRACKING_URI"] = cfg["mlflow"]["tracking_uri"]
    if cfg["mlflow"].get("secret_arn"):
        env["MLFLOW_SECRET_ARN"] = cfg["mlflow"]["secret_arn"]
    return env


def s3(cfg: dict, *parts) -> str:
    return "/".join([f"s3://{cfg['bucket']}", cfg["prefix"].strip("/"), *parts])
