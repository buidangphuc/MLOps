"""Typed configs the platform owns (knobs only — artifact paths live in constants/paths.py).
Pack-specific configs live in packs/<pack>/config.py."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GateConfig:
    enabled: bool
    on_fail: str                  # "error" | "warn"
    rules: list                   # [{metric, min}, ...]


@dataclass
class RegistryConfig:
    name: str
    remote_backend: str           # none | mlflow | s3
    remote_s3: Optional[dict] = None   # {bucket, prefix, region} when remote_backend == s3
    tracking_uri: Optional[str] = None
    experiment_name: Optional[str] = None
    alias: str = "champion"       # MLflow alias set on a registered version (the approval pointer)
