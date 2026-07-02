"""S3/MinIO remote registry backend — fills the registry.base seam.

Pushes a gated model directory to an S3-compatible bucket (AWS S3 or MinIO)
so deploy targets can pull it. Connection comes from the environment
(12-factor): ``REGISTRY_S3_BUCKET``, ``S3_ENDPOINT_URL`` (set for MinIO),
plus the standard ``AWS_*`` credentials. A client can be injected for tests.

Enable via ``registry.remote.backend: s3`` in params.yaml.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from datascience import log
from datascience.entity.config_entity import RegistryConfig


class S3Registry:
    def __init__(self, cfg: RegistryConfig, *, client=None, bucket: str | None = None):
        self.cfg = cfg
        self.bucket = bucket or os.getenv("REGISTRY_S3_BUCKET", "models")
        self._client = client

    def _get_client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "s3",
                endpoint_url=os.getenv("S3_ENDPOINT_URL") or None,
                region_name=os.getenv("AWS_REGION", "us-east-1"),
            )
        return self._client

    def _version_prefix(self, metadata: dict) -> str:
        data_version = metadata.get("data_version") or {}
        tag = data_version.get("sha256") if isinstance(data_version, dict) else None
        return f"{self.cfg.name}/{tag or 'latest'}"

    def push(self, model_dir, metrics: dict, metadata: dict) -> str:
        client = self._get_client()
        prefix = self._version_prefix(metadata)
        model_dir = Path(model_dir)

        for path in sorted(model_dir.rglob("*")):
            if path.is_file():
                key = f"{prefix}/{path.relative_to(model_dir).as_posix()}"
                client.upload_file(str(path), self.bucket, key)

        client.put_object(
            Bucket=self.bucket,
            Key=f"{prefix}/registry_meta.json",
            Body=json.dumps({"metrics": metrics, "metadata": metadata}).encode("utf-8"),
        )
        uri = f"s3://{self.bucket}/{prefix}"
        log.info(f"pushed model to remote registry: {uri}")
        return uri
