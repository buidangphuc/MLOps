"""S3-backed model registry — the "model registry (for now)" the deploy flow reads from.

Two symmetric halves, one S3 layout:

    s3://<bucket>/<prefix>/<name>/v<N>/        <- a version = a copy of the model dir
    s3://<bucket>/<prefix>/<name>/latest.json  <- {"version": N} pointer

  push(model_dir, metrics, metadata)  — TRAINING SIDE. After the gate passes, registry.base
    calls this so the trained weights land in S3 at the SAME version LocalRegistry assigned
    (passed in via metadata["local_version"]), plus a latest.json pointer.

  pull(version, dest)                 — SERVING SIDE. The serving image ships code+deps only
    (no weights); at startup it pulls the requested version here, then Predictor loads it.
    version="latest" resolves through latest.json. This is why the image is model-agnostic
    and swapping model version is an env/config change, not a rebuild.

Registered under _REMOTE_BACKENDS["s3"] in registry.base; serve/local.py builds one directly
from serving.yaml to pull.
"""

import json
import os

from datascience import log


class S3Registry:
    def __init__(self, bucket: str, prefix: str, region: str, name: str):
        if not bucket:
            raise ValueError("S3 registry needs a bucket (registry.remote.s3.bucket)")
        self.bucket = bucket
        self.prefix = (prefix or "registry").strip("/")
        self.region = region
        self.name = name

    def _client(self):
        import boto3

        return boto3.client("s3", region_name=self.region)

    def _model_prefix(self, version: str) -> str:
        return f"{self.prefix}/{self.name}/{version}"

    # --- training side -------------------------------------------------------------------
    def push(self, model_dir, metrics: dict, metadata: dict) -> str:
        """Upload model_dir to s3://bucket/prefix/<name>/v<N>/ (N = the local version, so
        S3 and local stay in lockstep) and refresh latest.json. Returns the version URI."""
        version = f"v{metadata.get('local_version', '1')}"
        s3 = self._client()
        base = self._model_prefix(version)
        model_dir = str(model_dir)

        for root, _, files in os.walk(model_dir):
            for fn in files:
                local = os.path.join(root, fn)
                rel = os.path.relpath(local, model_dir)
                s3.upload_file(local, self.bucket, f"{base}/{rel}")
        # version-scoped metadata + a name-scoped latest pointer
        s3.put_object(Bucket=self.bucket, Key=f"{base}/registry_meta.json",
                      Body=json.dumps({"name": self.name, "version": version,
                                       "metrics": metrics, "metadata": metadata},
                                      ensure_ascii=False, indent=2).encode())
        s3.put_object(Bucket=self.bucket, Key=f"{self.prefix}/{self.name}/latest.json",
                      Body=json.dumps({"version": version}, indent=2).encode())

        uri = f"s3://{self.bucket}/{base}/"
        log.info(f"[s3 registry] pushed {self.name} {version} -> {uri}")
        return uri

    # --- serving side --------------------------------------------------------------------
    def resolve_version(self, version: str) -> str:
        """'latest' -> the concrete v<N> from latest.json; a pinned 'v<N>' passes through."""
        if version and version != "latest":
            return version
        s3 = self._client()
        obj = s3.get_object(Bucket=self.bucket, Key=f"{self.prefix}/{self.name}/latest.json")
        return json.loads(obj["Body"].read())["version"]

    def pull(self, version: str, dest) -> str:
        """Download a version's model dir from S3 to `dest`. Skips if dest already populated
        (idempotent restart). Returns the local model dir path."""
        version = self.resolve_version(version)
        dest = os.path.join(str(dest), self.name, version)
        if os.path.isdir(dest) and os.listdir(dest):
            log.info(f"[s3 registry] {self.name} {version} already cached at {dest}")
            return dest

        s3 = self._client()
        base = self._model_prefix(version)
        os.makedirs(dest, exist_ok=True)
        paginator = s3.get_paginator("list_objects_v2")
        n = 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{base}/"):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(base) + 1:]
                if not rel:
                    continue
                local = os.path.join(dest, rel)
                os.makedirs(os.path.dirname(local), exist_ok=True)
                s3.download_file(self.bucket, obj["Key"], local)
                n += 1
        if n == 0:
            raise FileNotFoundError(f"no objects under s3://{self.bucket}/{base}/ — was it pushed?")
        log.info(f"[s3 registry] pulled {self.name} {version} ({n} files) -> {dest}")
        return dest


def from_config(cfg) -> "S3Registry":
    """Build from a RegistryConfig (training/push side)."""
    s3 = cfg.remote_s3 or {}
    return S3Registry(bucket=s3.get("bucket"), prefix=s3.get("prefix"),
                      region=s3.get("region"), name=cfg.name)
