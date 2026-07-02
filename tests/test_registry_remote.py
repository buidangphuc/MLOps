"""S3/MinIO remote registry backend (no real S3 — injected fake client)."""

from __future__ import annotations

import json

from datascience.entity.config_entity import RegistryConfig
from datascience.registry.base import _REMOTE_BACKENDS
from datascience.registry.remote_s3 import S3Registry


class _FakeS3:
    def __init__(self):
        self.uploaded: list[tuple[str, str]] = []  # (bucket, key)
        self.objects: dict[str, bytes] = {}

    def upload_file(self, filename, bucket, key):
        self.uploaded.append((bucket, key))

    def put_object(self, *, Bucket, Key, Body):
        self.objects[Key] = Body


def test_s3_registry_uploads_model_and_meta(tmp_path):
    model_dir = tmp_path / "model"
    (model_dir / "sub").mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")
    (model_dir / "sub" / "weights.bin").write_bytes(b"\x00\x01")

    fake = _FakeS3()
    reg = S3Registry(
        RegistryConfig(name="TextClassifier", remote_backend="s3"),
        client=fake,
        bucket="models",
    )
    uri = reg.push(
        model_dir,
        {"macro_f1": 0.91},
        {"base_model": "distilbert", "data_version": {"sha256": "abc123"}},
    )

    assert uri == "s3://models/TextClassifier/abc123"
    keys = {k for _, k in fake.uploaded}
    assert "TextClassifier/abc123/config.json" in keys
    assert "TextClassifier/abc123/sub/weights.bin" in keys
    meta = json.loads(fake.objects["TextClassifier/abc123/registry_meta.json"])
    assert meta["metrics"]["macro_f1"] == 0.91


def test_s3_backend_is_registered_in_seam():
    assert _REMOTE_BACKENDS.get("s3") is S3Registry
