"""Registry: local versioning + S3 push/pull in lockstep (no real S3 — fake client)."""

from __future__ import annotations

import io
import json

from datascience.entity.config_entity import RegistryConfig
from datascience.registry import base
from datascience.registry.s3 import S3Registry


class _FakeS3:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def upload_file(self, filename, bucket, key):
        self.objects[key] = open(filename, "rb").read()

    def put_object(self, *, Bucket, Key, Body):
        self.objects[Key] = Body

    def get_object(self, *, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}

    def get_paginator(self, _):
        objects = self.objects

        class _P:
            def paginate(self, *, Bucket, Prefix):
                yield {"Contents": [{"Key": k} for k in objects if k.startswith(Prefix)]}

        return _P()

    def download_file(self, bucket, key, local):
        with open(local, "wb") as f:
            f.write(self.objects[key])


def _model_dir(tmp_path):
    d = tmp_path / "model"
    (d / "sub").mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "sub" / "weights.bin").write_bytes(b"\x00\x01")
    return d


def test_register_local_then_s3_in_lockstep(tmp_artifacts, monkeypatch):
    fake = _FakeS3()
    monkeypatch.setattr(S3Registry, "_client", lambda self: fake)
    cfg = RegistryConfig(name="M", remote_backend="s3",
                         remote_s3={"bucket": "b", "prefix": "registry", "region": "x"})

    out = base.register(_model_dir(tmp_artifacts), {"f1": 0.9}, {"dataset": "d"}, cfg)

    assert out["local_version"] == "1"
    assert out["remote"] == "s3://b/registry/M/v1/"
    assert "registry/M/v1/sub/weights.bin" in fake.objects
    meta = json.loads(fake.objects["registry/M/v1/registry_meta.json"])
    assert meta["metrics"]["f1"] == 0.9 and meta["metadata"]["local_version"] == "1"
    assert (tmp_artifacts / "registry" / "M" / "v1" / "registry_meta.json").exists()


def test_s3_pull_resolves_latest(tmp_path, monkeypatch):
    fake = _FakeS3()
    monkeypatch.setattr(S3Registry, "_client", lambda self: fake)
    reg = S3Registry(bucket="b", prefix="registry", region="x", name="M")
    reg.push(_model_dir(tmp_path), {}, {"local_version": "3"})

    dest = reg.pull("latest", tmp_path / "cache")

    assert dest.endswith("M/v3")
    assert open(f"{dest}/sub/weights.bin", "rb").read() == b"\x00\x01"


def test_s3_backend_is_registered_in_seam():
    assert "s3" in base._REMOTE_BACKENDS
