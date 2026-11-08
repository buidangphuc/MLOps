"""Packaging, release manifest and the serving shell, driven by the toy pack (no AWS, no torch)."""

import gzip
import hashlib
import json
import sys
import tarfile

import pytest


def _build(toy_model, version="v1"):
    from datascience.package import package
    from datascience.pipeline import stages

    stages.run_all()
    return package(version)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pytorch_dlc_tars_are_versioned_deterministic_and_smoked(toy_model, monkeypatch):
    smoked = []
    toy = sys.modules["packs.toy.pack"]
    real_smoke = toy.smoke
    monkeypatch.setattr(toy, "smoke", lambda d, t, cm: smoked.append(real_smoke(d, t, cm)))

    built = _build(toy_model)
    first = {t.name: _sha(t) for t in toy_model.MME_TARS.glob("*.tar.gz")}
    _build(toy_model)
    second = {t.name: _sha(t) for t in toy_model.MME_TARS.glob("*.tar.gz")}

    assert set(first) == {"alpha-v1.tar.gz", "beta-v1.tar.gz"} and first == second
    assert built == {"alpha": {"kind": "pytorch-dlc", "serving_model_name": "alpha"},
                     "beta": {"kind": "pytorch-dlc", "serving_model_name": "beta"}}
    assert [s["target"] for s in smoked] == ["alpha", "beta", "alpha", "beta"]
    with tarfile.open(toy_model.MME_TARS / "alpha-v1.tar.gz") as tar:
        members = tar.getmembers()
    assert [m.name for m in members] == ["inference.py", "model_meta.json", "weights.json"]
    assert all(m.mtime == 0 and m.mode == 0o644 and m.uid == 0 and m.uname == "" for m in members)
    with gzip.open(toy_model.MME_TARS / "alpha-v1.tar.gz") as gz:
        assert gz.read(1)
    build = json.loads((toy_model.MME_TARS / "build-v1.json").read_text())
    assert build["model_id"] == "toy" and build["mlflow_run_id"]


def test_release_uploads_tars_and_writes_the_manifest(toy_model, monkeypatch):
    _build(toy_model)
    calls = {"upload": [], "put": []}

    class S3:
        def head_object(self, **k):
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

        def upload_file(self, f, bucket, key):
            calls["upload"].append(key)

        def put_object(self, Bucket, Key, Body, ContentType):
            calls["put"].append((Key, json.loads(Body)))

    class STS:
        def get_caller_identity(self):
            return {"Account": "000000000000"}

    import boto3

    monkeypatch.setattr(boto3, "client", lambda svc, **k: S3() if svc == "s3" else STS())
    monkeypatch.setenv("RELEASE_BUILD_STORE", "s3://other-bucket/store")
    from datascience.serve.release import release

    assert release("v1") == "s3://other-bucket/store/v1/"
    assert sorted(calls["upload"]) == ["store/v1/alpha-v1.tar.gz", "store/v1/beta-v1.tar.gz"]
    key, manifest = calls["put"][0]
    assert key == "store/v1/manifest.json"
    assert manifest["model_id"] == "toy" and manifest["version"] == "v1"
    assert manifest["targets"]["alpha"]["tar"] == "alpha-v1.tar.gz"
    assert manifest["targets"]["alpha"]["sha256"] == _sha(toy_model.MME_TARS / "alpha-v1.tar.gz")
    assert manifest["targets"]["beta"]["kind"] == "pytorch-dlc"


def test_a_released_version_is_never_overwritten(toy_model, monkeypatch):
    _build(toy_model)

    class S3:
        def head_object(self, **k):
            return {}

    import boto3

    monkeypatch.setattr(boto3, "client", lambda svc, **k: S3() if svc == "s3" else
                        type("STS", (), {"get_caller_identity": lambda self: {"Account": "0"}})())
    from datascience.serve.release import release

    with pytest.raises(FileExistsError, match="already released"):
        release("v1")


def test_serving_shell_answers_with_the_pack_contract(toy_model, monkeypatch):
    from datascience.constants import paths

    _build(toy_model)
    monkeypatch.setitem(sys.modules, "packs.toy.serving_app", sys.modules["packs.toy.pack"])
    monkeypatch.setenv("DS_PACK", "toy")
    monkeypatch.setenv("MODEL_SOURCE", "local")
    monkeypatch.setenv("MODEL_DIR", str(toy_model.MODEL_DIR))
    monkeypatch.syspath_prepend(str(paths.PROJECT_ROOT / "ops" / "serve"))
    for mod in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        monkeypatch.delitem(sys.modules, mod)

    from fastapi.testclient import TestClient

    from app.config import get_settings
    from app.predictor import load_predictor
    from app.registrar import register_app

    get_settings.cache_clear()
    load_predictor(get_settings())
    with TestClient(register_app()) as client:
        assert client.get("/ready").json() == {"status": "ready"}
        r = client.post("/api/v1/predict", json={"text": "hello"}, headers={"X-Request-ID": "abc"})
    assert r.json() == {"label": "a", "score": 0.7, "text": "hello"}
    assert r.headers["X-Request-ID"] == "abc"
    get_settings.cache_clear()
