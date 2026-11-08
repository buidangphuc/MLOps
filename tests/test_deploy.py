"""CI deploy: which targets changed, and the copy-then-smoke sync against stubbed AWS clients."""

import io
import json
import shutil

import boto3
import pytest
import yaml
from botocore.exceptions import ClientError

from datascience.serve import deploy


def test_only_new_or_changed_versions_are_planned():
    base = {("m", "staging"): {"endpoint": "ep", "targets": {"a": "v1", "b": "v1"}}}
    head = {("m", "staging"): {"endpoint": "ep", "targets": {"a": "v1", "b": "v2", "c": None}},
            ("m", "production"): {"endpoint": "ep-prod", "targets": {"a": "v1"}}}

    assert deploy.changes(base, head) == [
        {"model": "m", "env": "production", "target": "a", "version": "v1", "endpoint": "ep-prod"},
        {"model": "m", "env": "staging", "target": "b", "version": "v2", "endpoint": "ep"},
    ]


class _AWS:
    def __init__(self, existing=False, mode="MultiModel", targets=("alpha-v2.tar.gz",)):
        self.existing, self.mode, self.targets = existing, mode, targets
        self.copied, self.invoked = [], []

    def client(self, svc, **kw):
        return self

    def get_object(self, Bucket, Key):
        assert (Bucket, Key) == ("example-bucket", "builds/toy/v2/manifest.json")
        manifest = {"model_id": "toy", "targets": {t.split("-")[0]: {"tar": t} for t in self.targets}}
        return {"Body": io.BytesIO(json.dumps(manifest).encode())}

    def describe_endpoint(self, EndpointName):
        return {"EndpointConfigName": "cfg"}

    def describe_endpoint_config(self, EndpointConfigName):
        return {"ProductionVariants": [{"ModelName": "mdl"}]}

    def describe_model(self, ModelName):
        return {"PrimaryContainer": {"Mode": self.mode, "ModelDataUrl": "s3://serve-bucket/mme/"}}

    def head_object(self, Bucket, Key):
        if not self.existing:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

    def copy(self, src, bucket, key):
        self.copied.append((src["Key"], bucket, key))

    def invoke_endpoint(self, **kw):
        self.invoked.append(kw)
        return {"Body": io.BytesIO(b'{"predictions": [{"label": "a", "score": 0.9}]}')}


@pytest.fixture()
def env_file(toy_model, tmp_path):
    root = tmp_path / "deploy"
    (root / "toy").mkdir(parents=True)
    (root / "toy" / "staging.yaml").write_text(yaml.safe_dump({"endpoint": "ep", "targets": {"alpha": "v2"}}))
    (root / "toy" / "smoke.json").write_text('{"text": "hi"}')
    return root


def test_sync_copies_once_then_smokes_the_new_target_model(env_file, monkeypatch):
    aws = _AWS()
    monkeypatch.setattr(boto3, "client", aws.client)

    out = deploy.sync("toy", "staging", "alpha", root=env_file)

    assert aws.copied == [("builds/toy/v2/alpha-v2.tar.gz", "serve-bucket", "mme/alpha-v2.tar.gz")]
    assert aws.invoked[0]["TargetModel"] == "alpha-v2.tar.gz" and aws.invoked[0]["Body"] == b'{"text": "hi"}'
    assert out["copied"] and out["smoke"]["predictions"][0]["label"] == "a"


def test_sync_never_overwrites_a_key_already_in_the_prefix(env_file, monkeypatch):
    aws = _AWS(existing=True)
    monkeypatch.setattr(boto3, "client", aws.client)

    assert deploy.sync("toy", "staging", "alpha", root=env_file)["copied"] is False
    assert aws.copied == []


@pytest.mark.parametrize("aws, match", [(_AWS(targets=("beta-v2.tar.gz",)), "not a target"),
                                        (_AWS(mode="SingleModel"), "not a multi-model")])
def test_sync_refuses_unknown_tars_and_single_model_endpoints(env_file, monkeypatch, aws, match):
    monkeypatch.setattr(boto3, "client", aws.client)

    with pytest.raises(SystemExit, match=match):
        deploy.sync("toy", "staging", "alpha", root=env_file)
    assert aws.copied == []


def test_pull_restores_the_champion_into_the_model_dir(toy_model, monkeypatch):
    from datascience import cli
    from datascience.pipeline import stages

    stages.run_all()
    shutil.rmtree(toy_model.MODEL_DIR)
    monkeypatch.setenv("DS_MODEL", "toy")

    cli.main(["--model", "toy", "pull"])

    assert json.loads((toy_model.MODEL_DIR / "weights.json").read_text()) == {"score": 0.7}
