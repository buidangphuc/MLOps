"""SageMaker Training Job / Pipeline requests (validated against botocore's SageMaker API model)
and the in-container step runner, driven by the toy pack. No AWS calls."""

import json
import shutil
import tarfile

import botocore.session
import pytest
import yaml
from botocore.validate import validate_parameters

SM_ENV = {
    "SAGEMAKER_IMAGE_URI": "111122223333.dkr.ecr.us-east-1.amazonaws.com/mlops-train:abc",
    "SAGEMAKER_ROLE_ARN": "arn:aws:iam::111122223333:role/example-sagemaker-execution",
    "SAGEMAKER_BUCKET": "example-bucket",
    "AWS_REGION": "us-east-1",
}


@pytest.fixture()
def sm_env(toy_model, monkeypatch):
    for k, v in SM_ENV.items():
        monkeypatch.setenv(k, v)
    return toy_model


def _shape(op):
    model = botocore.session.get_session().get_service_model("sagemaker")
    return model.operation_model(op).input_shape


def _concrete(obj):
    if isinstance(obj, dict):
        if set(obj) & {"Get", "Std:Join", "Std:JsonGet"}:
            return "s3://placeholder/x"
        return {k: _concrete(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_concrete(v) for v in obj]
    return obj


def test_training_job_request_matches_the_api(sm_env):
    from datascience.cloud import config as C
    from datascience.cloud.train_job import request

    req = request(C.load(), job_name="toy-20260101")

    validate_parameters(req, _shape("CreateTrainingJob"))
    assert req["AlgorithmSpecification"]["ContainerArguments"] == ["--model", "toy", "sm-step", "all"]
    assert req["OutputDataConfig"]["S3OutputPath"] == "s3://example-bucket/mlops/toy/training"
    assert req["Environment"]["DS_MODEL"] == "toy"


def test_missing_account_settings_stop_before_any_call(toy_model):
    from datascience.cloud.train_job import train_remote

    with pytest.raises(SystemExit, match="SAGEMAKER_IMAGE_URI"):
        train_remote()


def _steps(defn):
    out = []
    for s in defn["Steps"]:
        out.append(s)
        if s["Type"] == "Condition":
            out += s["Arguments"]["IfSteps"] + s["Arguments"]["ElseSteps"]
    return {s["Name"]: s for s in out}


def test_pipeline_definition_steps_match_the_api_and_the_gate(sm_env):
    from datascience.cloud.pipeline import definition

    defn = definition()
    steps = _steps(defn)

    assert set(steps) == {"Train", "Evaluate", "Gate", "Register"}
    train_args = {**_concrete(steps["Train"]["Arguments"]), "TrainingJobName": "x"}
    validate_parameters(train_args, _shape("CreateTrainingJob"))
    assert train_args["AlgorithmSpecification"]["ContainerArguments"][-1] == "train"
    for name in ("Evaluate", "Register"):
        validate_parameters({**_concrete(steps[name]["Arguments"]), "ProcessingJobName": "x"},
                            _shape("CreateProcessingJob"))
    assert steps["Evaluate"]["Arguments"]["ProcessingInputs"][0]["S3Input"]["S3Uri"] == \
        {"Get": "Steps.Train.ModelArtifacts.S3ModelArtifacts"}
    (cond,) = steps["Gate"]["Arguments"]["Conditions"]
    assert cond["LeftValue"]["Std:JsonGet"]["Path"] == "score" and cond["RightValue"] == 0.5
    assert steps["Gate"]["Arguments"]["ElseSteps"] == []
    assert steps["Evaluate"]["PropertyFiles"][0]["FilePath"] == "evaluation.json"


def test_an_error_gate_fails_the_pipeline(sm_env):
    from datascience.cloud.pipeline import definition

    params = yaml.safe_load(sm_env.PARAMS_FILE.read_text())
    params["gate"]["on_fail"] = "error"
    sm_env.PARAMS_FILE.write_text(yaml.safe_dump(params))

    gate = _steps(definition())["Gate"]["Arguments"]
    assert [s["Type"] for s in gate["ElseSteps"]] == ["Fail"]


def test_sm_steps_round_trip_through_job_dirs(toy_model, tmp_path, monkeypatch):
    import os

    from mlflow import MlflowClient

    from datascience.cloud.steps import run

    sm = tmp_path / "opt_ml"
    monkeypatch.setenv("SM_ROOT", str(sm))

    def fresh_container():
        shutil.rmtree(toy_model.ARTIFACTS, ignore_errors=True)
        toy_model.ensure_dirs()

    run("train")
    assert (sm / "model" / "model" / "weights.json").exists() and (sm / "model" / "split").is_dir()

    model_in = sm / "processing" / "input" / "model"
    model_in.mkdir(parents=True)
    with tarfile.open(model_in / "model.tar.gz", "w:gz") as tar:
        for child in (sm / "model").iterdir():
            tar.add(child, arcname=child.name)
    fresh_container()
    run("evaluate")
    assert json.loads((sm / "processing" / "evaluation" / "evaluation.json").read_text()) == {"score": 0.7}

    shutil.rmtree(model_in)
    shutil.move(sm / "processing" / "model", model_in)
    shutil.move(sm / "processing" / "evaluation", sm / "processing" / "input" / "evaluation")
    fresh_container()
    run("register")

    uri = os.environ["MLFLOW_TRACKING_URI"]
    mv = MlflowClient(tracking_uri=uri, registry_uri=uri).get_model_version_by_alias("ToyModel", "champion")
    assert str(mv.version) == "1" and mv.tags["dataset"] == "toy"


def test_upsert_creates_then_updates(sm_env, monkeypatch):
    import boto3
    from botocore.exceptions import ClientError

    from datascience.cloud import pipeline

    calls = []

    class SM:
        def create_pipeline(self, **kw):
            if any(c[0] == "create" for c in calls):
                calls.append(("create-refused",))
                raise ClientError({"Error": {"Code": "ValidationException",
                                             "Message": "Pipeline names must be unique: already exists"}}, "CreatePipeline")
            calls.append(("create", json.loads(kw["PipelineDefinition"])["Version"]))

        def update_pipeline(self, **kw):
            calls.append(("update", kw["PipelineName"]))

    monkeypatch.setattr(boto3, "client", lambda svc, **k: SM())
    pipeline.upsert()
    pipeline.upsert()

    assert calls == [("create", "2020-12-01"), ("create-refused",), ("update", "toy")]
