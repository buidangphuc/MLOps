"""`python main.py --model <id> pipeline definition|upsert|start`: a SageMaker Pipeline

    Train (Training Job, stages before evaluate)
      -> Evaluate (Processing Job, writes evaluation.json as a property file)
      -> Gate (Condition: every params.yaml gate rule on evaluation.json)
           pass -> Register (Processing Job: platform gate + MLflow registration with the alias)
           fail -> GateFailed (Fail step) when gate.on_fail is error; nothing when it is warn

The definition JSON is built directly (the documented pipeline-definition schema) and sent with
boto3, so it does not depend on a SageMaker SDK major version. CI owns `upsert` and `start`
(.github/workflows/pipeline.yml); people don't run pipelines from laptops.
"""

import json

from datascience import log
from datascience.cloud import config as C
from datascience.cloud.train_job import request as training_request
from datascience.config.configuration import ConfigurationManager
from datascience.constants import paths

_EXEC_ID = {"Get": "Execution.PipelineExecutionId"}


def _join(*values):
    return {"Std:Join": {"On": "/", "Values": list(values)}}


def _processing(cfg, name, step, inputs, outputs) -> dict:
    p = cfg.get("processing") or {}
    model_id = paths.active().MODEL_ID
    return {
        "Name": name,
        "Type": "Processing",
        "Arguments": {
            "AppSpecification": {
                "ImageUri": cfg["image_uri"],
                "ContainerEntrypoint": ["python", "main.py", "--model", model_id, "sm-step", step],
            },
            "RoleArn": cfg["role_arn"],
            "ProcessingResources": {"ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": p.get("instance_type", "ml.g5.xlarge"),
                "VolumeSizeInGB": int(p.get("volume_gb", 50)),
            }},
            "ProcessingInputs": [
                {"InputName": n, "S3Input": {"S3Uri": uri, "LocalPath": f"/opt/ml/processing/input/{n}",
                                             "S3DataType": "S3Prefix", "S3InputMode": "File"}}
                for n, uri in inputs.items()
            ],
            "ProcessingOutputConfig": {"Outputs": [
                {"OutputName": n, "S3Output": {"S3Uri": _join(C.s3(cfg, "pipeline"), _EXEC_ID, name, n),
                                               "LocalPath": f"/opt/ml/processing/{n}",
                                               "S3UploadMode": "EndOfJob"}}
                for n in outputs
            ]},
            "Environment": C.job_environment(cfg),
            "StoppingCondition": {"MaxRuntimeInSeconds": int(p.get("max_runtime_s", 7200))},
        },
    }


def _output(step, name):
    return {"Get": f"Steps.{step}.ProcessingOutputConfig.Outputs['{name}'].S3Output.S3Uri"}


def definition() -> dict:
    cfg = C.load()
    gate = ConfigurationManager().gate()

    train_args = training_request(cfg, step="train")
    train = {"Name": "Train", "Type": "Training", "Arguments": train_args}
    evaluate = _processing(cfg, "Evaluate", "evaluate",
                           {"model": {"Get": "Steps.Train.ModelArtifacts.S3ModelArtifacts"}},
                           ["evaluation", "model"])
    evaluate["PropertyFiles"] = [{"PropertyFileName": "Evaluation", "OutputName": "evaluation",
                                  "FilePath": "evaluation.json"}]
    register = _processing(cfg, "Register", "register",
                           {"model": _output("Evaluate", "model"), "evaluation": _output("Evaluate", "evaluation")},
                           [])
    conditions = [
        {"Type": "GreaterThanOrEqualTo",
         "LeftValue": {"Std:JsonGet": {"PropertyFile": {"Get": "Steps.Evaluate.PropertyFiles.Evaluation"},
                                       "Path": r["metric"]}},
         "RightValue": float(r["min"])}
        for r in gate.rules
    ] if gate.enabled else []
    fail = {"Name": "GateFailed", "Type": "Fail",
            "Arguments": {"ErrorMessage": f"quality gate failed: {gate.rules}"}}
    steps = [train, evaluate]
    if conditions:
        steps.append({"Name": "Gate", "Type": "Condition", "Arguments": {
            "Conditions": conditions,
            "IfSteps": [register],
            "ElseSteps": [fail] if gate.on_fail == "error" else [],
        }})
    else:
        steps.append(register)
    return {"Version": "2020-12-01", "Metadata": {}, "Parameters": [], "PipelineExperimentConfig": {
        "ExperimentName": {"Get": "Execution.PipelineName"}, "TrialName": _EXEC_ID}, "Steps": steps}


def _name(cfg) -> str:
    return (cfg.get("pipeline") or {}).get("name") or paths.active().MODEL_ID.replace("_", "-")


def upsert() -> str:
    import boto3

    cfg = C.load()
    C.require(cfg, "image_uri", "role_arn", "bucket", "region")
    sm = boto3.client("sagemaker", region_name=cfg["region"])
    name, body = _name(cfg), json.dumps(definition())
    try:
        sm.describe_pipeline(PipelineName=name)
        sm.update_pipeline(PipelineName=name, PipelineDefinition=body, RoleArn=cfg["role_arn"])
        log.info(f"updated pipeline {name}")
    except sm.exceptions.ResourceNotFound:
        sm.create_pipeline(PipelineName=name, PipelineDefinition=body, RoleArn=cfg["role_arn"])
        log.info(f"created pipeline {name}")
    return name


def start() -> str:
    import boto3

    cfg = C.load()
    C.require(cfg, "region")
    arn = boto3.client("sagemaker", region_name=cfg["region"]).start_pipeline_execution(
        PipelineName=_name(cfg))["PipelineExecutionArn"]
    log.info(f"started {arn}")
    return arn
