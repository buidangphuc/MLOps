"""`python main.py --model <id> train-remote`: the whole pipeline (stages, gate, register) as one
SageMaker Training Job running ops/train's image. Built as a plain CreateTrainingJob request
(boto3), so it does not depend on a SageMaker SDK major version."""

import time

from datascience import log
from datascience.cloud import config as C
from datascience.constants import paths


def request(cfg: dict, step: str = "all", job_name: str | None = None) -> dict:
    t = cfg.get("training") or {}
    model_id = paths.active().MODEL_ID
    req = {
        "AlgorithmSpecification": {
            "TrainingImage": cfg["image_uri"],
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["python", "main.py"],
            "ContainerArguments": ["--model", model_id, "sm-step", step],
        },
        "RoleArn": cfg["role_arn"],
        "OutputDataConfig": {"S3OutputPath": C.s3(cfg, "training")},
        "ResourceConfig": {
            "InstanceType": t.get("instance_type", "ml.g5.xlarge"),
            "InstanceCount": 1,
            "VolumeSizeInGB": int(t.get("volume_gb", 100)),
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": int(t.get("max_runtime_s", 86400))},
        "Environment": C.job_environment(cfg),
    }
    if job_name:
        req["TrainingJobName"] = job_name
    return req


def train_remote(wait: bool = False) -> str:
    import boto3

    cfg = C.load()
    C.require(cfg, "image_uri", "role_arn", "bucket", "region")
    name = f"{paths.active().MODEL_ID.replace('_', '-')}-{time.strftime('%Y%m%d-%H%M%S')}"
    sm = boto3.client("sagemaker", region_name=cfg["region"])
    sm.create_training_job(**request(cfg, job_name=name))
    log.info(f"started training job {name} -> {C.s3(cfg, 'training', name)}")
    if wait:
        sm.get_waiter("training_job_completed_or_stopped").wait(TrainingJobName=name)
        status = sm.describe_training_job(TrainingJobName=name)["TrainingJobStatus"]
        if status != "Completed":
            raise SystemExit(f"training job {name} ended {status}")
    return name
