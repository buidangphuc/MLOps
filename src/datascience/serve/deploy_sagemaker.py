"""Deploy the Triton model_repository to a SageMaker real-time endpoint (Triton DLC).

Runs on AWS (boto3 + creds). Fill serving.sagemaker.{s3_bucket, role_arn, triton_image}
in params.yaml — the Triton DLC image URI is region-specific (see AWS docs). Untested
locally; this is the production deploy path you run on the GPU/AWS side.
"""

import tarfile

from datascience import log
from datascience.config.configuration import ConfigurationManager
from datascience.constants import paths


def deploy_sagemaker():
    cm = ConfigurationManager()
    sm = cm.serving().sagemaker
    name = cm.params.registry.name

    for key in ("s3_bucket", "role_arn", "triton_image"):
        if not sm.get(key):
            raise ValueError(f"set serving.sagemaker.{key} in params.yaml before deploying")

    import boto3

    region = sm.region
    s3 = boto3.client("s3", region_name=region)
    sagemaker = boto3.client("sagemaker", region_name=region)

    # 1. package the model_repository and upload to S3
    tar_path = paths.ARTIFACTS / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(paths.TRITON_REPO, arcname=".")
    key = f"{name}/model.tar.gz"
    s3.upload_file(str(tar_path), sm.s3_bucket, key)
    model_data = f"s3://{sm.s3_bucket}/{key}"
    log.info(f"uploaded {model_data}")

    # 2. SageMaker model (Triton DLC image) -> endpoint config -> endpoint
    sagemaker.create_model(
        ModelName=name,
        PrimaryContainer={
            "Image": sm.triton_image,
            "ModelDataUrl": model_data,
            "Environment": {"SAGEMAKER_TRITON_DEFAULT_MODEL_NAME": "text_classifier"},
        },
        ExecutionRoleArn=sm.role_arn,
    )
    sagemaker.create_endpoint_config(
        EndpointConfigName=name,
        ProductionVariants=[{
            "VariantName": "main",
            "ModelName": name,
            "InstanceType": sm.instance_type,
            "InitialInstanceCount": 1,
        }],
    )
    sagemaker.create_endpoint(EndpointName=name, EndpointConfigName=name)
    log.info(f"SageMaker endpoint '{name}' creating ({sm.instance_type}). Poll DescribeEndpoint for InService.")
    return name
