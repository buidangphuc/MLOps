"""Deploy the GLiNER Triton python-backend Multi-Model Endpoint (MME) to SageMaker.

Runs on AWS (boto3 + creds). Ports `gliner_multimodel_triton/zip_deploy_sagemaker.py`
{upload, deploy_endpoint, test_endpoint} and fixes the deployment-report bugs:

  - bucket / prefix / region / role from serving.yaml, NOT inferred from CWD/.env (#4,#7)
  - ONE tar per real model, NO `_1`/`_2` duplicates (#10)
  - application-autoscaling target-tracking from config, not a fixed InitialInstanceCount (scale)
  - test_endpoint RAISES on failure (old code swallowed it → false success) (#12)
  - endpoint-wait has a timeout, not `while True` (#13)

Can't be run from a non-AWS box — `deploy()` preflights (prints identity/bucket/prefix)
and refuses without creds/config. The real run happens on the AWS side.
"""

import json
import tarfile
import time

from src.datascience import log
from src.datascience.config.configuration import ConfigurationManager
from src.datascience.constants import paths

_WAIT_TIMEOUT_S = 1800


def _require(sm, keys):
    missing = [k for k in keys if not sm.get(k)]
    if missing:
        raise ValueError(f"set serving.sagemaker.{missing} in serving.yaml before deploying")


def _instance_type(sv, sm) -> str:
    return sm.instance_type_cpu if sv.get("device", "gpu") == "cpu" else sm.instance_type_gpu


def upload_mme(sv) -> str:
    """Upload each per-model tar (from package_triton) to s3://bucket/prefix/. Returns the
    prefix URI that MME's ModelDataUrl points at."""
    import boto3

    sm = sv.sagemaker
    _require(sm, ["s3_bucket", "s3_prefix"])
    tars = sorted(paths.MME_TARS.glob("*.tar.gz"))
    if not tars:
        raise FileNotFoundError(f"no tars in {paths.MME_TARS} — run `package --target triton` first")

    s3 = boto3.client("s3", region_name=sm.region)
    ident = boto3.client("sts", region_name=sm.region).get_caller_identity()
    log.info(f"[preflight] account={ident['Account']} bucket={sm.s3_bucket} prefix={sm.s3_prefix}")
    for tar in tars:
        key = f"{sm.s3_prefix}/{tar.name}"
        s3.upload_file(str(tar), sm.s3_bucket, key)
        log.info(f"uploaded s3://{sm.s3_bucket}/{key}")
    return f"s3://{sm.s3_bucket}/{sm.s3_prefix}/"


def deploy(endpoint_name=None):
    cm = ConfigurationManager()
    sv = cm.serving()
    sm = sv.sagemaker
    _require(sm, ["s3_bucket", "s3_prefix", "role_arn", "triton_image"])
    if str(sm.mode) != "MultiModel":
        raise ValueError("this deployer targets MME; set serving.sagemaker.mode: MultiModel")

    import boto3

    model_data_url = upload_mme(sv)               # uploads tars, returns the s3 PREFIX (MME)
    client = boto3.client("sagemaker", region_name=sm.region)
    ts = time.strftime("%d%m%Y-%H%M%S")           # human tag; not used as model version
    name = endpoint_name or f"gliner-mme-{ts}"
    instance_type = _instance_type(sv, sm)

    client.create_model(
        ModelName=name,
        ExecutionRoleArn=sm.role_arn,
        PrimaryContainer={
            "Image": sm.triton_image,
            "Mode": "MultiModel",                 # one endpoint, many models by TargetModel
            "ModelDataUrl": model_data_url,       # the s3 PREFIX, not a single tar
            "Environment": {"SERVE_DEVICE": sv.get("device", "gpu")},
        },
    )
    client.create_endpoint_config(
        EndpointConfigName=name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": name,
            "InstanceType": instance_type,
            "InitialInstanceCount": sm.autoscaling.get("min_instances", 1),
        }],
    )
    client.create_endpoint(EndpointName=name, EndpointConfigName=name)
    log.info(f"endpoint '{name}' creating ({instance_type}, MME)")

    _wait_in_service(client, name)
    if sm.autoscaling.get("enabled"):
        _register_autoscaling(sm, name)
    log.info(f"deployed '{name}'. Invoke a model with TargetModel='<name>.tar.gz'.")
    return name


def _wait_in_service(client, name):
    """Poll DescribeEndpoint with a timeout (not `while True`, #13)."""
    deadline = time.time() + _WAIT_TIMEOUT_S
    while time.time() < deadline:
        status = client.describe_endpoint(EndpointName=name)["EndpointStatus"]
        log.info(f"endpoint status: {status}")
        if status == "InService":
            return
        if status == "Failed":
            raise RuntimeError(f"endpoint '{name}' deployment Failed")
        time.sleep(30)
    raise TimeoutError(f"endpoint '{name}' not InService after {_WAIT_TIMEOUT_S}s")


def _register_autoscaling(sm, endpoint_name):
    """Target-tracking autoscaling from config (min/max/metric/target). Nothing hardcoded."""
    import boto3

    a = sm.autoscaling
    if a.get("max_instances") is None or a.get("target_value") is None:
        raise ValueError("set serving.sagemaker.autoscaling.{max_instances,target_value} to enable autoscaling")

    aas = boto3.client("application-autoscaling", region_name=sm.region)
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"
    aas.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=a.min_instances,
        MaxCapacity=a.max_instances,
    )
    aas.put_scaling_policy(
        PolicyName=f"{endpoint_name}-target-tracking",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": float(a.target_value),
            "PredefinedMetricSpecification": {"PredefinedMetricType": a.metric},
            "ScaleInCooldown": a.get("scale_in_cooldown_s", 300),
            "ScaleOutCooldown": a.get("scale_out_cooldown_s", 60),
        },
    )
    log.info(f"autoscaling: {a.metric} target={a.target_value}, {a.min_instances}-{a.max_instances} instances")


def test_endpoint(endpoint_name, text="cho thuê căn hộ 2 phòng ngủ quận 7"):
    """Invoke each model via TargetModel and RAISE on any failure (old code swallowed, #12)."""
    import boto3

    cm = ConfigurationManager()
    sm = cm.serving().sagemaker
    rt = boto3.client("sagemaker-runtime", region_name=sm.region)
    payload = json.dumps({"inputs": [{"name": "text_input", "shape": [1, 1],
                                      "datatype": "BYTES", "data": [text]}]})
    failures = []
    for m in sm.models:
        target = f"{m['name']}.tar.gz"
        try:
            resp = rt.invoke_endpoint(
                EndpointName=endpoint_name, ContentType="application/octet-stream",
                TargetModel=target, Body=payload,
            )
            log.info(f"OK {target}: {resp['Body'].read()[:200]}")
        except Exception as e:
            failures.append(f"{target}: {e}")
    if failures:
        raise RuntimeError("endpoint test FAILED: " + "; ".join(failures))
    log.info("endpoint test passed for all models")


def deploy_sagemaker():
    """CLI entry (`main.py deploy`)."""
    return deploy()
