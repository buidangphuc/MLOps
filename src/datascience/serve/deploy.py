"""CI-side deploy to a long-lived SageMaker Multi-Model Endpoint. Run by
.github/workflows/deploy.yml under a per-environment deploy role, never from a laptop.

deploy/<model>/<env>.yaml is the desired state of one environment:

    endpoint: <endpoint name>          # created and owned by infra (IaC), not by this repo
    targets:
      <target>: <release version>      # what clients send as TargetModel = <target>-<version>.tar.gz

  plan --base <sha>             changed (model, env, target, version) since <sha>, as a JSON matrix
  sync <model> <env> <target>   copy <target>-<version>.tar.gz from the build store into the
                                endpoint's model prefix (never overwrite: the MME caches by key),
                                then smoke-invoke it with deploy/<model>/smoke.json

Reverting the env file is the rollback: the previous key is still in the prefix.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from datascience.constants import paths

DEPLOY_DIR = paths.PROJECT_ROOT / "deploy"


def changes(base: dict, head: dict) -> list:
    """`base` / `head` map (model, env) -> parsed env file. Returns one item per target whose
    version is new or changed in head (a removed or nulled target needs no copy)."""
    out = []
    for (model, env), doc in sorted(head.items()):
        old = (base.get((model, env)) or {}).get("targets") or {}
        for target, version in sorted((doc.get("targets") or {}).items()):
            if version and str(version) != str(old.get(target)):
                out.append({"model": model, "env": env, "target": target, "version": str(version),
                            "endpoint": doc.get("endpoint")})
    return out


def _env_files(root: Path):
    return sorted(p for p in root.glob("*/*.yaml"))


def plan(base_sha: str, root: Path = DEPLOY_DIR) -> list:
    head, base = {}, {}
    for f in _env_files(root):
        key = (f.parent.name, f.stem)
        head[key] = yaml.safe_load(f.read_text()) or {}
        rel = f.relative_to(paths.PROJECT_ROOT).as_posix()
        shown = subprocess.run(["git", "show", f"{base_sha}:{rel}"], capture_output=True, text=True)
        base[key] = (yaml.safe_load(shown.stdout) or {}) if shown.returncode == 0 else {}
    return changes(base, head)


def _split(uri: str):
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key.strip("/")


def sync(model: str, env: str, target: str, root: Path = DEPLOY_DIR) -> dict:
    import boto3

    doc = yaml.safe_load((root / model / f"{env}.yaml").read_text())
    version, endpoint = str(doc["targets"][target]), doc["endpoint"]
    if not endpoint:
        raise SystemExit(f"deploy/{model}/{env}.yaml has no endpoint")
    serving = yaml.safe_load(paths.for_model(model).SERVING_FILE.read_text())["sagemaker"]
    store = os.environ.get("RELEASE_BUILD_STORE") or serving["release"]["build_store"]
    region = os.environ.get("AWS_REGION") or serving["region"]
    tar = f"{target}-{version}.tar.gz"

    s3 = boto3.client("s3", region_name=region)
    sm = boto3.client("sagemaker", region_name=region)
    store_bucket, store_prefix = _split(store)
    manifest = json.loads(s3.get_object(Bucket=store_bucket, Key=f"{store_prefix}/{version}/manifest.json")["Body"].read())
    if manifest.get("model_id") != model or tar not in [t["tar"] for t in manifest["targets"].values()]:
        raise SystemExit(f"{tar} is not a target of build {version} of {model}")

    config = sm.describe_endpoint(EndpointName=endpoint)["EndpointConfigName"]
    model_name = sm.describe_endpoint_config(EndpointConfigName=config)["ProductionVariants"][0]["ModelName"]
    container = sm.describe_model(ModelName=model_name)["PrimaryContainer"]
    if container.get("Mode") != "MultiModel":
        raise SystemExit(f"{endpoint} is not a multi-model endpoint")
    dest_bucket, dest_prefix = _split(container["ModelDataUrl"])
    dest_key = f"{dest_prefix}/{tar}" if dest_prefix else tar

    from botocore.exceptions import ClientError

    try:
        s3.head_object(Bucket=dest_bucket, Key=dest_key)
        copied = False
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchKey"):
            raise
        s3.copy({"Bucket": store_bucket, "Key": f"{store_prefix}/{version}/{tar}"}, dest_bucket, dest_key)
        copied = True

    body = (root / model / "smoke.json").read_bytes()
    resp = boto3.client("sagemaker-runtime", region_name=region).invoke_endpoint(
        EndpointName=endpoint, TargetModel=tar, ContentType="application/json", Body=body)
    answer = json.loads(resp["Body"].read())
    return {"endpoint": endpoint, "target_model": tar, "copied": copied, "smoke": answer}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m datascience.serve.deploy")
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--base", required=True)
    sy = sub.add_parser("sync")
    sy.add_argument("model")
    sy.add_argument("env")
    sy.add_argument("target")
    args = p.parse_args(argv)
    out = plan(args.base) if args.cmd == "plan" else sync(args.model, args.env, args.target)
    json.dump(out, sys.stdout)
    print()


if __name__ == "__main__":
    main()
