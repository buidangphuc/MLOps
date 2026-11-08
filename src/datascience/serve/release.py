"""Publish packaged MME tars to the S3 build store — the only thing this repo ships.

An MME loads whatever tar a client names in `TargetModel`, on demand, from its `ModelDataUrl`
prefix, and caches it by S3 key. So every build carries its version in the file name
(`tar_name`) and a released version is immutable (`release` refuses to overwrite one).

    release  (CI build role) : local tars -> <build_store>/<version>/ + manifest.json

Which env serves which version lives in deploy/<model>/<env>.yaml; the CI deploy workflow copies
the tar into that env's endpoint prefix and smoke-invokes it. `manifest.json` is the contract it reads.
"""

import json
import os
import re
import time

from datascience import log
from datascience.config.configuration import ConfigurationManager
from datascience.constants import paths
from datascience.utils.io import read_data_version

# Version goes into an S3 key and a tar name — keep it boring.
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _serving():
    return ConfigurationManager().serving().sagemaker


def release_cfg(sm):
    rel = sm.get("release")
    if not rel:
        raise ValueError("add a `release:` block under serving.sagemaker in config/<model>/serving.yaml")
    return rel


def split_uri(uri, what):
    """`s3://bucket/some/prefix/` -> ('bucket', 'some/prefix'). Locations are full URIs so each
    bucket is named where it is used — no bucket is ever inferred."""
    if not uri or not str(uri).startswith("s3://"):
        raise ValueError(f"{what} must be a full s3:// URI in config/<model>/serving.yaml, got {uri!r}")
    bucket, _, prefix = str(uri)[len("s3://"):].partition("/")
    if not bucket or not prefix.strip("/"):
        raise ValueError(f"{what} needs both a bucket and a prefix: {uri!r}")
    return bucket, prefix.strip("/")


def resolve_version(explicit=None) -> str:
    """The artifact's identity: it names the tar, goes into serving_meta.json, and is what
    clients send as TargetModel. Defaults to the data fingerprint so a build is never anonymous
    (and never the container-start date — that was the old lineage bug)."""
    version = explicit or os.environ.get("RELEASE_VERSION")
    if not version:
        dv = read_data_version()
        version = f"data-{dv['sha256'][:8]}" if dv else None
    if not version:
        raise ValueError(
            "no release version — pass --version (or set RELEASE_VERSION), or run `transform` "
            "so a data fingerprint exists"
        )
    if not _VERSION_RE.match(version):
        raise ValueError(f"invalid release version {version!r} — allowed: A-Z a-z 0-9 . _ -")
    return version


def tar_name(model_name: str, version: str) -> str:
    """`<model>-<version>.tar.gz`. The version lives IN THE KEY on purpose — see module docstring."""
    return f"{model_name}-{version}.tar.gz"


def local_tars(version: str):
    tars = sorted(paths.MME_TARS.glob(f"*-{version}.tar.gz"))
    if not tars:
        raise FileNotFoundError(
            f"no *-{version}.tar.gz in {paths.MME_TARS} — run "
            f"`python main.py package --version {version}` first"
        )
    return tars


def _sha256(path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _key_exists(s3, bucket, key) -> bool:
    from botocore.exceptions import ClientError

    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def release(version=None) -> str:
    """Upload this build's tars to `s3://<bucket>/<build_prefix>/<version>/`.

    A version is immutable: if it already exists this RAISES instead of overwriting, because a
    second, different artifact under a name that is already live somewhere is exactly the bug
    this module exists to prevent."""
    import boto3

    sm = _serving()
    rel = release_cfg(sm)
    version = resolve_version(version)
    tars = local_tars(version)

    bucket, build_prefix = split_uri(os.environ.get("RELEASE_BUILD_STORE") or rel.get("build_store"),
                                     "sagemaker.release.build_store (or RELEASE_BUILD_STORE)")
    s3 = boto3.client("s3", region_name=sm.region)
    ident = boto3.client("sts", region_name=sm.region).get_caller_identity()
    base = f"{build_prefix}/{version}"
    log.info(f"[preflight] account={ident['Account']} -> s3://{bucket}/{base}/")

    if _key_exists(s3, bucket, f"{base}/manifest.json"):
        raise FileExistsError(
            f"version {version} is already released at s3://{bucket}/{base}/ — "
            "bump --version instead of overwriting (MME caches by key)"
        )

    for tar in tars:
        s3.upload_file(str(tar), bucket, f"{base}/{tar.name}")
        log.info(f"uploaded s3://{bucket}/{base}/{tar.name}")

    build_file = paths.MME_TARS / f"build-{version}.json"
    build = json.loads(build_file.read_text()) if build_file.exists() else {}
    sha = {t.name: _sha256(t) for t in tars}
    targets = build.pop("targets", {})
    for t in tars:
        name = t.name[: -len(f"-{version}.tar.gz")]
        targets[name] = {"tar": t.name, "sha256": sha[t.name], "size_bytes": t.stat().st_size, **targets.get(name, {})}
    manifest = {
        "version": version,
        "tars": [t.name for t in tars],
        "models": [t.name[: -len(f"-{version}.tar.gz")] for t in tars],
        "sha256": sha,
        **build,
        "targets": targets,
        "created_utc": _now(),
    }
    s3.put_object(
        Bucket=bucket,
        Key=f"{base}/manifest.json",
        Body=json.dumps(manifest, indent=2).encode(),
        ContentType="application/json",
    )
    log.info(f"released {len(tars)} tar(s) as {version} — set it as an env's desired version in deploy/<model>/<env>.yaml")
    return f"s3://{bucket}/{base}/"
