---
name: release-model
description: Ship a registered model to an endpoint: pull, package with smoke, release, deploy, roll back. Use when releasing a version, deploying to staging or production, rolling back, or rehearsing release and deploy without AWS.
---

# Release and deploy a model

A SageMaker Multi-Model Endpoint caches each model **by S3 key**, so every tar is
`<target>-<version>.tar.gz`, a released version is immutable, and a change always ships as a new
version. Endpoints themselves are created by infra (IaC), outside this repo.

## 1. Choose the two versions

- Registry version: the MLflow alias (`registry.remote.mlflow.alias`, default `champion`) or a
  version number.
- Release version: a new string naming the tars (e.g. `v20261108`); it must not exist in the build
  store yet.

Done when both are written down.

## 2. Release

Run `.github/workflows/release.yml` with `model`, `registry_version`, `version`. It runs
`pull` → `package` (every target smoke-tested) → `release` (tars + `manifest.json` to
`RELEASE_BUILD_STORE`). A smoke failure stops the release; fix the pack and release a new version.

Done when `<build_store>/<version>/manifest.json` lists every target with its tar and sha256.

## 3. Deploy

Open a PR that sets `deploy/<model>/<env>.yaml`: `endpoint` and `<target>: <version>`.
`.github/workflows/deploy.yml` plans the changed targets, copies each tar into the endpoint's model
prefix (an existing key is left as is) and smoke-invokes it with `deploy/<model>/smoke.json`.
`production` runs in a GitHub environment with required reviewers. After merge, clients send
`TargetModel: <target>-<version>.tar.gz`.

Done when the deploy job prints `copied` and a smoke answer for every changed target.

## 4. Roll back

Revert the env-file PR (the previous key is still in the prefix), and move the MLflow alias back if
the registry pointer should follow.

## Rehearsal without AWS

`docs/usage.md` § Rehearse the whole flow runs the same commands against a local moto emulator
(S3, STS, SageMaker); `python -m datascience.serve.deploy plan --base <sha>` and
`... sync <model> <env> <target>` replay the deploy job.
