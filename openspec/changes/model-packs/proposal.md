## Why

The repo trains, versions and serves exactly one model, and its serving and packaging code is still GLiNER-shaped (broken for the classifier). Deployment is a laptop script (`serve/deploy_sagemaker.py`) that creates and updates endpoints with personal credentials, which the AWS Well-Architected ML Lens lists as an anti-pattern (MLOPS04-BP01: manual deploys, one-off deploy scripts). The platform part of a sibling multi-model repo (model packs, per-model trees, generic packaging and release) is ported here, without its real models.

## What Changes

- **Per-model trees.** `ModelPaths` roots `config/<id>/` and `artifacts/<id>/`; `--model` / `DS_MODEL` selects one (default `text_classification`). `hardware.yaml`, `perf.yaml` and perf results stay shared.
- **Model packs.** A pack is `packs/<pack>/pack.py` exposing `STAGES`, `METRICS` and optional `validate_config`, `lineage`, `load_predictor`, `targets`, `smoke`, `COMMANDS`, plus `serving_app.py`. The runner, gate, registry, tune loop and CLI are generic. The classifier moves to `packs/text_classification` (history kept).
- **Registry.** New MLflow backend: a registered version gets the approval alias (`champion`). `pull` downloads a version or alias.
- **Packaging and release.** `package` builds a pack's targets with a platform writer per kind (`triton-python`, deterministic `pytorch-dlc`), names them `<target>-<version>.tar.gz` and smoke-tests them. `release` uploads tars + `manifest.json` to a build store and never overwrites a version.
- **Serving.** A pack-agnostic FastAPI shell in `ops/serve` (one image per pack, allowlist build context).
- **SageMaker.** `train-remote` (one Training Job), `pipeline definition|upsert|start` (Train -> Evaluate -> Gate -> Register), `sm-step` inside jobs. Requests are built from the AWS API schemas and sent with boto3.
- **CI.** `ci.yml` (detect-secrets + tests), `pipeline.yml`, `release.yml`, `deploy.yml` (per GitHub environment, desired versions in `deploy/<model>/<env>.yaml`).
- **Removed.** `serve/deploy_sagemaker.py` + `deploy`, `serve/local.py`, `optimize.py`, `serve_analyzer.py` + `analyze`, `ops/serve-cpu`, `.env.test.example`, `.env.staging.example`, the `sagemaker` / `tritonclient` / `triton-model-analyzer` dev deps.

## Impact

- Config moved: `config/params.yaml` -> `config/text_classification/params.yaml` (same for `experiments.yaml`, `serving.yaml`); artifacts under `artifacts/text_classification/`.
- Imports moved: `datascience.components.*`, `datascience.model`, `datascience.metrics`, `datascience.infer.*` -> `packs.text_classification.*`; the perf entry is `packs.text_classification.perf_entry:target`.
- CLI: `deploy`, `analyze`, `optimize` are gone; `serve` runs the shell (`serve-triton` for the local Triton stack); new `package --version`, `release`, `pull`, `variants`, `train-remote`, `pipeline`, `sm-step`.
- Running on AWS needs the variables and OIDC roles listed in `docs/usage.md#ci-and-aws-setup`; endpoints themselves stay infra-as-code outside this repo.
