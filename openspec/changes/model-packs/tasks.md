## 1. Per-model trees

- [x] 1.1 `constants/paths.py`: `ModelPaths`, `for_model`, `use`, `active`, module `__getattr__`; shared hardware / perf paths
- [x] 1.2 `git mv config/{params,experiments,serving}.yaml config/text_classification/`; repoint `dvc.yaml` and the Triton compose
- [x] 1.3 make test green

## 2. Pack seam

- [x] 2.1 `datascience/pack.py` (loader, `pack:` key, `Target`); generic `pipeline/stages.py`; `cli.py` pre-parses `--model`
- [x] 2.2 Move the classifier into `packs/text_classification` with `config.py` (its config sections and validation) and `pack.py`
- [x] 2.3 Generic `tune/{variants,loop,promote}.py`; the trainer writes `mlflow_run.json`
- [x] 2.4 Delete `deploy_sagemaker.py`, `serve/local.py`, `optimize.py`, `serve_analyzer.py`
- [x] 2.5 make test green

## 3. Registry

- [x] 3.1 `registry/mlflow_registry.py` (push with alias, pull) wired as `remote.backend: mlflow`
- [x] 3.2 `tests/toy/pack.py` + `toy_model` fixture; tests for run/gate/register, tune, promote, CLI
- [x] 3.3 make test green

## 4. Packaging, release, serving

- [x] 4.1 Generic `package.py` (`triton-python`, `pytorch-dlc`, smoke) and `serve/release.py` (manifest, immutable versions, `RELEASE_BUILD_STORE`)
- [x] 4.2 `ops/serve` shell + Dockerfile (`ARG PACK`) + allowlist dockerignore; `serve` command
- [x] 4.3 Example pack: `serving_meta.json`, `serving.py`, `serving_app.py`, `dlc/inference.py`, `targets`, `smoke`
- [x] 4.4 Tests: deterministic tars, manifest, no overwrite, shell contract, DLC smoke on the tiny model; image file list has no artifacts or env files
- [x] 4.5 make test green

## 5. SageMaker

- [x] 5.1 `cloud/{config,steps,train_job,pipeline}.py`; `train-remote`, `pipeline`, `sm-step`; `ops/train` image
- [x] 5.2 Tests: requests validated against botocore's SageMaker model, gate branches, `sm-step` round trip
- [x] 5.3 make test green

## 6. CI

- [x] 6.1 `ci.yml`, `pipeline.yml`, `release.yml`, `deploy.yml`; `serve/deploy.py`; `deploy/text_classification/*`; `pull`
- [x] 6.2 Tests for plan / sync / pull; actionlint clean
- [x] 6.3 make test green

## 7. Docs

- [x] 7.1 README, `docs/usage.md`, `docs/agents/pipeline.md`, `docs/agents/project.md`
- [x] 7.2 make test green; `pre-commit run --all-files` passes
