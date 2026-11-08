# MLOps: Multi-Model Platform + GPU Performance Platform

A **multi-model MLOps platform**: every model is a pack in `packs/`, and the platform runs it through
ingest → validate → train → evaluate → gate → register → package → release → deploy, locally or on
SageMaker, with deploys done by CI rather than from a laptop. Next to it, a **GPU perf platform**
(`datascience perf …`) tunes training throughput, inference and kernels. **Every tuned result is bound
to the GPU it was measured on**: training and inference GPUs are declared separately in
`config/hardware.yaml`, and a config tuned on GPU A is refused on GPU B.

## Scope

- **Platform** (`src/datascience`): per-model config and artifacts (`--model <id>`), the stage runner,
  a quality gate, the registry (local + MLflow with an approval alias, S3, staging/production
  pointers), a tuning loop, deterministic versioned packaging (`triton-python`, `pytorch-dlc`), an
  immutable release manifest, SageMaker Training Jobs and Pipelines, CI deploy to a multi-model
  endpoint, and a FastAPI serving shell.
- **Example pack** (`packs/text_classification`): fine-tunes a causal LM (Qwen2.5-0.5B-Instruct, AG
  News posed as generation), validates it (calibration, confidence intervals, error analysis).
  Fine-tune methods: `full`, `lora`, `qlora`, `qat`. Copy it to start a new model.

Philosophy (fastai-style): a minimal core that runs with sane defaults; every advanced capability is opt-in.
This repo is all core AI. A business app only calls the deployed endpoint.

## Flow

```text
train (laptop / GPU box | SageMaker Training Job | SageMaker Pipeline via CI)
  -> evaluate -> gate -> register (MLflow alias = approval)
  -> release (CI: versioned tars + manifest.json, immutable)
  -> deploy (CI, per environment: PR on deploy/<model>/<env>.yaml -> copy into the endpoint, smoke)
  -> rollback = revert the env file
```

## Layout

```text
src/datascience/   platform: paths, runner, gate, registry, tuning, packaging, release, cloud, perf
packs/             one dir per model (text_classification is the example)
config/            hardware.yaml + perf.yaml (shared), config/<model>/ per model
deploy/            desired release version per model and environment (applied by CI)
tests/             offline tests (no GPU, no downloads, no AWS); tests/toy is a minimal pack
docs/              usage.md, agents/ (rules and code map)
ops/               serve/ (FastAPI image), train/ (SageMaker job image), compose stacks
.github/workflows/ ci, pipeline, release, deploy
openspec/          specs and change proposals
```

## Quickstart

```bash
pip install -e .[dev,cpu]
python main.py run                      # full pipeline (downloads a sample dataset on first run)
python main.py infer --text "Central bank raises interest rates"
python main.py serve                    # FastAPI shell on the trained model
make test                               # offline, no GPU
```

## Status

Built: per-model packs, fine-tune methods, MLflow / S3 registry with pointers and rollback, versioned
packaging + release, SageMaker Training Job / Pipeline, CI deploy, GPU perf platform with PTQ schemes.
GPU smoke passed on a V100 on 2026-10-12; int4-wo, fp8 and bf16 are unverified (need sm_80+). The
SageMaker and deploy paths are tested offline against the AWS API schemas; they need an AWS account
with the roles in [docs/usage.md](docs/usage.md#ci-and-aws-setup) to run for real.
**Deferred:** wiring tuned perf configs into packaging, production monitoring, endpoint IaC.

## Docs

- [docs/usage.md](docs/usage.md): pipeline, commands, CI and AWS setup, install profiles, config files, GPU smoke runbook
- [docs/agents/pipeline.md](docs/agents/pipeline.md): model packs, lifecycle, code map, invariants
- [docs/agents/project.md](docs/agents/project.md): checks, tracker, rules for contributors
- [openspec/](openspec/): specs and changes
