# MLOps: Fine-Tuning Pipeline + GPU Performance Platform

Two products in one repo: a fine-tuning **pipeline** (ingest → validate → train → evaluate → gate → version → serve)
and a **GPU perf platform** (`datascience perf …`) that tunes training throughput, inference and kernels, and
profiles any workload. **Every tuned result is bound to the GPU it was measured on**: training and inference GPUs
are declared separately in `config/hardware.yaml`, and a config tuned on GPU A is refused on GPU B.

## Scope

The pipeline fine-tunes a causal LM (Qwen2.5-0.5B-Instruct, AG News posed as generation), validates it
(calibration, confidence intervals, error analysis), gates it on a metric and versions it with lineage.
Fine-tune methods: `full`, `lora`, `qlora`, `qat`. It is generic via config, not tied to one model or dataset.
Philosophy (fastai-style): a minimal core that runs with sane defaults; every advanced capability is opt-in.

This repo is all core AI. A business app only calls the deployed endpoint (text in, `{label, score}`).

## Layout

```text
src/datascience/   pipeline components, tuning, registry, serving, perf platform
config/            params, experiments, serving, hardware, perf (one file per concern)
tests/             offline tests (no GPU, no downloads)
docs/              usage.md, agents/ (rules and code map)
ops/               docker compose stacks, serving images
openspec/          specs and change proposals
```

## Quickstart

```bash
pip install -e .[dev,cpu]
python main.py run                      # full pipeline (downloads a sample dataset on first run)
python main.py infer --text "Central bank raises interest rates"
make test                               # offline, no GPU
```

## Status

Built: fine-tune methods, S3 registry with staging/production pointers and rollback, GPU perf platform with PTQ
schemes. GPU smoke passed on a V100 on 2026-10-12; int4-wo, fp8 and bf16 are unverified (need sm_80+).
**Broken:** serving/packaging (`serve/local.py`, `package.py`, `optimize.py`, `serve_analyzer` are still GLiNER-shaped).
**Deferred:** wiring tuned perf configs into package/serve, production monitoring, CI/CD.

## Docs

- [docs/usage.md](docs/usage.md): pipeline, commands, install profiles, config files, GPU smoke runbook
- [docs/agents/project.md](docs/agents/project.md): checks, tracker, rules for contributors
- [openspec/](openspec/): specs and changes
