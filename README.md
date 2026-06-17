# Text-Classification Fine-Tuning Pipeline

A production-grade pipeline that fine-tunes a pretrained Hugging Face transformer for text
classification, validates it like an expert (calibration, confidence intervals, per-class error
analysis), **gates** it on a metric, **versions** it with full lineage, and **serves** it.

The **pipeline is the product** — generic via config, not tied to any one model or dataset.
Philosophy (fastai-style): a minimal core that runs immediately with sane defaults; every advanced
capability (DAPT, LoRA, TensorBoard histograms, ONNX serving) is **opt-in**.

## Pipeline

```
ingest → validate (gate) → transform (stratified split + label map) → train (HF Trainer)
       → evaluate (metrics + calibration + CI + error analysis) → gate → local registry

tuning loop:  experiments.yaml (variants) → tune (train+eval+rank, 1 MLflow run each) → promote <winner>
serving:      torch (default)   or   optimize → package → serve / deploy  (Triton + SageMaker)
```

## Quickstart

```bash
pip install -r requirements-dev.txt          # dev / training stack

python main.py run                            # full pipeline (downloads a sample dataset on first run)
python main.py infer --text "Central bank raises interest rates"
python main.py serve --backend local --port 8080   # FastAPI endpoint: text -> {label, score}
```

Use your own data: set `ingestion.dataset_name: null` in `config/params.yaml` and drop a
`{"text": ..., "label": ...}` JSONL at `artifacts/data_ingestion/data.jsonl`. Any binary or
multi-class text classification works without code changes (label map + metrics are derived from
your data).

## Commands

| Command | What it does |
|---|---|
| `run` | full pipeline (slice with `--only` / `--from` / `--to`) |
| `ingest` · `validate` · `transform` · `train` · `eval` | a single stage |
| `sanity` | pre-flight: overfit-one-batch + loss@init |
| `tune` | train+eval+rank the variants in `experiments.yaml` |
| `promote <variant>` | gate-check + version the chosen variant (with lineage) |
| `dapt` | R4 domain-adaptive pretraining → adapted backbone (GPU box) |
| `infer` | predict `--text` or batch over `--input-file` (JSONL, resumable) |
| `optimize` | (opt-in) export the trained model to ONNX |
| `package` | build a deployable bundle or a Triton model repository |
| `serve` | run an endpoint (`--backend local` or `triton`) |
| `deploy` | deploy to a SageMaker Triton endpoint |
| `bench` · `analyze` | measure one serving config · sweep configs (Model Analyzer) |

`--gpu N` on any stage sets `CUDA_VISIBLE_DEVICES`. `dvc repro` runs the same DAG with caching.

## Architecture

Three layers: **(1) Pipeline** — one config → one reproducible result · **(2) Tuning** — a loop over
variants, ranked + promoted with lineage · **(3) Lifecycle** (planned) — staging/prod profiles +
promote/rollback.

Two repos, boundary = the deployed endpoint. **This repo = all core AI** (data → train → tune → eval
→ optimize → package → serve/deploy). A business app (e.g. `fastapi_template`) only **calls** the
endpoint (text in → `{label, score}`); it holds no model internals.

## Install profiles

| Use | Install |
|---|---|
| Dev / training | `pip install -r requirements-dev.txt` |
| Serving — torch (default) | the model stack (torch + transformers; = the dev install) |
| Serving — ONNX (opt-in, torch-free) | `requirements.txt` + `requirements-cpu.txt` (CPU) **or** `requirements-gpu.txt` (GPU) |

## Config (`config/`)

- `params.yaml` — the base experiment: data, model, training, metrics, gate, tracking, registry (+ an off-by-default `tuning:` block).
- `experiments.yaml` — the tuning loop: named variants (sparse overrides on `params.yaml`).
- `serving.yaml` — serving lifecycle: runtime, optimize, triton, sagemaker, analyze.

Artifact paths + the MLflow URI are fixed in `src/datascience/constants/paths.py` (not config).
Tracking is local MLflow: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Status

Built + verified: full pipeline, tuning loop (`tune`/`promote` with lineage), torch & ONNX serving,
local FastAPI/ONNX-Runtime. Triton / SageMaker / GPU paths run on a CUDA box. **Deferred:** lifecycle
stages (staging/prod + rollback), production monitoring (drift + retrain), CI/CD.

Conventions for editing the code live in [CLAUDE.md](CLAUDE.md).
