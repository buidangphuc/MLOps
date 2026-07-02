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
pip install -e .[dev,cpu]                    # dev / training stack (editable)

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
| `monitor` | drift check on captured prod traffic vs the eval reference (warn-only) |
| `promote-stage <name> <vN> --to staging\|production` | point a lifecycle stage at a registered version |
| `rollback <name> --stage production` | restore the stage's previous version |
| `stage-path <name> <stage>` | print the model dir a stage points at (deploy tooling) |

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
| Dev / training | `pip install -e .[dev,cpu]` (editable; requirements*.txt are legacy pointers) |
| Serving — torch (default) | the model stack (torch + transformers; = the dev install) |
| Serving — ONNX (opt-in, torch-free) | `pip install '.[cpu]'` **or** `'.[gpu]'` (mutually exclusive onnxruntime wheels) |

## Config (`config/`)

- `params.yaml` — the base experiment: data, model, training, metrics, gate, tracking, registry (+ an off-by-default `tuning:` block).
- `experiments.yaml` — the tuning loop: named variants (sparse overrides on `params.yaml`).
- `serving.yaml` — serving lifecycle: runtime, optimize, triton, sagemaker, analyze.

Artifact paths + the MLflow URI are fixed in `src/datascience/constants/paths.py` (not config).
Tracking is local MLflow: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Optional capabilities (opt-in)

The core runs on defaults; everything below is **off unless you turn it on** (fastai-style — add a
capability only when you have a concrete need).

| Capability | Turn on | Notes |
|---|---|---|
| **TensorBoard** scalars | `tracking.tensorboard: true` | runs alongside MLflow |
| **Weight/gradient histograms** (TB-only) | `tracking.log_histograms: true` | diagnose vanishing/exploding grads; `tensorboard --logdir logs/tensorboard` |
| **R3 — freeze encoder layers** | `tuning.freeze_encoder_layers: N` | freeze embeddings + bottom N layers |
| **R3 — differential LR** | `tuning.differential_lr: true` | lower LR for encoder, full for head |
| **R5 — LoRA / PEFT** | `tuning.lora: true` (+ `lora_target_modules` for distilbert) | small adapters instead of full weights |
| **R4 — DAPT** | `tuning.dapt.corpus: <jsonl>` → `python main.py dapt` | domain-adaptive pretraining → adapted backbone |
| **Group-aware split** | `data.group_field: <field>` | no leakage across groups |
| **Per-segment metrics** | `metrics.slice_field: <field>` | metrics per slice |
| **Objective metric** | `metrics.objective` (`accuracy`/`macro_f1`/`micro_f1`/`pr_auc`/`mcc`) | drives best-model + gate |
| **ONNX serving** (default torch) | `serving.runtime: onnx` | run `optimize`+`package` first; torch-free |
| **INT8 quantization** | `serving.optimize.quantize: dynamic-int8` | ONNX path only |
| **Triton config sweep** | `python main.py analyze` (+ `serving.analyze.traffic_file`) | real traffic = accurate; null = mock |
| **Drift monitor** | `python main.py monitor` (+ `serving.monitor.traffic_file`) | warn-only; needs no labels |
| **Remote registry** | `registry.remote.backend: mlflow\|sagemaker` | seam — not connected yet |

Toggleable but **on by default** (turn off in config if unwanted): the quality `gate`, calibration,
per-class report, error dump, bootstrap CI, leakage check.

## Status

Built + verified: full pipeline, tuning loop (`tune`/`promote` with lineage), torch & ONNX serving,
local FastAPI/ONNX-Runtime, drift monitoring (`monitor`, warn-only). Triton / SageMaker / GPU paths run
on a CUDA box. **Deferred:** lifecycle stages (staging/prod + rollback), the *action* half of monitoring
(retrain trigger + live quality gate — needs labels), CI/CD.

**Scaling note (optional, not built):** SageMaker **Multi-Model Endpoint (MME)** is a future *deploy
target*, not a pipeline change — run the pipeline N times → N self-contained tars → one MME. Reach for
it only with **many models + sporadic traffic** (one endpoint, cheaper); a single steady model stays
**SingleModel** (always warm, no cold-start). If you do go MME, keep tars **small** (ONNX, no bundled
torch) — big tars make MME's on-demand model loads slow.

**Reference-only paths:** `serve/triton.py` and `serve/deploy_sagemaker.py` are
thin stubs sketching the GPU-box/AWS route — the implemented serving path is
`serve` with the local FastAPI backend (torch or ONNX).

Conventions for editing the code live in [CLAUDE.md](CLAUDE.md).
