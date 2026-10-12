---
name: ml-pipeline-repo
description: Rules and map for this MLOps repo (fine-tuning pipeline + serving/registry platform + GPU perf platform). Use before changing code, configs, serving, registry, or perf tuning, and for architecture questions.
---

# MLOps repo

This file is where the rules for this repo live. `CLAUDE.md` routes here; when a convention changes, edit this file.

## What this repo is

Three layers share one package (`src/datascience`, imported as `datascience`):

1. **Fine-tuning pipeline**: ingest → validate → transform → train → evaluate → gate → registry, plus a tuning loop (`tune/`). The trainer and evaluator are **reference code you copy and edit** for a new task. They are not a config engine.
2. **Serving and registry platform**:
   - FastAPI CPU serve (`serve/local.py`)
   - Triton python-backend MME (`package.py`)
   - SageMaker deploy (`serve/deploy_sagemaker.py`)
   - S3 registry with local versions (`registry/`)
   - staging/production pointers with rollback (`registry/stages.py`)
3. **GPU perf platform** (`perf/`): tuning for training throughput, inference, profiling and kernel autotuning. It is task-agnostic and **bound to hardware** (see the "Perf platform" section).

**Known broken:** on `main`, train, eval and infer are half-migrated from text classification to GLiNER NER:
- `configuration.py` reads `params.gliner`, which `params.yaml` does not have.
- `model.export_gliner_onnx` and `load_labels` are missing.
- `Predictor` has the wrong signature.

The full migration lives on the `mlop-gliner` branch and has not been merged. Until it is fixed, `perf/` must not import the trainer, predictor or packager. Models reach `perf/` only through `--entry`.

## Where things live

- `constants/paths.py`: every artifact path, `MLFLOW_URI`, and the config file paths. Paths are absolute and never hardcoded anywhere else.
- `config/`, one file per concern:
  - `params.yaml`: the experiment.
  - `experiments.yaml`: tuning variants.
  - `serving.yaml`: serving lifecycle.
  - `hardware.yaml`: the train and infer GPU targets plus SageMaker instance types.
  - `perf.yaml`: perf grids, objectives and constraints.
- `components/` holds stage logic. `pipeline/stages.py` wires config to components. `cli.py` dispatches and imports the training stack only for stage commands.
- `registry/`:
  - `base.register`: local always, remote through `_REMOTE_BACKENDS` (`s3`), and records `train_hw`.
  - `s3.py`: push (training side) and pull (serving side).
  - `stages.py`: `promote-stage`, `rollback`, `stage-path`.
- `perf/`:
  - `hw.py`: fingerprint and role binding.
  - `store.py`: hardware-bound results.
  - `runner.py`: sweeps.
  - `workloads/{infer,train,kernel}.py`
  - `profile.py`
  - `kernels/`: copy-and-edit OpenAI-Triton kernels.
  - `cli.py`: `datascience perf …`
- `tests/`: offline pure-logic tests. The `tmp_artifacts` fixture in `conftest.py` redirects artifact paths. Tests run without a GPU or a dataset.

## Working rules

- **Lean.** Add a guard only where a real failure mode exists. Serving, ops and perf code stays happy-path and config-driven. The trainer's robustness (auto-resume, OOM retry, NaN guard, atomic writes) is the bar to match.
- **Reference vs shared.** Trainer and evaluator files are copied and edited per task. Pure, stable, task-agnostic logic lives once in `utils/`, `registry/` or `perf/` and is reused.
- **Opt-in.** New capabilities are off by default (DAPT, LoRA, TensorBoard histograms, ONNX serving, the `trt` extra, and applying tuned perf configs).
- **Minimal deps.** The pyproject base and `requirements.txt` (the serving runtime) stay torch-free; keep the two in sync. Training, dev and perf deps go in the `dev` extra, and TensorRT goes in `trt`. The `cpu` and `gpu` extras are mutually exclusive ORT wheels. The only pre-commit hook is detect-secrets.
- **GPU, Triton and SageMaker code** is written clean for the GPU box, with no laptop fallbacks.

## Invariants

- **Fine-tune, don't train from scratch.** To change architecture, swap `model.base_model`.
- **No leakage.** `val` selects, `test` is touched once, and labels/entity types come from TRAIN only.
- **Objective drives everything.** `metrics.objective` is the best-model metric and also what the gate reads by NAME, so the gate stays task-agnostic.
- **Storage always; promotion gated.** A model is always saved to disk. Only a model that passes the gate enters the registry. Stage pointers move only through `promote-stage`, and `rollback` restores the previous holder.
- **Lineage.** Both register sites (`pipeline/stages.py`, `tune/promote.py`) pass metrics and `data_version`. The tune path also passes the MLflow `run_id` and a config snapshot. `registry.base.register` adds `train_hw` centrally.
- **Tracking ≠ registry.** MLflow logs metrics and curves, with perf runs in experiment `perf`. The registry is the model store.

## Perf platform

- **Hardware binding** is the core invariant:
  - Each result is stored at `artifacts/perf/<role>/<hw_id>/<workload>.json`.
  - `hw_id` is built from GPU model, compute capability, VRAM and count. Software versions are recorded but not part of the key.
  - `load_tuned` refuses a result that was tuned on another GPU, so re-tune on the target box.
  - `sweep` and `perf tune` call `assert_role` against `hardware.yaml targets.<role>`, so the infer config is tuned on the infer GPU.
- **Capability comes from the target, not the box.** `supports("bf16", "infer")` reads `targets.infer`; a T4 never gets bf16 even when the sweep runs on an A100.
- **Effective batch is constant** in `perf tune train`: per-device batch × grad_accum × GPUs = `effective_batch`. Tuning changes speed only.
- **Quality guard.** Infer candidates must satisfy `quality_drop ≤` the constraint, measured against the first backend at fp32. Kernel configs must match the torch reference within `max_abs_err`.
- **Inputs are traffic, not knobs.** Infer measures every length bucket and scores the worst one. Kernels run one sweep per shape.
- **Entry:** `--entry module:fn` supplies the model or trainer. See the workload docstrings for the contract. A new fine-tuning experiment plugs in here without edits to `perf/`.
- **perf, tune, analyze:**
  - `perf` tunes speed on hardware.
  - `tune/` tunes model quality.
  - `analyze` tunes Triton serving params against real traffic after deploy.
- **Not wired yet** (needs the broken training side fixed first):
  - the trainer applies `load_tuned("train", "train")` via `perf.workloads.train.training_args`
  - `package_triton` writes precision, batch and `infer_hw` into `serving_meta.json` and `config.pbtxt`
  - serve refuses an `infer_hw` mismatch
  - `serve_analyzer` targets the GLiNER input name

## Verification (narrow → wide)

1. `make test`: offline, no GPU.
2. `datascience perf fingerprint`: shows whether this box matches each target.
3. On the GPU box: `datascience perf tune infer|train --entry …`, `perf profile …`, `perf tune kernel layernorm`.

## Gotchas

- `serving.runtime` defaults to `torch`; `onnx` is opt-in and needs `optimize` and `package` first. Never auto-convert.
- The tuning loop clears `CKPT_DIR` between variants. Otherwise auto-resume loads a mismatched checkpoint.
- The CPU serving image runs from source with `PYTHONPATH=/app/src`. Everywhere else, install with `pip install -e .`.

## Deferred (not built)

- Monitoring and retrain triggers.
- CI/CD (there is no workflow on `main`).
- Remote registry backends beyond s3.
- The perf hooks listed under "Not wired yet".
