---
name: ml-pipeline-repo
description: Rules and map for this MLOps repo (fine-tuning pipeline + serving/registry platform + GPU perf platform). Use before changing code, configs, serving, registry, or perf tuning, and for architecture questions.
---

# MLOps repo

This file is where the rules for this repo live. `CLAUDE.md` routes here; when a convention changes, edit this file.

## What this repo is

Three layers share one package (`src/datascience`, imported as `datascience`):

1. **Fine-tuning pipeline**: ingest → validate → transform → train → evaluate → gate → registry, plus a tuning loop (`tune/`). It is a causal-LM SFT (Qwen2.5-0.5B-Instruct) with AG News posed as generation: the label is the completion, and prediction picks the label with the highest log-likelihood (`infer/scoring.py`, shared by train, eval, predict and the perf entry). The trainer and evaluator are **reference code you copy and edit** for a new task. They are not a config engine.
2. **Serving and registry platform**:
   - FastAPI CPU serve (`serve/local.py`)
   - Triton python-backend MME (`package.py`)
   - SageMaker deploy (`serve/deploy_sagemaker.py`)
   - S3 registry with local versions (`registry/`)
   - staging/production pointers with rollback (`registry/stages.py`)
3. **GPU perf platform** (`perf/`): tuning for training throughput, inference, profiling and kernel autotuning. It is task-agnostic and **bound to hardware** (see the "Perf platform" section).

**Known broken (serving and packaging only):** `serve/local.py`, `package.py`, `optimize.py` and `serve_analyzer` are still GLiNER-shaped. For example `Predictor(..., onnx=True)` and `ev.flat_ner` / `ev.max_width` no longer exist. Train, eval, infer, tune and perf work. Do not build on those four files before they are ported.

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
  - `quant.py`: PTQ `SCHEMES` and `apply`.
  - `report.py`: quant-vs-quality table.
  - `profile.py`
  - `kernels/`: copy-and-edit OpenAI-Triton kernels.
  - `cli.py`: `datascience perf …`
- `infer/`: `scoring.py` (label log-likelihood), `predict.py` (`Predictor`), `perf_entry.py` (real-model perf entries; kept out of `perf/`).
- `tests/`: offline pure-logic tests. The `tmp_artifacts` fixture in `conftest.py` redirects artifact paths. Tests run without a GPU or a dataset.

## Working rules

- **Lean.** Add a guard only where a real failure mode exists. Serving, ops and perf code stays happy-path and config-driven. The trainer's robustness (auto-resume, OOM retry, NaN guard, atomic writes) is the bar to match.
- **Reference vs shared.** Trainer and evaluator files are copied and edited per task. Pure, stable, task-agnostic logic lives once in `utils/`, `registry/` or `perf/` and is reused.
- **Opt-in.** New capabilities are off by default (DAPT, LoRA/QLoRA/QAT, TensorBoard histograms, ONNX serving, the `trt` and `quant` extras, and `perf.apply_tuned`).
- **Minimal deps.** The pyproject base and `requirements.txt` (the serving runtime) stay torch-free; keep the two in sync. Training, dev and perf deps go in the `dev` extra, TensorRT in `trt`, bitsandbytes and torchao in `quant`. The `cpu` and `gpu` extras are mutually exclusive ORT wheels. The only pre-commit hook is detect-secrets.
- **GPU, Triton and SageMaker code** is written clean for the GPU box, with no laptop fallbacks.

## Invariants

- **Fine-tune, don't train from scratch.** To change architecture, swap `model.base_model`.
- **No leakage.** `val` selects, `test` is touched once, and label names come from TRAIN only.
- **Objective reports, loss selects.** The trainer keeps the best checkpoint on `eval_loss`. `metrics.objective` is what evaluation reports and what the gate and `tune` ranking read by NAME, so they stay task-agnostic.
- **Storage always; promotion gated.** A model is always saved to disk. Only a model that passes the gate enters the registry. Stage pointers move only through `promote-stage`, and `rollback` restores the previous holder.
- **Lineage.** Both register sites (`pipeline/stages.py`, `tune/promote.py`) pass metrics and `data_version`. The tune path also passes the MLflow `run_id` and a config snapshot. `registry.base.register` adds `train_hw` centrally.
- **Tracking ≠ registry.** MLflow logs metrics and curves, with perf runs in experiment `perf`. The registry is the model store.

## Fine-tune methods vs quantization schemes

- **Methods** (`full|lora|qlora|qat`, `params.yaml finetune`) are quality knobs, compared as `tune` variants in `experiments.yaml`.
- **PTQ schemes** are infer precisions in the `perf tune infer` sweep, under the quality guard. See `perf/quant.py` `SCHEMES` for the table. The `perf tune train` sweep stays speed-only.
- **Filtered by the role TARGET** capability (`hardware.yaml`), never by this box.
- **QAT** scheme must be supported by `targets.infer`. QAT saves float weights plus `quant_scheme.json`; `model.load_trained` applies the recorded scheme, and that is the convert step.

## Perf platform

- **Hardware binding** is the core invariant:
  - Each result is stored at `artifacts/perf/<role>/<hw_id>/<workload>.json`.
  - `hw_id` is built from GPU model, compute capability, VRAM and count. Software versions are recorded but not part of the key.
  - `load_tuned` refuses a result that was tuned on another GPU, so re-tune on the target box.
  - `sweep` and `perf tune` call `assert_role` against `hardware.yaml targets.<role>`, so the infer config is tuned on the infer GPU.
- **Capability comes from the target, not the box.** `supports("bf16", "infer")` reads `targets.infer`; a T4 never gets bf16 even when the sweep runs on an A100.
- **Effective batch is constant** in `perf tune train`: per-device batch × grad_accum × GPUs = `effective_batch`. Tuning changes speed only.
- **Apply tuned:** `perf.apply_tuned` (off by default) makes the trainer merge `load_tuned("train", "train")`; a result from another GPU stops training.
- **Quality guard.** Infer candidates must satisfy `quality_drop ≤` the constraint, measured against the first backend at fp32. Kernel configs must match the torch reference within `max_abs_err`.
- **Inputs are traffic, not knobs.** Infer measures every length bucket and scores the worst one. Kernels run one sweep per shape.
- **Entry:** `--entry module:fn` supplies the model or trainer. See the workload docstrings for the contract. A new fine-tuning experiment plugs in here without edits to `perf/`.
- **perf, tune, analyze:**
  - `perf` tunes speed on hardware.
  - `tune/` tunes model quality.
  - `analyze` tunes Triton serving params against real traffic after deploy.
- **Not wired yet** (needs the serving side ported):
  - `package_triton` writes precision, batch and `infer_hw` into `serving_meta.json` and `config.pbtxt`
  - serve refuses an `infer_hw` mismatch
  - `serve_analyzer` targets the GLiNER input name

## Verification (narrow → wide)

1. `make test`: offline, no GPU.
2. `datascience perf fingerprint`: shows whether this box matches each target.
3. On the GPU box: `datascience perf tune infer --entry datascience.infer.perf_entry:target`, `perf tune train --entry datascience.infer.perf_entry:train_target`, `perf profile …`, `perf tune kernel layernorm`.
4. `datascience perf report`: quant-vs-quality table from the stored infer sweep (writes `report.md` / `report.csv` beside it).

## Gotchas

- transformers>=5 API: the trainer passes `warmup_steps` as a ratio and `dtype=`, not the 4.x names.
- The `quant` extra is GPU-only. bitsandbytes and torchao import lazily; when missing, the scheme raises `Unsupported` and the sweep records it infeasible.
- `serving.runtime` defaults to `torch`; `onnx` is opt-in and needs `optimize` and `package` first. Never auto-convert.
- The tuning loop clears `CKPT_DIR` between variants. Otherwise auto-resume loads a mismatched checkpoint.
- The CPU serving image runs from source with `PYTHONPATH=/app/src`. Everywhere else, install with `pip install -e .`.

## Deferred (not built)

- Monitoring and retrain triggers.
- CI/CD (there is no workflow on `main`).
- Remote registry backends beyond s3.
- The serving hooks listed under "Not wired yet".
