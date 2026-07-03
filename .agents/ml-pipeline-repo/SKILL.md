---
name: ml-pipeline-repo
description: Understand and modify this ML fine-tuning pipeline template. Use for repo architecture questions, adding/adapting a pipeline for a new task, changing components/config/registry/serving, or reviewing pipeline conventions.
---

# ML Pipeline Repo

This repo-local skill is the canonical AI instruction for this pipeline
template. Read it before touching code; `CLAUDE.md`/`AGENTS.md` route here.

## What this repo is

A **text-classification fine-tuning** pipeline: ingest → validate →
transform → train → evaluate → **quality gate** → versioned registry, plus a
tuning loop and a torch-free serving path. The pipeline is the product —
generic via config. The trainer/evaluator are **reference code you copy &
edit** for a new task, NOT a configurable framework.

## Where things live

- `src/datascience/constants/paths.py` — ALL artifact paths + `MLFLOW_URI`,
  fixed & absolute. Never hardcode a path anywhere else.
- `config/` — `params.yaml` (base experiment: data/model/training/metrics/
  gate/tracking/registry + off-by-default `tuning:`), `experiments.yaml`
  (tuning-loop variants), `serving.yaml` (runtime/optimize/triton/sagemaker/
  analyze/monitor).
- `src/datascience/components/` — stage logic (ingestion, validation,
  transformation, trainer, evaluation). `pipeline/stages.py` wires
  config→component; `cli.py` dispatches (console script `datascience`).
- `src/datascience/utils/` — **shared, stable, task-agnostic helpers**:
  `split.py` (stratify check, group split, leakage guard), `quantize.py`
  (ONNX dynamic INT8; `matmul_only=True` for conv-patch models),
  `io.py`/`common.py`/`runtime.py`. `config/validation.py` holds shared
  config checks (split sizes, gate shape).
- `src/datascience/registry/` — `local.py` (versioned catalog v1,v2… +
  `latest.json`), `stages.py` (**staging/production pointers + rollback**;
  CLI `promote-stage`/`rollback`/`stage-path`), `base.py` +
  `remote_s3.py` (remote push to S3/MinIO via `registry.remote.backend: s3`;
  more backends plug into `_REMOTE_BACKENDS`).
- `src/datascience/tune/` — `loop.py` (run+rank variants), `promote.py`
  (gate→registry+lineage), `dapt.py` (R4).
- `src/datascience/serve/` — `local.py` (FastAPI, torch or ONNX; `--stage`
  resolves the dir a registry stage points at). `triton.py` +
  `deploy_sagemaker.py` are reference stubs. `serve_analyzer.py` = serving
  config sweep (GPU host).

  > `serve --stage` is an **open seam**: the CLI resolves the registry dir but
  > model loading never uses it — the default torch runtime loads `MODEL_DIR`
  > regardless (`_TorchModel` doesn't pass the dir to `Predictor(model_dir=…)`),
  > and the onnx runtime can't serve a registry dir (versions are torch model
  > dirs without `model.onnx`/`model_card.json`; `stage_path` also returns a
  > `str` where `local.py` expects a `Path`). Wire it per fork when
  > stage-based serving is actually needed.
- `src/datascience/{model,metrics,gate}.py` — HF factory + atomic save;
  metric/calibration/CI helpers; name-based quality gate (`on_fail: error`
  exits non-zero for CI).
- `tests/` — offline pure-logic tests; `conftest.py` provides
  `tmp_artifacts` (monkeypatches the fixed paths into a tmp tree);
  `pytest tests/ -q` needs no GPU/dataset.

## Working rules

- **Lean, not defensive.** Add a guard only where a real failure mode
  exists. Serving/ops code stays happy-path + config-driven. The trainer's
  robustness (auto-resume, OOM retry, NaN guard, atomic writes) is the bar —
  match it, don't exceed it elsewhere.
- **Reference vs shared utils — the line:** trainer/evaluator
  (`components/model_trainer.py`, `components/model_evaluation.py`) are
  copy-and-edit reference files — never generalize them into a config
  engine. Pure, stable, task-agnostic logic (splits, quantize, config
  checks, io) lives once in `utils/`/`config/validation.py` and is reused.
- **Opt-in addons.** DAPT, LoRA, TensorBoard histograms, ONNX serving are
  OFF by default. Keep anything new off-by-default.
- **Minimal deps.** The pyproject base (serving runtime) stays torch-free —
  training/dev deps go in the `dev` extra only. `cpu`/`gpu` extras are
  mutually exclusive onnxruntime wheels. No lint/format tooling; the only
  pre-commit hook is detect-secrets.
- **GPU/Triton/SageMaker code can't run locally** — write it clean for the
  GPU box; no laptop fallbacks, no "can't verify locally" comment padding.

## Conventions / invariants

- **Fine-tune, don't train from scratch** — swap `model.base_model`.
- **No leakage.** Stratified (or group-aware) 3-way split; `val` selects,
  `test` touched once; leakage guard drops cross-split text dups; label map
  built from TRAIN only.
- **Objective drives everything.** `metrics.objective` = best-model metric
  = what the gate reads; `compute_metrics` emits exactly `metrics.compute`.
  The gate reads metrics by NAME — task-agnostic.
- **Storage always; promotion gated.** Model always saved to disk; only a
  gate-passing model enters the registry. Stage pointers
  (staging/production) move only via `promote-stage`; `rollback` restores
  the previous holder.
- **Lineage.** Both register sites (`pipeline/stages.py`, `tune/promote.py`)
  record metrics + `data_version` (hash+rows); the base pipeline also passes
  `base_model` + `dataset`, and the **tune path additionally** records the
  MLflow `run_id` + full config snapshot (a plain `run` never captures a
  `run_id` and has no variant config to snapshot). Keep each site passing its
  fields.
- **Tracking ≠ registry.** MLflow logs metrics/curves; the registry is the
  model store.

## Adding / adapting a pipeline (the recipe)

1. Same task shape (any binary/N-class text classification): change
   `ingestion.dataset_name` or drop a `{text,label}` JSONL at the data path —
   no code change (label map + num_labels derive from data).
2. New task shape (multi-label, NER, vision…): copy & edit the two
   reference files (`components/model_trainer.py`,
   `components/model_evaluation.py`) + `model.py` as a sibling set; reuse
   `utils/` + `gate.py` + `registry/` unchanged; add paths in
   `constants/paths.py`; wire a stage list in `pipeline/` and dispatch in
   `cli.py`.
3. Compare options → variants in `experiments.yaml` → `tune` → `promote`
   the winner (never hand-edit params run-after-run).

## Verification (narrow → wide)

- `make test` (or `pytest tests/ -q`) — offline, no GPU/dataset.
- One stage smoke: `datascience run --only ingest` (console script;
  `python main.py …` is the same thing).
- Full DAG: `datascience run` (downloads the sample dataset on first run);
  the gate exits non-zero on failure — CI-safe.
- `datascience --help` lists every command.

## Gotchas (check before changing)

- `serving.runtime`: `torch` default; `onnx` opt-in (needs
  `optimize`+`package` first). Never auto-convert.
- `analyze` ≠ `tune`: tune/ tunes the MODEL offline; `analyze` sweeps
  SERVING params with Model Analyzer on a GPU host.
- The tuning loop clears `CKPT_DIR` between variants — else auto-resume
  loads a mismatched checkpoint.
- `monitor` is observe-only (PSI + confidence + length vs the eval
  reference profile) — WARN flags, no action.
- Tuning ladder: climb only when eval demands — R1 backbone · R2 data +
  `sanity` · R3 freeze/differential-LR · R4 DAPT · R5 LoRA.

## Deferred (not built — don't assume these exist)

The action half of monitoring (retrain trigger + live quality gate — needs
prod labels), CD (CI runs pytest only — a deploy target is a per-fork
decision, not a template one), additional remote registry backends beyond
s3 (mlflow/sagemaker stay open seams in `_REMOTE_BACKENDS`).
