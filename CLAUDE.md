# CLAUDE.md — guidance for AI editing this repo

A **text-classification fine-tuning** pipeline: fine-tune a pretrained HF transformer, validate it
like an expert, gate it, version it, serve it. The pipeline is the product — generic via config; the
trainer/eval are concrete **reference code you copy & edit**, NOT a configurable framework.
Human overview, the pipeline diagram, and the command list: see [README.md](README.md).

## Working rules (follow these)

- **Lean, not defensive.** No "rule-dư": don't sprinkle guards / validation / retries / auto-detect
  everywhere — add a guard only where a real failure mode exists. Serving/ops code stays happy-path and
  config-driven. The trainer's robustness (auto-resume, OOM retry, NaN guard, atomic writes) is the
  deliberate bar; match it, don't exceed it elsewhere.
- **Reference, not framework.** `components/model_trainer.py` + `components/model_evaluation.py` are
  meant to be read and edited for a new task. Don't abstract them into a mega-config engine.
- **Opt-in addons.** The core runs on defaults; advanced capabilities (DAPT, LoRA, TensorBoard
  histograms, ONNX serving) are OFF by default, enabled via config. Keep anything new off-by-default.
- **Minimal deps.** Prod/serving (`requirements.txt`) stays torch-free + tiny — never add a
  training/dev dependency to it. No lint/format tooling; the only pre-commit hook is `detect-secrets`.
- **Paths are fixed.** Never hardcode an artifact path outside `constants/paths.py`.
- **GPU/Triton/SageMaker code can't run locally (no CUDA)** — write it clean for the GPU box; don't add
  laptop fallbacks. Don't pad responses/comments with "can't verify on Mac" caveats.

## Where things live

- `constants/paths.py` — ALL artifact paths + `MLFLOW_URI`, hardcoded & absolute. Not config.
- `config/` — split by concern (open only the file for the phase you're in):
  - `params.yaml` — base **experiment** config (data, model, training, metrics, gate, tracking, registry
    + an off-by-default `tuning:` block). This is what one `train` snapshots.
  - `experiments.yaml` — the tuning **loop**: named variants (sparse overrides on `params.yaml`).
  - `serving.yaml` — serving lifecycle: runtime, optimize, triton, sagemaker, analyze.
- `components/` — stage logic. `pipeline/stages.py` wires config→component; `cli.py` dispatches.
- `tune/` — Layer-2 model tuning: `loop.py` (run+rank variants), `promote.py` (gate→registry+lineage), `dapt.py` (R4).
- `serve_analyzer.py` — serving-config tuning (Model Analyzer). `serve/` — runtime servers (local FastAPI/ORT, triton, sagemaker).
- `model.py` — HF model/tokenizer build + load + atomic save. `metrics.py` — metric/calibration/CI helpers.
- `gate.py` — quality gate. `registry/` — local versioned catalog (remote = pluggable seam).
- `infer/predict.py` — single/batch/file inference (applies the saved calibration temperature).

## Conventions / invariants

- **Fine-tune, don't train from scratch.** `model.base_model` in params; swap it to change architecture.
- **No leakage.** Stratified (or group-aware) 3-way split; `val` selects, `test` is touched once. The
  leakage guard drops text duplicated across splits. The label map is built from TRAIN only.
- **Objective drives everything.** `metrics.objective` is both the best-model selection metric and what
  the gate reads. `compute_metrics` emits exactly `metrics.compute` (so `objective: mcc/pr_auc` works) —
  don't re-hardcode a fixed metric set. The gate reads metrics by NAME — task-agnostic.
- **Storage always; promotion gated.** The model is always saved to disk; only a model passing the gate
  is versioned in the registry. `gate.on_fail: error` fails the run (exit≠0) for CI.
- **Lineage.** A registered model records metrics + full config snapshot + MLflow run_id + `data_version`
  (content hash + row count). Keep passing these at BOTH register sites (`pipeline/stages.py`,
  `tune/promote.py`). Remote registry = seam (`registry/base.py._REMOTE_BACKENDS`), not built.
- **Tracking ≠ registry.** MLflow (local sqlite) only logs metrics/curves; the registry is the model store.

## Swap points (copy & edit)

- **Your data / task** → `ingestion.dataset_name: null` + drop a `{text,label}` JSONL at the data path
  (or change `dataset_name`). Any binary/N-class text-classification works as-is (label map + `num_labels`
  derived from your data; nothing assumes the demo's classes). Other shapes (multi-label, regression,
  NER) = copy & edit the two reference files: `components/model_trainer.py` + `components/model_evaluation.py`.
- **Different model** → `model.base_model`.   **One-off knob** → `params.yaml`.
- **Compare options** → add variants to `experiments.yaml`, `tune`, then `promote` the winner — don't
  hand-edit `params.yaml` run after run (that loses the record; the loop versions each variant).

## Gotchas (non-obvious — check before changing)

- **Serving runtime** `serving.runtime`: `torch` (DEFAULT, serves the trained model directly) vs `onnx`
  (opt-in; run `optimize`+`package` first). ONNX is never automatic — convert only when enabled.
- **`analyze` ≠ `tune`.** `tune/` tunes the MODEL (offline, pre-deploy). `serve_analyzer.py` (`analyze`)
  tunes SERVING PARAMS with Model Analyzer, POST-deploy, against REAL traffic
  (`serving.analyze.traffic_file`) on a prod-class GPU host. `bench` measures one config; `analyze` sweeps+picks.
- **The tuning loop clears `CKPT_DIR`** before each variant and on finish — else auto-resume loads a
  mismatched checkpoint into a later `train`.
- **Tracking**: MLflow is the default and overlays variants for comparison. TensorBoard is opt-in
  (`tracking.tensorboard` / `tracking.log_histograms`) — turn on only for weight/gradient histograms,
  which MLflow can't show.
- **Tuning ladder** (climb only when eval demands): R1 backbone · R2 data + `sanity` · R3 recipe
  (`tuning.freeze_encoder_layers` / `differential_lr`) · R4 DAPT (`dapt`) · R5 LoRA. R4/R5 are
  off-by-default code paths for the GPU box.

## Deferred (not built — don't assume these exist)

Lifecycle stages (staging/prod profiles + promote/rollback), production monitoring (drift + retrain
trigger), CI/CD.
