## Context

See proposal.md for motivation. Current state that shapes the approach:
- `components/data_transformation.py` already does the stratified/group split, the cross-split leakage guard and a TRAIN-only label map. It stays; tokenization happens in the trainer.
- `components/model_evaluation.py` turns a `[n, n_labels]` logits matrix into metrics, per-class report, bootstrap CI, temperature calibration, error dump and slices. Everything after `_predict_logits` is reused unchanged.
- `perf/` is task-agnostic: `runner.sweep` stores trials with `feasible`/`reason`; `workloads/infer.py` filters precisions via `hw.supports` and an `_FEATURE` map; `workloads/train.training_args` maps a tuned config to TrainingArguments kwargs; `store.load_tuned` raises `HardwareMismatch`.
- Rules live in `.agents/ml-pipeline-repo/SKILL.md` (lean, opt-in, torch-free base deps, paths only in `constants/paths.py`, perf must not import the trainer/predictor — the entry lives outside `perf/`).

## Goals / Non-Goals

**Goals:**
- `ingest → validate → transform → train → evaluate → gate → registry` runs end to end on Qwen2.5-0.5B, including on CPU with small `sample_size`/`max_steps`.
- Every scheme and method from the specs is reachable from config/CLI and verified offline where it does not need a GPU.

**Non-Goals:**
- Serving/packaging migration (GLiNER-shaped `serve/local.py`, `package.py`, `optimize.py` stay as is).
- Free-text generation quality metrics (BLEU, judge models); GPTQ/AWQ.
- Making `perf tune train` vary the fine-tune method.

## Decisions

1. **Speed knobs vs quality knobs.** Fine-tune method changes what is learned, so it is a `params.yaml` setting compared via `tune` variants; the perf train sweep keeps its "speed only, constant effective batch" invariant and measures whatever method is configured. PTQ schemes are infer precisions because the infer sweep already carries a quality guard. *Alternative:* method as a perf-grid dimension — rejected, it would let a speed sweep pick a model that learns differently.

2. **AG News as generation, scored by label log-likelihood.** For each text, compute `sum log p(label_tokens | prompt)` for every candidate label (length-normalised by token count to avoid favouring short names) in one batched forward per (text-chunk × label). The `[n, n_labels]` matrix feeds the existing metrics code as logits. *Alternative:* `generate()` + string parsing — rejected, nondeterministic and needs a parser for off-label output.

3. **One shared scoring helper** (`infer/scoring.py`: prompt building, example encoding with -100 masking, label scoring) used by transform/trainer, evaluator, predictor and the perf entry, so training and inference prompts never drift.

4. **Schemes table in `perf/quant.py`.** `SCHEMES[name] = (feature, kind)`; `hw._MIN_CC` gains `int8: 7.0`, `int4: 8.0`, `nf4: 7.0` (bf16/fp16/fp8 exist). Kinds: `dtype` (torch cast), `torchao` (`quantize_` with `Int8WeightOnlyConfig`, `Int8DynamicActivationInt8WeightConfig`, `Int4WeightOnlyConfig`, `Float8DynamicActivationFloat8WeightConfig`), `load` (bnb `BitsAndBytesConfig` at `from_pretrained`, so `apply` takes a model dir for these). Missing library or unsupported device → `Unsupported`, which the runner already records as infeasible. `workloads/infer.py` replaces `_FEATURE` with the table. *Alternative:* separate quant workload — rejected, duplicates the sweep and the quality guard.

5. **QLoRA** = bnb nf4 base + `peft.prepare_model_for_kbit_training` + LoRA; on save, reload the base in the train dtype, merge adapters, save a plain directory. **LoRA** merges directly.

6. **QAT** uses torchao `QATConfig(base_config, step="prepare")` before `Trainer.train()` and `step="convert"` after; the scheme is checked with `supports(scheme, "infer")` before any step. `quant_scheme.json` in the model dir records the scheme; `load_trained` applies it unless overridden.

7. **Opt-in tuned perf hook.** `params.yaml perf.apply_tuned: false`. When true the trainer merges `training_args(load_tuned("train", "train"))` over its own args; `HardwareMismatch` propagates.

8. **Entry outside perf.** `infer/perf_entry.py` exposes `target()` (infer contract: `predictor`, `inputs`, `score`, `lengths`, `new_tokens=None`) and `train_target()` (`build_trainer(overrides)`), keeping `perf/` free of pipeline imports.

9. **Report is pure formatting** over `store.load` + `paths.TUNE_DIR` files; writes `report.md`/`report.csv` under `PERF_DIR/infer/<hw_id>/`.

10. **Offline tests** use a tiny `Qwen2Config` model built from config and a fake word-level tokenizer — no downloads; GPU-only kinds (torchao CUDA kernels, bnb) are tested only through the capability table and the `Unsupported` path.

## Risks / Trade-offs

- [torchao/bnb APIs move fast] → pin minimum versions in the `quant` extra; schemes import lazily and map import errors to `Unsupported`.
- [Label log-likelihood is O(n_labels) forwards] → 4 labels for AG News; batch texts × labels; acceptable. Document the cost for large label sets.
- [CPU smoke on 0.5B is slow] → smoke uses `sample_size: 200`, `max_steps: 5`, `max_length: 128`.
- [Serving still GLiNER-shaped] → SKILL.md "Known broken" narrows to serving/package; separate PR.
- [Removing `tuning.*` knobs breaks old variants] → `experiments.yaml` is rewritten in the same change.

## Migration Plan

Branch `feat/llm-quant` from `main`, one commit per task group, `make test` green after each. Rollback = revert the PR; no stored artifact formats change except new `quant_scheme.json` and report files.
