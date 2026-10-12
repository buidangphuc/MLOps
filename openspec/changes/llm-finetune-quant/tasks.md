## 1. Migrate the training side to causal-LM SFT

- [x] 1.1 Rewrite `config/params.yaml`: `model.base_model: Qwen/Qwen2.5-0.5B-Instruct`, `task:` (`system_prompt`, `prompt_template`), `finetune:` (`method: full`, `lora: {r, alpha, dropout, target_modules}`, `qat: {scheme}`), `perf: {apply_tuned: false}`; drop encoder-only `tuning.*` knobs; rename `tracking.experiment_name`/`registry.name`. Verify: YAML loads.
- [x] 1.2 Rewrite `config/configuration.py` and `entity/config_entity.py` coherently against the new params (no GLiNER sections/fields; `_VALID_METRICS` matches `metrics.compute`; method validation). Verify: test "Shipped params.yaml loads" and "Unknown fine-tune method rejected" pass.
- [x] 1.3 Add `infer/scoring.py`: prompt building via chat template, example encoding with prompt/padding labels = -100, length-normalised label log-likelihood scoring returning `[n, n_labels]`. Verify: tests for prompt masking and "Highest-likelihood label wins" with a tiny Qwen2Config model + fake tokenizer.
- [x] 1.4 Rewrite `model.py`: causal-LM `build_for_training(cfg)` applying `full|lora|qlora` (qat hook left for group 3), `load_trained(dir, device, scheme=None)`, adapter merge before `save_pretrained_atomic`. Verify: test "LoRA trains only adapters" on the tiny model.
- [x] 1.5 Update `components/model_trainer.py`: tokenization via `infer/scoring`, causal-LM padding collator, `supports("bf16","train")` instead of `_bf16_ok`, keep resume/NaN guard/OOM retry/atomic save, opt-in `perf.apply_tuned` hook via `training_args(load_tuned("train","train"))`. Verify: tests "Opt-in off" and "Result from another GPU".
- [x] 1.6 Update `components/model_evaluation.py` (`_predict_logits` → label scoring, `scheme` option) and `components/data_ingestion.py` to match the entity; `infer/predict.py` → `Predictor(model_dir, scheme=None)` on the shared scorer. Verify: test "Metrics file keeps gate keys" with a stubbed scorer; label-from-train test.
- [x] 1.7 `tune/dapt.py` → continued causal-LM training; `experiments.yaml` variants `baseline`, `lora`, `qlora`, `qat-int8`, commented `qwen-1.5b`; `pyproject.toml` `quant` extra (`bitsandbytes`, `torchao`). Verify: `pip install -e .[dev,cpu]` still resolves, base deps torch-free.
- [x] 1.8 make test green

## 2. PTQ schemes in the perf platform

- [x] 2.1 Add `perf/quant.py` (`SCHEMES`, `apply(model_or_dir, scheme)`, lazy imports, `Unsupported` on missing lib/device) and extend `hw._MIN_CC`. Verify: tests "T4 infer target" and "H100 infer target".
- [x] 2.2 `perf/workloads/infer.py`: filter precisions from `quant.SCHEMES`; `torch_precision` delegates non-dtype schemes to `quant.apply`; optional `new_tokens` → `tokens_per_s`. Verify: tests "Unsupported scheme never measured", "Backend cannot build a scheme", "Large quality drop rejected".
- [x] 2.3 `config/perf.yaml` infer precision grid `[fp32, fp16, bf16, int8-wo, int8-dyn, int4-wo, nf4, fp8]`.
- [x] 2.4 Add `infer/perf_entry.py` (`target()`, `train_target()`). Verify: test "Entry resolves" via `runner.resolve_entry` with the predictor stubbed.
- [x] 2.5 make test green

## 3. QAT

Owns: `src/datascience/model.py`, `src/datascience/components/model_trainer.py`, `src/datascience/config/configuration.py`, `src/datascience/entity/config_entity.py`, `tests/test_finetune.py`

- [x] 3.1 QAT method in `model.py`/trainer: QAT-capable schemes `int8-dyn`, `int4-wo`; `supports(scheme, "infer")` checked before any step; torchao prepare → train → convert; write `quant_scheme.json`. Verify: tests "Scheme unsupported by the serving GPU" and "Scheme not QAT-capable".
- [x] 3.2 `load_trained` applies a recorded `quant_scheme.json` unless a scheme is passed. Verify: test "Scheme recorded and reused" with `quant.apply` stubbed.
- [x] 3.3 make test green

## 4. Quant-vs-quality report

Owns: `src/datascience/perf/report.py` (new), `src/datascience/perf/cli.py`, `tests/test_report.py` (new)

- [x] 4.1 Add `perf/report.py` and `datascience perf report [--role infer]` in `perf/cli.py`: rows from the stored sweep (chosen marked, reasons shown) and tune runs; write `report.md`/`report.csv`; MLflow log when tracking. Verify: tests "Infer results only", "No stored results for this box", "Chosen and rejected rows".
- [x] 4.2 make test green

## 5. Docs and end-to-end check

- [x] 5.1 Update `.agents/ml-pipeline-repo/SKILL.md` (known-broken narrowed to serving/package, "fine-tune methods vs perf knobs" rule, schemes pointer, trainer hook wired) and README (`quant` extra, `perf report`, methods). Verify: no stale reference to `params.gliner` or `tuning.lora`.
- [ ] 5.2 CPU smoke: Qwen2.5-0.5B, `sample_size: 200`, `max_steps: 5`, run `ingest → evaluate` for `full` and `lora`; `perf tune infer --entry datascience.infer.perf_entry:target` with `fp32, int8-dyn`; `perf report`. Verify: metrics.json, stored result with quality_drop, printed report.
- [x] 5.3 make test green
