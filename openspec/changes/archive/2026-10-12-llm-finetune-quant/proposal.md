## Why

The perf platform (PR #3) is exercised only by fakes, and the training side on `main` is broken (half text-classification, half GLiNER: `configuration.py` requires a missing `params.gliner`). Quantization and fine-tuning techniques are the next platform layer, and they need a real model to prove they work end to end.

## What Changes

- **BREAKING** The fine-tuning pipeline (config, transform, train, evaluate, predict) migrates from a SequenceClassification model to a causal-LM SFT of `Qwen/Qwen2.5-0.5B-Instruct`. The task stays AG News, posed as generation: the prompt asks for the topic and the completion is the label name. Evaluation picks the label with the highest log-likelihood, so metrics, gate, calibration and bootstrap CI keep their meaning.
- **BREAKING** `params.yaml`: new `task:` and `finetune:` blocks. The encoder-only `tuning` knobs (freeze layers, differential LR, LoRA flags) are removed; GLiNER config sections and fields are dropped.
- Fine-tune methods `full | lora | qlora | qat`, selected in `params.yaml` and compared as `tune` variants. They are never a perf-sweep dimension.
- PTQ schemes (`fp16, bf16, int8-wo, int8-dyn, int4-wo, fp8, nf4, bnb-int8`) as a shared perf module. Each scheme declares the compute capability it needs and is filtered against the role's TARGET GPU, extending the hardware-binding invariant to quantization.
- QAT for a target scheme that the infer target must support; the saved model records its scheme so evaluation and inference run it quantized.
- The trainer applies the box's tuned perf config when `perf.apply_tuned` is on (opt-in), closing one "Not wired yet" hook.
- A real `--entry` for the perf platform (`datascience.infer.perf_entry`) backed by the pipeline's own predictor and trainer.
- `datascience perf report`: a quant-vs-quality table from the stored infer sweep and the tune ranking.
- New optional `quant` dependency extra.

## Capabilities

### New Capabilities
- `llm-finetune`: causal-LM SFT pipeline for a classification task posed as generation — config loading, prompt/label-masked training data, fine-tune methods, label-scoring evaluation and prediction, and the opt-in tuned-perf hook.
- `quant-schemes`: named post-training quantization schemes, their hardware requirements, filtering against role targets, and their use in the perf infer sweep through a real entry.
- `qat`: quantization-aware fine-tuning bound to a scheme the infer target supports, producing a model that is evaluated and served as quantized.
- `quant-report`: a quant-vs-quality report over stored perf results and tune runs.

### Modified Capabilities
(none — `openspec/specs/` is empty)

## Impact

- Code: `config/{params,experiments,perf}.yaml`, `config/configuration.py`, `entity/config_entity.py`, `model.py`, `components/{data_transformation,model_trainer,model_evaluation}.py`, `infer/predict.py`, `tune/dapt.py`, `perf/{hw,cli}.py`, `perf/workloads/infer.py`; new `perf/quant.py`, `perf/report.py`, `infer/scoring.py`, `infer/perf_entry.py`.
- Dependencies: new `quant` extra (`bitsandbytes`, `torchao`), GPU box only. Base deps and `requirements.txt` stay torch-free.
- Out of scope: serving and packaging (`serve/local.py`, `package.py`, `optimize.py`) stay GLiNER-shaped and are listed as known-broken; GPTQ/AWQ.
