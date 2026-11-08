
# MLOps repo

Rules, invariants and the code map for this repo. When a convention changes, edit this file.

## What this repo is

A multi-model MLOps platform. Generic code is one package (`src/datascience`, imported as
`datascience`); each model is a **pack** in `packs/<pack>/`.

1. **Platform** (`src/datascience`): per-model paths, the stage runner, gate, registry (local,
   MLflow with an approval alias, S3, staging/production pointers), the tuning loop, package
   writers, release, SageMaker jobs and pipelines, CI deploy, and the FastAPI serving shell
   (`ops/serve`).
2. **Packs** (`packs/`): one per model. The shipped example is `packs/text_classification`, a
   causal-LM SFT (Qwen2.5-0.5B-Instruct) with AG News posed as generation: the label is the
   completion, and prediction picks the label with the highest log-likelihood (`scoring.py`,
   shared by train, eval, predict, serving and the perf entry). A pack's trainer and evaluator are
   **reference code you copy and edit**. They are not a config engine.
3. **GPU perf platform** (`perf/`): tuning for training throughput, inference, profiling and kernel
   autotuning. It is task-agnostic and **bound to hardware** (see "Perf platform").

## Model packs

`--model <id>` (or env `DS_MODEL`, default `text_classification`) selects `config/<id>/` and
`artifacts/<id>/`. `params.yaml` may set `pack: <pack>` so several model ids share one pack.

A pack is a plain module `packs/<pack>/pack.py`, loaded by `datascience/pack.py`:

| Name | Required | What the platform does with it |
|---|---|---|
| `STAGES = [(name, fn(cm, p, **kw))]` | yes | `run` / stage commands / `tune`; must contain `evaluate`, which writes `p.METRICS_FILE`. The runner gates and registers after `evaluate`. |
| `METRICS` | yes | valid names for `metrics.compute`, `objective` and gate rules |
| `validate_config(params)` | no | called by `ConfigurationManager` after the common checks |
| `lineage(cm)` | no | extra registry tags |
| `load_predictor(model_dir)` | no | `infer` |
| `targets(cm, p, version) -> [Target]` | no | `package`. `Target.kind` picks the writer: `triton-python` (model-dir tar, optional bundled deps, Python 3.10 build) or `pytorch-dlc` (flat, deterministic tar). |
| `smoke(target_dir, target, cm)` | no | run on every built target; raise on a miss |
| `COMMANDS = {name: (help, fn)}` | no | extra one-off subcommands (`sanity`, `dapt`) |
| `serving_app.py` (`Request`, `WARMUP`, `load`, `respond`) | no | the FastAPI shell (`ops/serve`, `--build-arg PACK=<pack>`) |

Add a pack: `packs/<pack>/{__init__,pack}.py`, `config/<id>/params.yaml` (sections `model`,
`metrics`, `gate`, `tracking`, `registry` plus the pack's own), `serving.yaml`
(`sagemaker.models`, `release.build_store`), optionally `sagemaker.yaml`, `experiments.yaml` and
`deploy/<id>/<env>.yaml`. Then `python main.py --model <id> run`. `tests/toy/pack.py` is the
smallest working pack.

## Lifecycle (what runs where)

| Step | Where | Command / workflow |
|---|---|---|
| train, evaluate, gate, register | laptop or GPU box | `python main.py --model <id> run` |
| the same on AWS | SageMaker Training Job | `train-remote` (one job, every stage) |
| the same as a pipeline | SageMaker Pipeline, owned by CI | `.github/workflows/pipeline.yml` -> `pipeline upsert` + `start` |
| approve | a human | point the MLflow alias (`registry.remote.mlflow.alias`, `champion`) at a version |
| release | CI build role | `.github/workflows/release.yml`: `pull` -> `package` (smoke) -> `release` |
| deploy | CI deploy role, per GitHub environment | a PR changing `deploy/<id>/<env>.yaml` -> `.github/workflows/deploy.yml` |
| rollback | a human | revert the env file (the old key is still in the prefix), or move the alias back |

Nothing deploys from a laptop. Endpoint creation (the long-lived MME, its IAM and autoscaling) is
infra-as-code outside this repo. A SageMaker MME caches a model **by S3 key**, so every tar is
`<target>-<version>.tar.gz`, a released version is immutable (`release` refuses to overwrite), and
`deploy` never overwrites a key in the endpoint prefix.

## Where things live

- `constants/paths.py`: `ModelPaths` (every per-model artifact and config path), shared
  `HARDWARE_FILE`, `PERF_FILE`, `PERF_DIR`, `MLFLOW_URI`. Module attribute access (`paths.X`)
  resolves against the active model. Paths are never hardcoded anywhere else.
- `config/`: `hardware.yaml` and `perf.yaml` (shared), and per model `config/<id>/`:
  `params.yaml` (the experiment), `experiments.yaml` (tuning variants), `serving.yaml` (targets,
  build store), `sagemaker.yaml` (instances, prefix; account values come from env).
- In `src/datascience/`: `pack.py` (loader + `Target`), `pipeline/stages.py` (runner + gate/register), `cli.py`
  (pre-parses `--model`, builds stage commands from the pack), `config/configuration.py` (common
  sections only).
- `registry/`: `base.register` (local always, remote through `_REMOTE_BACKENDS`: `mlflow`, `s3`;
  records `train_hw`), `mlflow_registry.py` (push with alias, pull), `s3.py`, `stages.py`.
- `tune/`: `variants.py` (the one place a variant name resolves), `loop.py`, `promote.py`.
- `package.py` (writers), `serve/release.py` (build store + manifest), `serve/deploy.py` (CI plan
  and sync), `serve/triton.py` (local Triton compose).
- `cloud/`: `steps.py` (`sm-step`, inside a job), `train_job.py`, `pipeline.py`, `config.py`.
- `perf/`: `hw.py`, `store.py`, `runner.py`, `workloads/{infer,train,kernel}.py`, `quant.py`,
  `report.py`, `profile.py`, `kernels/`, `cli.py`.
- `packs/text_classification/`: `data_*.py`, `train.py`, `evaluate.py`, `model.py`, `metrics.py`,
  `scoring.py`, `predict.py`, `serving.py` + `serving_app.py` + `dlc/inference.py`,
  `perf_entry.py`, `sanity.py`, `dapt.py`, `config.py`, `pack.py`.
- `ops/serve/` (FastAPI shell + image), `ops/train/` (SageMaker job image), `deploy/` (desired
  versions per env), `.github/workflows/`.
- `tests/`: offline tests. `tmp_artifacts` redirects artifact paths; `toy_model` roots the toy
  pack in a tmp tree with a tmp MLflow. No GPU, dataset or AWS.

## Working rules

- **Lean.** Add a guard only where a real failure mode exists. Serving, ops and perf code stays happy-path and config-driven. The trainer's robustness (auto-resume, OOM retry, NaN guard, atomic writes) is the bar to match.
- **Platform vs pack.** The platform keeps only the common baseline; a pack owns its data, training, metrics, contract and serving code and may change them freely. Pure, stable, task-agnostic logic lives once in `utils/`, `registry/` or `perf/`.
- **Opt-in.** New capabilities are off by default (DAPT, LoRA/QLoRA/QAT, TensorBoard histograms, the `trt` and `quant` extras, and `perf.apply_tuned`).
- **Minimal deps.** The pyproject base and `requirements.txt` (the serving runtime) stay torch-free; keep the two in sync. Training, dev and perf deps go in the `dev` extra, TensorRT in `trt`, bitsandbytes and torchao in `quant`. The `cpu` and `gpu` extras are mutually exclusive ORT wheels. SageMaker calls use boto3 only. The only pre-commit hook is detect-secrets.
- **No account values in git.** Image, role, bucket, region, build store and MLflow come from env (CI variables); YAML keeps them `null`.
- **GPU, Triton and SageMaker code** is written clean for the GPU box / AWS, with no laptop fallbacks.

## Invariants

- **Fine-tune, don't train from scratch.** To change architecture, swap `model.base_model`.
- **No leakage.** `val` selects, `test` is touched once, and label names come from TRAIN only.
- **Objective reports, loss selects.** The trainer keeps the best checkpoint on `eval_loss`. `metrics.objective` is what evaluation reports and what the gate and `tune` ranking read by NAME, so they stay task-agnostic.
- **Storage always; promotion gated.** A model is always saved to disk. Only a model that passes the gate enters the registry. The MLflow alias moves on registration and by hand; stage pointers move only through `promote-stage`, and `rollback` restores the previous holder.
- **Lineage.** Both register sites (`pipeline/stages.py`, `tune/promote.py`) pass metrics, `data_version` and the training `run_id`. The tune path also passes a config snapshot. `registry.base.register` adds `train_hw` centrally.
- **Serving reads the model dir only.** A pack's serving code must not need `params.yaml` or the training package; the example pack writes `serving_meta.json` beside the model in `evaluate`.

## Fine-tune methods vs quantization schemes

- **Methods** (`full|lora|qlora|qat`, `params.yaml finetune`) are quality knobs, compared as `tune` variants in `experiments.yaml`.
- **PTQ schemes** are infer precisions in the `perf tune infer` sweep, under the quality guard. See `perf/quant.py` `SCHEMES` for the table. The `perf tune train` sweep stays speed-only.
- **Filtered by the role TARGET** capability (`hardware.yaml`), never by this box.
- **QAT** scheme must be supported by `targets.infer`. QAT saves float weights plus `quant_scheme.json`; `model.load_trained` applies the recorded scheme, and that is the convert step. The serving path (`serving.py`) loads plain weights only.

## Perf platform

- **Hardware binding** is the core invariant:
  - Each result is stored at `artifacts/perf/<role>/<hw_id>/<workload>.json` (shared by every model).
  - `hw_id` is built from GPU model, compute capability, VRAM and count. Software versions are recorded but not part of the key.
  - `load_tuned` refuses a result that was tuned on another GPU, so re-tune on the target box.
  - `sweep` and `perf tune` call `assert_role` against `hardware.yaml targets.<role>`, so the infer config is tuned on the infer GPU.
- **Capability comes from the target, not the box.** `supports("bf16", "infer")` reads `targets.infer`; a T4 never gets bf16 even when the sweep runs on an A100.
- **Effective batch is constant** in `perf tune train`: per-device batch × grad_accum × GPUs = `effective_batch`. Tuning changes speed only.
- **Train objective is whole-run throughput** (`samples_per_s_e2e`): the startup measured before the timed steps (torch.compile took ~5 min on a V100) is paid once over `amortize_steps` in `perf.yaml`. Set it near the real run's optimizer steps, or compile wins sweeps it loses in practice.
- **Apply tuned:** `perf.apply_tuned` (off by default) makes the trainer merge `load_tuned("train", "train")`; a result from another GPU stops training.
- **Quality guard.** Infer candidates must satisfy `quality_drop ≤` the constraint, measured against the first backend at fp32. Kernel configs must match the torch reference within `max_abs_err`.
- **Inputs are traffic, not knobs.** Infer measures every length bucket and scores the worst one. Kernels run one sweep per shape.
- **Entry:** `--entry module:fn` supplies the model or trainer, e.g. `packs.text_classification.perf_entry:target`. See the workload docstrings for the contract. A new pack plugs in here without edits to `perf/`.
- **Not wired yet:** tuned precision / batch into a pack's targets, and an `infer_hw` check at serve time.

## Verification (narrow → wide)

1. `make test`: offline, no GPU (also run by `ci.yml`).
2. `python main.py pipeline definition`: the SageMaker pipeline JSON, with no AWS call.
3. `datascience perf fingerprint`: shows whether this box matches each target.
4. On the GPU box: `datascience perf tune infer --entry packs.text_classification.perf_entry:target`, `perf tune train --entry packs.text_classification.perf_entry:train_target`, `perf profile …`, `perf tune kernel layernorm`.
5. `datascience perf report`: quant-vs-quality table from the stored infer sweep (writes `report.md` / `report.csv` beside it).

## Gotchas

- transformers>=5 API: the trainer passes `warmup_steps` as a ratio and `dtype=`, not the 4.x names.
- The `quant` extra is GPU-only. bitsandbytes and torchao import lazily; when missing, the scheme raises `Unsupported` and the sweep records it infeasible.
- The tuning loop clears `CKPT_DIR` between variants. Otherwise auto-resume loads a mismatched checkpoint.
- A failing tune variant is recorded with its `error` in `results.json` and ranked last; the loop exits non-zero after ranking the rest, and `promote` refuses it.
- `--variant` applies to stage commands only: `promote` takes a positional also called `variant`.
- `CUDA_VISIBLE_DEVICES` defines the box: the fingerprint counts only visible GPUs, so pin one GPU on a shared machine before `perf` commands.
- MLflow 3 returns model versions as ints; code that builds paths from them must `str()` them.
- `triton-python` targets with bundled deps must be packaged on Python 3.10 (the Triton DLC ABI); `pytorch-dlc` targets have no such constraint.
- The SageMaker SDK v3 dropped the v2 Estimator / workflow API; `cloud/` builds the API requests itself, validated in tests against botocore's SageMaker model.

## Deferred (not built)

- Monitoring and retrain triggers.
- An approval-event trigger for `release` (an MLflow webhook or registry event); today a human runs the workflow.
- Endpoint IaC (lives with infra).
