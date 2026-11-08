# Usage

Reference for running the pipeline and the perf platform. Overview: [README](../README.md). Rules and code map: [agents/pipeline.md](agents/pipeline.md).

## Pipeline

Every model is a pack (`packs/<pack>/`) selected with `--model <id>` (default `text_classification`).
The example pack's stages:

```
ingest → validate (gate) → transform (stratified split + label map) → train (HF Trainer, causal-LM SFT)
       → evaluate (metrics + calibration + CI + error analysis) → gate → registry (local + MLflow @champion)

tuning loop:  experiments.yaml (variants) → tune (every stage per variant, 1 MLflow run each) → promote <winner>
on AWS:       train-remote (one Training Job)  or  pipeline (Train → Evaluate → Gate → Register), owned by CI
release:      CI: pull @champion → package (versioned tars + smoke) → release (build store + manifest.json)
deploy:       CI: PR on deploy/<model>/<env>.yaml → copy tar into the endpoint prefix → smoke invoke
```

## Run it

```bash
pip install -e .[dev,cpu]                   # dev / training stack (+ [trt], [quant] on a GPU box)

python main.py run                            # full pipeline (downloads a sample dataset on first run)
python main.py infer --text "Central bank raises interest rates"
```

Use your own data: set `ingestion.dataset_name: null` in `config/text_classification/params.yaml` and drop a
`{"text": ..., "label": ...}` JSONL at `artifacts/text_classification/data_ingestion/data.jsonl`. Label names and metrics are derived from your data.

A new model: add a pack (see [agents/pipeline.md](agents/pipeline.md#model-packs)) and run `python main.py --model <id> run`.

## Commands

| Command | What it does |
|---|---|
| `run` | full pipeline of the active pack (slice with `--only` / `--from` / `--to`, `--variant <name>`) |
| `ingest` · `validate` · `transform` · `train` · `eval` | a single stage (whatever the pack's `STAGES` are) |
| `sanity` · `dapt` | example-pack commands: overfit-one-batch pre-flight · domain-adaptive pretraining (GPU box) |
| `variants` · `tune` · `promote <variant>` | list · run+rank · version the chosen variant (with lineage) |
| `infer` | predict `--text` or batch over `--input-file` (JSONL, resumable) |
| `pull [--version <alias or N>]` | download a registered MLflow model into the model dir |
| `package --version <v>` | the pack's targets → `<target>-<v>.tar.gz` + smoke |
| `release --version <v>` | tars + `manifest.json` → the S3 build store (immutable) |
| `serve` · `serve-triton` | the FastAPI shell on the trained model · the packaged Triton repo (docker compose) |
| `train-remote [--wait]` | every stage as one SageMaker Training Job |
| `pipeline definition\|upsert\|start` | the SageMaker Pipeline (CI runs upsert/start) |
| `sm-step all\|train\|evaluate\|register` | the entrypoint inside a SageMaker job |
| `promote-stage` · `rollback` · `stage-path` | move / restore / resolve staging & production pointers |
| `perf fingerprint` | this box's `hw_id` + whether it matches the train / infer target |
| `perf tune infer --entry m:f` | backend × precision × batch sweep, quality-guarded, worst length bucket |
| `perf tune train --entry m:f` | precision / TF32 / compile / checkpointing / workers, auto max-fit batch, effective batch constant |
| `perf tune kernel <name>` | OpenAI-Triton launch-config autotune per shape, checked vs torch |
| `perf profile train\|infer --entry m:f` | torch.profiler trace + top kernels + NVML util |
| `perf tune infer --entry packs.text_classification.perf_entry:target` | the real model: PTQ schemes (`perf/quant.py`) x batch, quality-guarded |
| `perf report` | quant-vs-quality table (`report.md` / `.csv`) from the stored infer sweep |
| `perf show` | stored tuned results by role / `hw_id` |

`--gpu N` on any stage sets `CUDA_VISIBLE_DEVICES`. `dvc repro` runs the same DAG with caching.

## Architecture

The platform (`src/datascience`) owns paths, the runner, gate, registry, tuning, packaging,
release, SageMaker jobs/pipelines, CI deploy and the serving shell; each pack owns its model.
**This repo = all core AI** (data → train → tune → eval → package → release). Endpoints are infra
(IaC); CI ships versions into them. A business app only **calls** the endpoint (for the example
pack: text in → `{label, score}`).

## CI and AWS setup

Workflows read repository variables (never committed): `AWS_REGION`, `BUILD_ROLE_ARN` (OIDC role:
ECR push, SageMaker pipeline upsert/start, build-store write), `TRAIN_IMAGE_REPOSITORY` (ECR URI),
`SAGEMAKER_ROLE_ARN` + `SAGEMAKER_BUCKET` (the jobs' execution role and bucket),
`RELEASE_BUILD_STORE` (`s3://bucket/prefix/<model>`), `MLFLOW_TRACKING_URI`, `MLFLOW_SECRET_ARN`
(Secrets Manager `{username, password}` the jobs read); secrets `MLFLOW_TRACKING_USERNAME` /
`MLFLOW_TRACKING_PASSWORD` (release pulls the model). Each GitHub environment (`staging`,
`production`) sets `DEPLOY_ROLE_ARN` (copy into its endpoint prefix + invoke); give `production`
required reviewers. Fill `deploy/<model>/<env>.yaml` (`endpoint`, then a version per target) in a
PR to deploy.

## Install profiles

| Use | Install |
|---|---|
| Dev / training / perf | `pip install -e .[dev,cpu]` (`[trt]` adds TensorRT, `[quant]` adds bitsandbytes + torchao; GPU box) |
| Serving (FastAPI) | the image: `docker build -f ops/serve/Dockerfile --build-arg PACK=<pack> .` (torch CPU + the pack's `serving-requirements.txt`) |
| SageMaker jobs | the image: `docker build --platform linux/amd64 -f ops/train/Dockerfile .` (built and pushed by `pipeline.yml`) |

## Config (`config/`)

Per model, in `config/<id>/`:
- `params.yaml` — the base experiment: model, metrics, gate, tracking, registry, plus the pack's own sections.
- `experiments.yaml` — the tuning loop: named variants (sparse overrides on `params.yaml`).
- `serving.yaml` — the targets (`sagemaker.models`) and the release build store.
- `sagemaker.yaml` — instance types and the S3 prefix; image / role / bucket come from env.

Shared:
- `hardware.yaml` — the train and infer GPU targets. Tuned results bind to these.
- `perf.yaml` — perf sweep grids, objectives, constraints.

Artifact paths + the MLflow URI are fixed in `src/datascience/constants/paths.py` (not config).
Tracking and the registry are local MLflow unless `MLFLOW_TRACKING_URI` is set: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## GPU smoke

The runbook that verified the pipeline and perf platform on a 1x Tesla V100 (sm_70) on 2026-10-12. Copy it and adjust the marked lines. Run it on a CUDA box, never on the Mac.


```bash
git clone https://github.com/buidangphuc/MLOps.git && cd MLOps
uv venv -p 3.11 .venv && . .venv/bin/activate
uv pip install -e '.[dev,gpu,quant]'
# V100 / sm_70 only: default torch wheels (cu128+) dropped Volta -> reinstall the cu126 build AFTER the extras
uv pip install --reinstall-package torch "torch==2.14.1+cu126" \
  --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match
# no system C compiler? torch eager compiles triton kernels -> local gcc
conda create -y -p $PWD/.cc -c conda-forge --override-channels gcc gxx
export CC=$PWD/.cc/bin/gcc CXX=$PWD/.cc/bin/g++
export CUDA_VISIBLE_DEVICES=7          # ADJUST: one free GPU (nvidia-smi)
```

```bash
datascience perf fingerprint           # copy gpu name / capability / VRAM into the next step
python - <<'EOF'
import yaml
p = "config/hardware.yaml"; h = yaml.safe_load(open(p))
for role in ("train", "infer"):        # ADJUST to the fingerprint (perf refuses a mismatched box)
    h["targets"][role].update(gpu_name="V100", compute_capability="7.0", gpu_count=1, min_vram_gb=30)
yaml.safe_dump(h, open(p, "w"), sort_keys=False)
s = open("config/text_classification/params.yaml").read()
s = s.replace("sample_size: 3000", "sample_size: 2000").replace("max_steps: -1", "max_steps: 60")
open("config/text_classification/params.yaml", "w").write(s.replace("on_fail: error", "on_fail: warn"))
EOF
make test
```

```bash
datascience run                                                        # ~3 min
datascience tune                                                       # ~6 min, 4 variants
datascience perf tune infer --entry packs.text_classification.perf_entry:target       # ~18 min
# perf.yaml train.amortize_steps: set to the optimizer steps of the run you'll launch
datascience perf tune train --entry packs.text_classification.perf_entry:train_target # ~45 min, full grid
datascience perf report               # -> artifacts/perf/infer/<hw_id>/report.md
datascience infer --text "Stocks rallied after the earnings beat."
```

Don't commit the `hardware.yaml` / `params.yaml` edits — they describe the smoke box, not the real targets.
