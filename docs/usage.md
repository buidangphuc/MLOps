# Usage

Reference for running the pipeline and the perf platform. Overview: [README](../README.md). Rules and code map: [agents/pipeline.md](agents/pipeline.md).

## Pipeline

```
ingest → validate (gate) → transform (stratified split + label map) → train (HF Trainer, causal-LM SFT)
       → evaluate (metrics + calibration + CI + error analysis) → gate → local registry

tuning loop:  experiments.yaml (variants) → tune (train+eval+rank, 1 MLflow run each) → promote <winner>
serving:      torch (default)   or   optimize → package → serve / deploy  (Triton + SageMaker)
```

## Run it

```bash
pip install -e .[dev,cpu]                   # dev / training stack (+ [trt], [quant] on a GPU box)

python main.py run                            # full pipeline (downloads a sample dataset on first run)
python main.py infer --text "Central bank raises interest rates"
```

Use your own data: set `ingestion.dataset_name: null` in `config/params.yaml` and drop a
`{"text": ..., "label": ...}` JSONL at `artifacts/data_ingestion/data.jsonl`. Label names and metrics are derived from your data.

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
| `analyze` | sweep Triton serving configs on real traffic (Model Analyzer) |
| `promote-stage` · `rollback` · `stage-path` | move / restore / resolve staging & production pointers |
| `perf fingerprint` | this box's `hw_id` + whether it matches the train / infer target |
| `perf tune infer --entry m:f` | backend × precision × batch sweep, quality-guarded, worst length bucket |
| `perf tune train --entry m:f` | precision / TF32 / compile / checkpointing / workers, auto max-fit batch, effective batch constant |
| `perf tune kernel <name>` | OpenAI-Triton launch-config autotune per shape, checked vs torch |
| `perf profile train\|infer --entry m:f` | torch.profiler trace + top kernels + NVML util |
| `perf tune infer --entry datascience.infer.perf_entry:target` | the real model: PTQ schemes (`perf/quant.py`) x batch, quality-guarded |
| `perf report` | quant-vs-quality table (`report.md` / `.csv`) from the stored infer sweep |
| `perf show` | stored tuned results by role / `hw_id` |

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
| Dev / training / perf | `pip install -e .[dev,cpu]` (`[trt]` adds TensorRT, `[quant]` adds bitsandbytes + torchao; GPU box) |
| Serving — torch (default) | the model stack (torch + transformers; = the dev install) |
| Serving — ONNX (opt-in, torch-free) | `pip install -e .[cpu]` **or** `.[gpu]` |

## Config (`config/`)

- `params.yaml` — the base experiment: data, model, training, metrics, gate, tracking, registry (+ an off-by-default `tuning:` block).
- `experiments.yaml` — the tuning loop: named variants (sparse overrides on `params.yaml`).
- `serving.yaml` — serving lifecycle: runtime, optimize, triton, sagemaker, analyze.
- `hardware.yaml` — the train and infer GPU targets (+ SageMaker instance types). Tuned results bind to these.
- `perf.yaml` — perf sweep grids, objectives, constraints.

Artifact paths + the MLflow URI are fixed in `src/datascience/constants/paths.py` (not config).
Tracking is local MLflow: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

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
s = open("config/params.yaml").read()
s = s.replace("sample_size: 3000", "sample_size: 2000").replace("max_steps: -1", "max_steps: 60")
open("config/params.yaml", "w").write(s.replace("on_fail: error", "on_fail: warn"))
EOF
make test
```

```bash
datascience run                                                        # ~3 min
datascience tune                                                       # ~6 min, 4 variants
datascience perf tune infer --entry datascience.infer.perf_entry:target       # ~18 min
# perf.yaml train.amortize_steps: set to the optimizer steps of the run you'll launch
datascience perf tune train --entry datascience.infer.perf_entry:train_target # ~45 min, full grid
datascience perf report               # -> artifacts/perf/infer/<hw_id>/report.md
datascience infer --text "Stocks rallied after the earnings beat."
```

Don't commit the `hardware.yaml` / `params.yaml` edits — they describe the smoke box, not the real targets.
