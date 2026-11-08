---
name: gpu-box-run
description: Run pipeline steps on a CUDA box. Use when a step needs a GPU: real-model training or evaluation, perf tune or profile, the GPU smoke runbook, or rehearsing the whole flow on real weights.
---

# Run on a GPU box

## 1. Ship the tree

Copy the working tree into a fresh directory of its own, leaving every other checkout on the box
untouched:

```bash
rsync -az --exclude=.git --exclude=.venv --exclude=artifacts --exclude='*.db' --exclude=mlruns ./ <host>:<dir>/
```

Done when `<dir>` holds `main.py`, `src/`, `packs/`, `config/`.

## 2. Pick the interpreter

Reuse a box venv that already has CUDA torch and `transformers>=5`; check with
`<venv>/bin/python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"`.
Run it against the tree with `PYTHONPATH=$PWD/src:$PWD`, so nothing is installed into a shared
venv. With no fitting venv, build one per `docs/usage.md` § GPU smoke (Volta needs the cu126 torch
build and a local gcc for compiled kernels).

Done when that one-liner prints a 5.x transformers and `True`.

## 3. Pin the run

- One free GPU: read `nvidia-smi`, then `export CUDA_VISIBLE_DEVICES=<n>` (perf fingerprints count
  visible GPUs only).
- Tracking: `export MLFLOW_TRACKING_URI=sqlite:///$PWD/<name>.db` unless the box reaches the team
  MLflow.
- Laptop-sized checks first: a small `--variant` (few hundred rows, few steps) before the full config.

## 4. Run detached and watch

Put the steps in a script with `set -euo pipefail` and a banner per step, and stop background
helpers (servers, emulators) with `trap 'kill $PIDS 2>/dev/null' EXIT`, since an early `set -e`
abort skips any later `kill`. Start it so it survives the ssh session:

```bash
ssh <host> 'cd <dir> && setsid nohup ./job.sh > job.log 2>&1 < /dev/null &'
ssh <host> 'tail -n +1 -F <dir>/job.log' | grep -E --line-buffered "########|Traceback|Error|Killed|DONE"
```

Match process names with a bracket (`pkill -f "[m]oto_server"`), so the pattern never matches the
ssh command that carries it.

Done when the log reaches its final banner, or the first failure is read and fixed.

## 5. Leave it clean

Done when `pgrep -fa` shows nothing of yours, the GPU you pinned is back to idle memory, and the
report names the metrics, the timings and the `<dir>` that can be deleted.

Reference, one V100 (2026-11-08): default `run` ~3 min with macro_f1 ≈ 0.905; `package` of the
2 GB model ~2 min; the serving shell ready in ~15 s.
