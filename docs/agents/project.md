# Project facts for agents

## Feedback loops
<!-- read by: diagnosing-bugs, tdd -->
- Unit: `make test` (offline, no GPU, no downloads). One test: `make test-one k=<expr>`.
- Real-model and GPU smoke never run on the Mac (not enough RAM). Run them on a CUDA box with [docs/usage.md#gpu-smoke](../usage.md#gpu-smoke).

## Checks
<!-- read by: pre-commit-ci. No CI workflow exists; run these locally. -->
- `pre-commit run --all-files` (the only hook is detect-secrets)
- `make test`

## Tracker
<!-- read by: to-tickets -->
- GitHub `buidangphuc/MLOps`.
- Changes are planned as OpenSpec changes under `openspec/changes/` (archived to `openspec/changes/archive/`, specs in `openspec/specs/`).

## Docs
<!-- read by: domain-modeling, research -->
- Rules, invariants, code map, gotchas: `docs/agents/pipeline.md` (read before touching code, configs, serving, registry or perf)
- Usage (commands, install profiles, config files, GPU smoke runbook): `docs/usage.md`
- Specs and change proposals: `openspec/`

## Hot paths
<!-- read by: everyone. Things that look alike but must not be confused. -->
- Artifact paths and `MLFLOW_URI` live only in `src/datascience/constants/paths.py`; never hardcode them.
- `requirements.txt` is the torch-free serving runtime that `package.py` bundles; keep it in sync with `pyproject.toml`.
- Tests use the `tmp_artifacts` fixture in `tests/conftest.py`, never the real `artifacts/`.
- Fine-tune methods (`finetune.method`, compared as `tune` variants) are not perf knobs (PTQ schemes in `perf tune infer`).
- `quant_scheme.json` (QAT, read by `load_trained`) vs `train_stats.json` (trainer speed/memory, read by `perf report`).
