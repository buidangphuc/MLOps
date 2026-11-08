# Project facts for agents

## Feedback loops
<!-- read by: diagnosing-bugs, tdd -->
- **Unit**: `make test` (one test: `make test-one k=<expr>`) — offline, no GPU, dataset or AWS, ~20 s. `tests/toy/pack.py` + the `toy_model` fixture drive the platform end to end in a tmp tree with a tmp MLflow.
- **Pipeline definition**: `python main.py pipeline definition` — the SageMaker pipeline JSON, no AWS call.
- **Whole flow on real weights**: [docs/usage.md](../usage.md#rehearse-the-whole-flow) — run → pull → package (smoke) → serve → release → deploy → SageMaker job/pipeline, AWS emulated by moto. One V100: `run` ~3 min (macro_f1 ≈ 0.905), `package` ~2 min. Needs a CUDA box or a small `--variant` on a laptop (skill `gpu-box-run`).

## Checks
<!-- read by: pre-commit-ci. Mirror CI exactly. -->
- Source: `.github/workflows/ci.yml` (last read: 2026-11-08). GitHub Actions on this account was blocked by billing on 2026-11-08; run the checks locally.
- `pre-commit run --all-files` — security (detect-secrets, the only hook)
- `make test` — test
- Workflows changed: `docker run --rm -v "$PWD":/repo -w /repo rhysd/actionlint:latest` — lint

## API collection
<!-- read by: diagnosing-bugs, everyone calling the project's HTTP APIs -->
- N/A. The serving shell's API is `POST /api/v1/predict` (see `ops/serve/app/router.py`); call it with curl after `python main.py serve`.

## Tracker
<!-- read by: to-tickets -->
- System: GitHub `buidangphuc/MLOps` (issues and pull requests).

## Docs
<!-- read by: domain-modeling, research -->
- Agent guide: `AGENTS.md` | rules and code map: `docs/agents/pipeline.md` | usage: `docs/usage.md` | glossary, ADRs: none yet.

## Hot paths
<!-- read by: everyone. Things that look alike but must not be confused. -->
- Paths are per model (`config/<id>/`, `artifacts/<id>/`); `config/hardware.yaml`, `config/perf.yaml` and `artifacts/perf/` are shared by every model.
- `src/datascience/` is platform code; `packs/<pack>/` is model code. `tests/toy/pack.py` is a test pack, not a shipped one.
- `deploy/<model>/<env>.yaml` is deployed state (CI acts on a PR that changes it); `config/<id>/serving.yaml` is build config.
- `requirements.txt` is the torch-free runtime bundled into `triton-python` targets (keep it in sync with `pyproject.toml`); the FastAPI image installs `ops/serve/requirements.txt` + `packs/<pack>/serving-requirements.txt` instead.
- Tests use the `tmp_artifacts` / `toy_model` fixtures, never the real `artifacts/`.
- Fine-tune methods (`finetune.method`, compared as `tune` variants) are not perf knobs (PTQ schemes in `perf tune infer`).
- `quant_scheme.json` (QAT, read by `load_trained`) vs `train_stats.json` (trainer speed/memory, read by `perf report`).
