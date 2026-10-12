## Why

The repo root mixes product files with leftovers, and the agent rules sit in agent-specific files (`CLAUDE.md` routing to `.agents/ml-pipeline-repo/SKILL.md`). The owner keeps agent behaviour in user-level instructions, which say project facts live in `docs/agents/project.md`. Mirrors ai-platform-template's `repo-tidy` (archived there as `2026-10-12-repo-tidy`).

Findings:
- `app.py` is dead: a FastAPI page that mounts `templates/` and `static/`, neither of which exists.
- `requirements-cpu.txt`, `requirements-gpu.txt`, `requirements-dev.txt` duplicate the `cpu` / `gpu` / `dev` extras in `pyproject.toml`; nothing reads them.
- `scripts/format.sh` runs isort, pyink and ruff, none of which is a dependency (the repo has no lint/format tooling; detect-secrets is the only hook).
- `research/*.ipynb` are April 2025 notebooks for the old pipeline; nothing references them.
- `.claude/` holds the files `openspec init --tools claude` generated (`commands/opsx/*`, `skills/openspec-*`).
- `artifacts/data_validation/status.txt` and `artifacts/model_evaluation/metrics.json` are tracked although `.gitignore` ignores `artifacts/` (pipeline output).
- `infra/` is the ops folder; the template calls it `ops/`.
- `README.md` mixes the overview with usage reference (commands table, install profiles, config files).

Kept on purpose: `requirements.txt` (`src/datascience/package.py:159` bundles it into the Triton deps and `pyproject.toml` keeps it in sync), `.env.test.example` and `.env.staging.example` (`config/serving.yaml:22` points to them), `dvc.yaml`, `.dvc/`, `.dvcignore`.

## What Changes

- Delete `app.py`, `requirements-{cpu,gpu,dev}.txt`, `scripts/format.sh`, `research/`, `.claude/`; add `.claude/` to `.gitignore`; untrack the two `artifacts/` files.
- `git mv infra ops`, fixing the paths that name `infra/` inside it.
- `.agents/ml-pipeline-repo/SKILL.md` becomes the plain doc `docs/agents/pipeline.md`; new `docs/agents/project.md` (Feedback loops, Checks, Tracker, Docs, Hot paths) indexes it. `CLAUDE.md` and `.agents/` are deleted.
- `README.md` becomes an overview (what, scope, layout, quickstart, status, links); the commands table, install profiles, config reference and the GPU smoke runbook from PR #4 move to `docs/usage.md`.
- `openspec/config.yaml` points to `docs/agents/`.

No runtime behaviour, CLI command, config key or test changes.

## Impact

Paths change for the ops files (`ops/` instead of `infra/`) and for the agent docs. `make` targets, `datascience` commands and `pip install -e .[...]` extras stay the same.
