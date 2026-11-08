# Agent guide

A multi-model MLOps platform. Generic code lives in `src/datascience/` (imported as
`datascience`); each model is a **pack** in `packs/<pack>/`. The shipped example is
`packs/text_classification` (Qwen2.5-0.5B fine-tuned to classify AG News). `--model <id>` (env
`DS_MODEL`, default `text_classification`) selects `config/<id>/` and `artifacts/<id>/`.

## Orient

- `python main.py --model <id> --help`: every command for that model; its stages come from `packs/<pack>/pack.py`.
- `docs/agents/pipeline.md`: pack contract, lifecycle, code map, invariants, gotchas. Read it before changing code, configs, serving, registry, cloud or perf.
- `docs/agents/project.md`: checks, feedback loops, tracker, hot paths.
- `docs/usage.md`: the human command reference, CI/AWS setup, runbooks.
- Task recipes in `.claude/skills/<name>/SKILL.md`: `add-model-pack`, `gpu-box-run`, `release-model`, `compare-and-gate`.

## Rules

- Model-specific code goes in its pack; `src/datascience/` keeps only what every pack shares.
- Paths come from `datascience.constants.paths`; account values (role, bucket, image, MLflow URI) come from env, and YAML keeps them `null`.
- Deploys run in CI from `deploy/<model>/<env>.yaml`; a released version is immutable, so a change ships as a new version.
- A new capability ships off by default and is switched on by config.
- Done means `make test` is green (offline: no GPU, dataset or AWS) and `pre-commit run --all-files` is clean.
