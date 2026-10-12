# How to work in this repo

The rules and the map for this repo live in `.agents/ml-pipeline-repo/SKILL.md`. Read it before touching anything. It covers where code lives, the working rules, the known-broken training side on `main`, the hardware-binding rules of the perf platform, verification, and gotchas.

That skill is the one place rules live. When a convention changes, edit the skill rather than this file.

## Quick entry points

- Install: `pip install -e .[dev,cpu]`. Add `[trt]` on a GPU box for TensorRT.
- Test: `make test` runs offline without a GPU. One test: `make test-one k=<expr>`.
- Commands: `datascience --help`. Perf commands: `datascience perf --help`.
- Tests get a tmp artifact tree through the `tmp_artifacts` fixture in `tests/conftest.py`, so they never write into the real `artifacts/`.
