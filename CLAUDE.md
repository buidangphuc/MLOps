# How to work in this repo

## Architecture: source of truth

The architecture knowledge for this repo lives in the repo-local skill at
`.agents/ml-pipeline-repo/SKILL.md` — where code lives, working rules
(lean-not-defensive, reference-vs-shared-utils line, opt-in addons, fixed
paths, torch-free serving deps), the add-a-pipeline recipe, verification
ladder, and gotchas. **Read it before touching anything.**

Keep the skill as the one place rules live. When a convention changes, edit
the skill — don't copy rules into this file.

## Quick entry points

- Install: `pip install -e .[dev,cpu]` (pyproject extras; the
  requirements*.txt files are legacy pointers).
- Test: `make test` (= `pytest tests/ -q`, offline, no GPU). One test:
  `make test-one k=<expr>`.
- Pipeline: `datascience run [--only <stage>]` (console script; `python
  main.py …` is equivalent). Discover commands: `datascience --help`.
- Tests get a tmp artifact tree via the `tmp_artifacts` fixture in
  `tests/conftest.py` — don't write into the real `artifacts/`.

## Offloading context-heavy work to subagents

Hand bulk work to a subagent (read-only search → `Explore`; multi-step →
`general-purpose`) and keep only the distilled result: large log analysis,
bulk text→schema extraction, mass mock-data generation, repo-wide regex
sweeps. Keep in the main thread: editing code in `src/`, small-file reads,
anything needing the context you've already built.

<!-- SPECKIT START -->
Spec-driven work uses the /speckit-* skills (constitution → specify → plan
→ tasks → implement). When a feature is active, its current plan lives at
`specs/<NNN>-<name>/plan.md` — read it for tech context before coding.
<!-- SPECKIT END -->
