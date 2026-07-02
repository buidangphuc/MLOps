# Agent instructions

Architecture knowledge for this repo lives in the repo-local skill:

- `.agents/ml-pipeline-repo/SKILL.md` — where code lives, working rules,
  the add-a-pipeline recipe, verification ladder, gotchas.

Read it before modifying anything. Keep it the single source: when a
convention changes, edit the skill — don't duplicate rules into this file
or `CLAUDE.md`.

Quick facts: install `pip install -e .[dev,cpu]` · test `make test` ·
run a stage `datascience run --only <stage>` · all commands
`datascience --help`.
