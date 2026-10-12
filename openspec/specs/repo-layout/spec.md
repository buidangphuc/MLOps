# repo-layout Specification

## Purpose
TBD - created by archiving change repo-tidy. Update Purpose after archive.

## Requirements

### Requirement: Small repository root
The repository root SHALL hold only product files: `README.md`, `Makefile`, `pyproject.toml`, `main.py`, `requirements.txt`, `dvc.yaml`, the `.env*.example` templates, standard dotfiles (`.gitignore`, `.dvcignore`, `.pre-commit-config.yaml`, `.secrets.baseline`, `.dvc/`) and the directories `src/`, `tests/`, `config/`, `docs/`, `ops/`, `openspec/`. Generated pipeline output SHALL NOT be tracked.

#### Scenario: Root listing
- **WHEN** listing tracked files at the repository root
- **THEN** none of `app.py`, `requirements-cpu.txt`, `requirements-gpu.txt`, `requirements-dev.txt`, `CLAUDE.md`, `AGENTS.md`, `.agents/`, `.claude/`, `research/`, `scripts/`, `infra/` exists, and no file under `artifacts/` is tracked

#### Scenario: Behaviour unchanged
- **WHEN** running `make test` and `datascience --help` after the move
- **THEN** both behave as before: same tests pass, same commands listed

### Requirement: Ops files live under ops/
Docker compose stacks, serving images and their env templates SHALL live under `ops/`, and every path written inside them SHALL resolve from their new location.

#### Scenario: Compose paths resolve
- **WHEN** searching the tracked files outside `openspec/changes/` for `infra/`
- **THEN** there is no match, and `ops/serve-cpu/Dockerfile` copies `ops/serve-cpu/requirements.txt`

### Requirement: Agent docs are plain project docs
Repository rules and the code map SHALL live in `docs/agents/` as plain Markdown, indexed by `docs/agents/project.md`. The repository SHALL contain no agent-specific instruction files or tool directories.

#### Scenario: Agent finds the rules
- **WHEN** an agent reads `docs/agents/project.md`
- **THEN** its `## Docs` section points to `docs/agents/pipeline.md`, `docs/usage.md` and `openspec/`, and its `## Checks` section lists detect-secrets and `make test`

#### Scenario: No dangling references
- **WHEN** searching tracked files outside `openspec/changes/` for `.agents/`, `CLAUDE.md` or `SKILL.md`
- **THEN** there is no match

### Requirement: README is an overview
`README.md` SHALL explain what the repo is, its scope, its layout, a minimal quickstart and its status, and link to `docs/usage.md` and `docs/agents/project.md`. Command references, install profiles and runbooks SHALL live in `docs/usage.md`.

#### Scenario: No reference tables in README
- **WHEN** reading `README.md`
- **THEN** it holds no command table or install-profile table; both are in `docs/usage.md`, and every relative link in `README.md`, `docs/usage.md` and `docs/agents/*.md` resolves
