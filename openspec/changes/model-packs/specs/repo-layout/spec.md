## MODIFIED Requirements

### Requirement: Small repository root
The repository root SHALL hold only product files: `README.md`, `Makefile`, `pyproject.toml`, `main.py`, `requirements.txt`, `dvc.yaml`, the `.env*.example` templates, standard dotfiles (`.gitignore`, `.dvcignore`, `.pre-commit-config.yaml`, `.secrets.baseline`, `.dvc/`, `.github/`) and the directories `src/`, `packs/`, `tests/`, `config/`, `deploy/`, `docs/`, `ops/`, `openspec/`. Generated pipeline output SHALL NOT be tracked.

#### Scenario: Root listing
- **WHEN** listing tracked files at the repository root
- **THEN** none of `app.py`, `requirements-cpu.txt`, `requirements-gpu.txt`, `requirements-dev.txt`, `CLAUDE.md`, `AGENTS.md`, `.agents/`, `.claude/`, `research/`, `scripts/`, `infra/` exists, and no file under `artifacts/` is tracked

### Requirement: Ops files live under ops/
Docker compose stacks, serving and job images and their env templates SHALL live under `ops/`, and every path written inside them SHALL resolve from their location.

#### Scenario: Image paths resolve
- **WHEN** reading `ops/serve/Dockerfile` and `ops/train/Dockerfile`
- **THEN** every `COPY` source exists in the repository and is allowed by the image's `Dockerfile.dockerignore`

### Requirement: Agent docs are plain project docs
Repository rules and the code map SHALL live in `docs/agents/` as plain Markdown, indexed by `docs/agents/project.md`. The repository SHALL contain no agent-specific instruction files or tool directories.

#### Scenario: Agent finds the rules
- **WHEN** an agent reads `docs/agents/project.md`
- **THEN** its `## Docs` section points to `docs/agents/pipeline.md`, `docs/usage.md` and `openspec/`, and its `## Checks` section lists detect-secrets, `make test` and actionlint for workflow changes
