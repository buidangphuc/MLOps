## ADDED Requirements

### Requirement: One tree per model
Each model SHALL have its config under `config/<id>/` and its artifacts under `artifacts/<id>/`, selected by `--model <id>` or `DS_MODEL` (default `text_classification`). Hardware and perf files SHALL stay shared.

#### Scenario: Model selection
- **WHEN** running `python main.py --model toy run --to transform` with a `config/toy/` tree
- **THEN** only `artifacts/toy/` is written and the active paths report model `toy`

### Requirement: A pack drives the pipeline
A model SHALL be run through the pack named by `params.yaml` `pack:` (default: the model id), whose `STAGES` define the stage commands and must include `evaluate`. The platform SHALL gate after `evaluate` and register only a passing model, locally and in the configured remote, with lineage (base model, pack lineage, data version, training run id).

#### Scenario: Gate and register
- **WHEN** the toy pack runs end to end with a passing score
- **THEN** a local version and an MLflow version exist, the MLflow alias points at it and its tags carry the lineage

#### Scenario: Failing gate
- **WHEN** the toy pack's score is below the gate rule
- **THEN** nothing is registered

### Requirement: Tuning runs every stage per variant
`tune` SHALL run every pack stage up to `evaluate` for each variant in `experiments.yaml`, snapshot each variant's model, metrics and data version, and `promote <variant>` SHALL register that snapshot with the variant's own config.

#### Scenario: Data-only variants
- **WHEN** two variants differ only in data size
- **THEN** their recorded data versions differ and promoting one registers its data version

### Requirement: Versioned, immutable artifacts
`package --version V` SHALL write one `<target>-V.tar.gz` per pack target, byte-identical across rebuilds for `pytorch-dlc`, and run the pack's smoke on each. `release --version V` SHALL upload the tars and a `manifest.json` (model id, targets with tar and sha256) and SHALL refuse a version that is already released.

#### Scenario: Deterministic tars
- **WHEN** packaging the toy pack twice with the same version
- **THEN** the tar checksums are equal and every member has mtime 0, mode 0644 and owner 0

#### Scenario: No overwrite
- **WHEN** releasing a version whose manifest already exists in the build store
- **THEN** the release fails without uploading

### Requirement: Pack-agnostic serving shell
The FastAPI shell SHALL take its request model, warm-up text, loader and response from `packs/<DS_PACK>/serving_app.py`, and expose `/health`, `/ready`, `/metrics` and `POST /api/v1/predict` with request-id propagation.

#### Scenario: Toy contract
- **WHEN** the shell serves the toy pack and receives `{"text": "hello"}` with an `X-Request-ID`
- **THEN** it answers the toy pack's response and echoes the request id
