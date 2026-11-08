---
name: add-model-pack
description: Add a model to the platform as a pack (packs/<pack>/ + config/<id>/). Use when adding a new model, porting training code into packs/, or giving an existing pack a second model id.
---

# Add a model pack

The contract a pack must meet (required `STAGES` and `METRICS`, optional hooks) is the table in
`docs/agents/pipeline.md` § Model packs. This skill is the order of work.

## 1. Name it

`<pack>` is the code dir, `<id>` the model id (its `config/<id>/` and `artifacts/<id>/`). They are
equal unless a second model reuses a pack: then `config/<id>/params.yaml` sets `pack: <pack>` and
only step 3 applies.

Done when both names are chosen and `config/<id>/` does not exist yet.

## 2. Write the pack

Copy the smallest working pack, `tests/toy/pack.py` (no ML deps), into `packs/<pack>/pack.py`
with an empty `packs/<pack>/__init__.py`; borrow from `packs/text_classification/` for real
training code.

- A stage is `fn(cm, p)`: read the pack's own sections from `cm.params`, write only under `p`
  (`ModelPaths`), import heavy libraries inside the function.
- `evaluate` writes `p.METRICS_FILE` with names from `METRICS`; the platform gates and registers
  after it.
- `train` writes `p.TRAIN_RUN_FILE` (`{"run_id": ...}`) when it logs to MLflow: lineage and `tune`
  read it.
- Serving reads the model dir only: `evaluate` writes every sidecar serving needs (labels,
  thresholds, prompt) into `p.MODEL_DIR`.
- Pack-only config sections are checked in `validate_config(params)`.

Done when every required name exists and each stage writes only under `p`.

## 3. Write the config

- `config/<id>/params.yaml`: `model.base_model`, `metrics.compute` + `objective` (from `METRICS`),
  `gate` (`on_fail: warn` and `min: 0.0` until a measured baseline exists, see skill
  `compare-and-gate`), `tracking.experiment_name`, `registry.name` + `remote`, plus the pack's own
  sections.
- Shipped models: `serving.yaml` (`sagemaker.models`, `release.build_store: null`) and
  `deploy/<id>/{staging,production}.yaml` (`endpoint: null`, each target `null`) + `smoke.json`.
- SageMaker training: `sagemaker.yaml`, copied from `config/text_classification/`; account values
  stay `null` (they come from env).
- Variants to compare: `experiments.yaml`.

Done when `python main.py --model <id> --help` lists the pack's stages.

## 4. Make it shippable (served models only)

- `targets(cm, p, version)` returns one `Target` per entry of `serving.yaml` `sagemaker.models`.
  `kind: pytorch-dlc` for the SageMaker PyTorch container (an `inference.py` with
  `model_fn`/`input_fn`/`predict_fn`/`output_fn` plus its code under `code/`, as in
  `packs/text_classification/dlc/`), `kind: triton-python` for the Triton python backend (package
  on Python 3.10 when deps are bundled).
- `smoke(target_dir, target, cm)` runs the built target on a fixed input and raises on a miss.
- `serving_app.py` exposes `Request`, `WARMUP`, `load(model_dir, settings)`, `respond(model, req)`
  for the FastAPI shell; `serving-requirements.txt` lists its deps.

Done when `python main.py --model <id> package --version dev` builds every target and smoke passes.

## 5. Prove it

1. A tiny run: an `experiments.yaml` variant with a few hundred rows and a few steps, then
   `python main.py --model <id> run --variant <tiny>` (real weights: skill `gpu-box-run`).
2. `python main.py --model <id> serve`, then `curl -X POST localhost:8080/api/v1/predict -d ...`.
3. `make test` and `pre-commit run --all-files`.

Done when the run ends with the gate decision you expect (registered on pass), serve answers with
the pack's response, and both checks are green.
