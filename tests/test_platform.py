"""Platform seams driven by the dependency-free toy pack: runner, gate, registry, tune, CLI."""

import json

import yaml

from datascience.constants import paths


def _mlflow():
    from mlflow import MlflowClient

    import os

    uri = os.environ["MLFLOW_TRACKING_URI"]
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def test_run_gates_and_registers_with_the_alias(toy_model):
    from datascience.pipeline import stages

    stages.run_all()

    assert json.loads(toy_model.METRICS_FILE.read_text()) == {"score": 0.7}
    assert (toy_model.REGISTRY_DIR / "ToyModel" / "v1" / "weights.json").exists()
    mv = _mlflow().get_model_version_by_alias("ToyModel", "champion")
    assert str(mv.version) == "1"
    assert mv.tags["dataset"] == "toy" and mv.tags["base_model"] == "toy-base"
    assert mv.run_id == json.loads(toy_model.TRAIN_RUN_FILE.read_text())["run_id"]


def test_a_failing_gate_does_not_register(toy_model):
    from datascience.pipeline import stages

    params = yaml.safe_load(toy_model.PARAMS_FILE.read_text())
    params["toy"]["score"] = 0.1
    toy_model.PARAMS_FILE.write_text(yaml.safe_dump(params))

    stages.run_all()

    assert not (toy_model.REGISTRY_DIR / "ToyModel").exists()


def test_tune_runs_every_stage_per_variant_and_promote_versions_the_winner(toy_model):
    from datascience.tune.loop import run_tuning
    from datascience.tune.promote import promote

    toy_model.EXPERIMENTS_FILE.write_text(yaml.safe_dump({
        "experiment_name": "toy-tuning",
        "variants": [
            {"name": "small-data", "toy": {"rows": 2, "score": 0.6}},
            {"name": "big-data", "toy": {"rows": 10, "score": 0.9}},
        ],
    }))

    runs = run_tuning()

    assert [r["variant"] for r in runs] == ["big-data", "small-data"]
    versions = {r["variant"]: json.loads((toy_model.TUNE_DIR / r["variant"] / "data_version.json").read_text())
                for r in runs}
    assert versions["big-data"]["n_rows"] == 10 and versions["small-data"]["n_rows"] == 2
    assert all(r["run_id"] for r in runs)

    promote("small-data")

    meta = json.loads((toy_model.REGISTRY_DIR / "ToyModel" / "v1" / "registry_meta.json").read_text())
    assert meta["metadata"]["variant"] == "small-data"
    assert meta["metadata"]["data_version"]["n_rows"] == 2
    assert _mlflow().get_model_version_by_alias("ToyModel", "champion").tags["variant"] == "small-data"


def test_cli_builds_stage_commands_from_the_pack(toy_model, monkeypatch):
    from datascience import cli

    monkeypatch.setenv("DS_MODEL", "toy")
    cli.main(["--model", "toy", "run", "--to", "transform"])

    assert paths.active().MODEL_ID == "toy"
    assert toy_model.DATA_VERSION.exists() and not toy_model.MODEL_DIR.exists()
    parser = cli.build_parser([n for n, _ in __import__("packs.toy.pack", fromlist=["x"]).STAGES])
    assert parser.parse_args(["evaluate"]).command == "evaluate"
