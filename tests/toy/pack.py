"""A dependency-free pack used by the platform tests (no torch, no downloads)."""

import json
from pathlib import Path

from pydantic import BaseModel

from datascience import pack
from datascience.utils.io import atomic_write_json, data_fingerprint, write_jsonl

METRICS = frozenset({"score"})
WARMUP = "warm up"


def _ingest(cm, p):
    rows = [{"text": f"row {i}", "label": "a" if i % 2 else "b"} for i in range(int(cm.params.toy.rows))]
    write_jsonl(p.RAW_DATA, rows)


def _transform(cm, p):
    p.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    p.RAW_DATA.replace(p.TRAIN_FILE)
    atomic_write_json(p.DATA_VERSION, data_fingerprint(p.TRAIN_FILE))


def _train(cm, p):
    p.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (p.MODEL_DIR / "weights.json").write_text(json.dumps({"score": cm.params.toy.score}))
    p.TRAIN_RUN_FILE.unlink(missing_ok=True)
    if cm.params.tracking.backend == "mlflow":
        import mlflow

        mlflow.set_tracking_uri(cm.tracking_uri())
        mlflow.set_experiment(cm.params.tracking.experiment_name)
        with mlflow.start_run() as run:
            mlflow.log_param("score", cm.params.toy.score)
        atomic_write_json(p.TRAIN_RUN_FILE, {"run_id": run.info.run_id})


def _evaluate(cm, p):
    score = json.loads((p.MODEL_DIR / "weights.json").read_text())["score"]
    atomic_write_json(p.METRICS_FILE, {"score": score})


STAGES = [
    ("ingest", _ingest),
    ("transform", _transform),
    ("train", _train),
    ("evaluate", _evaluate),
]


def lineage(cm) -> dict:
    return {"dataset": "toy"}


def targets(cm, p, version):
    return [
        pack.Target(
            name=m.name, kind="pytorch-dlc",
            files={
                "weights.json": p.MODEL_DIR / "weights.json",
                "inference.py": "def model_fn(model_dir):\n    return model_dir\n",
                "model_meta.json": json.dumps({"release_id": version, "mode": m.name}).encode(),
            },
            meta={"serving_model_name": m.name},
        )
        for m in cm.serving().sagemaker.models
    ]


def smoke(target_dir, t, cm):
    meta = json.loads((target_dir / "model_meta.json").read_text())
    if meta["mode"] != t.name:
        raise ValueError(f"smoke: {meta}")
    return {"target": t.name}


class Request(BaseModel):
    text: str


def load(model_dir, settings):
    return json.loads((Path(model_dir) / "weights.json").read_text())


def respond(model, req: Request) -> dict:
    return {"label": "a", "score": model["score"], "text": req.text}
