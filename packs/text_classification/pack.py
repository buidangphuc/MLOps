"""Example pack: a causal LM fine-tuned to classify text (the task posed as generation)."""

import json
from pathlib import Path

from datascience import pack
from packs.text_classification import config
from packs.text_classification.config import validate_config

METRICS = frozenset({"accuracy", "macro_f1", "micro_f1", "mcc", "pr_auc"})


def _ingest(cm, p):
    from packs.text_classification.data_ingestion import DataIngestion

    DataIngestion(config.ingestion(cm)).ingest()


def _validate(cm, p):
    from packs.text_classification.data_validation import DataValidation

    if not DataValidation(config.validation(cm)).validate():
        raise ValueError("data validation failed (see status.txt) — aborting pipeline")


def _transform(cm, p):
    from packs.text_classification.data_transformation import DataTransformation

    DataTransformation(config.transformation(cm)).transform()


def _train(cm, p):
    from packs.text_classification.train import ModelTrainer

    ModelTrainer(config.trainer(cm)).train()


def _evaluate(cm, p):
    from packs.text_classification.evaluate import ModelEvaluation

    ModelEvaluation(config.evaluation(cm)).evaluate()
    _write_serving_meta(cm, p)


def _write_serving_meta(cm, p):
    from datascience.utils.io import atomic_write_json
    from packs.text_classification.model import load_label_map
    from packs.text_classification.serving import META_FILE, serving_meta

    _, id2label = load_label_map(p.LABEL_MAP)
    temp = p.MODEL_DIR / "temperature.json"
    temperature = json.loads(temp.read_text())["temperature"] if temp.exists() else 1.0
    ev = config.evaluation(cm)
    atomic_write_json(p.MODEL_DIR / META_FILE,
                      serving_meta([id2label[i] for i in sorted(id2label)], ev.task, ev.max_length, temperature))


STAGES = [
    ("ingest", _ingest),
    ("validate", _validate),
    ("transform", _transform),
    ("train", _train),
    ("evaluate", _evaluate),
]


def _sanity(cm, p):
    from packs.text_classification.sanity import run_sanity

    run_sanity(config.trainer(cm))


def _dapt(cm, p):
    from packs.text_classification.dapt import run_dapt

    run_dapt(config.dapt(cm))


COMMANDS = {
    "sanity": ("overfit a tiny batch to check the training loop", _sanity),
    "dapt": ("domain-adaptive pretraining -> adapted backbone (artifacts/<model>/dapt)", _dapt),
}


def lineage(cm) -> dict:
    return {"dataset": config.ingestion(cm).dataset_name}


def load_predictor(model_dir=None, **_):
    from packs.text_classification.predict import Predictor

    return Predictor(model_dir=model_dir)


_HERE = Path(__file__).resolve().parent
SMOKE_TEXTS = ["Stocks rally as the central bank holds rates", "The home team wins the final match"]


def targets(cm, p, version):
    out = []
    model_files = {f.relative_to(p.MODEL_DIR).as_posix(): f for f in sorted(p.MODEL_DIR.rglob("*")) if f.is_file()}
    for m in cm.serving().sagemaker.models:
        files = dict(model_files)
        files["code/inference.py"] = _HERE / "dlc" / "inference.py"
        files["code/serving.py"] = _HERE / "serving.py"
        files["code/scoring.py"] = _HERE / "scoring.py"
        files["model_meta.json"] = (json.dumps({"release_id": version, "target": m.name}, indent=2,
                                               sort_keys=True) + "\n").encode()
        out.append(pack.Target(name=m.name, kind="pytorch-dlc", files=files,
                               meta={"runtime_image": m.get("runtime_image")}))
    return out


def smoke(target_dir, t, cm):
    import importlib.util
    import sys

    code = Path(target_dir) / "code"
    sys.modules.pop("serving", None)
    sys.modules.pop("scoring", None)
    sys.path.insert(0, str(code))
    try:
        spec = importlib.util.spec_from_file_location(f"inference_{t.name}", code / "inference.py")
        inference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference)
        model = inference.model_fn(str(target_dir))
        body, _ = inference.output_fn(
            inference.predict_fn(inference.input_fn(json.dumps({"texts": SMOKE_TEXTS}), "application/json"), model),
            "application/json")
    finally:
        sys.path.remove(str(code))
    got = json.loads(body)
    labels = json.loads((Path(target_dir) / "serving_meta.json").read_text())["labels"]
    preds = got["predictions"]
    if len(preds) != len(SMOKE_TEXTS) or any(r["label"] not in labels or not 0 <= r["score"] <= 1 for r in preds):
        raise ValueError(f"{t.name} smoke failed: {got}")
    if got["model_version"] != json.loads((Path(target_dir) / "model_meta.json").read_text())["release_id"]:
        raise ValueError(f"{t.name} smoke: model_version {got['model_version']!r} != the release")
    return got
