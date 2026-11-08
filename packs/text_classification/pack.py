"""Example pack: a causal LM fine-tuned to classify text (the task posed as generation)."""

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
