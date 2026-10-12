"""Pipeline stages: thin wrappers wiring config -> component, plus the runner.

Each stage is independently runnable (the CLI / DVC call them by name). The
`evaluate` stage also runs the gate and, on pass, versions the model locally.
"""

import json

from datascience import log
from datascience import registry
from datascience.components.data_ingestion import DataIngestion
from datascience.components.data_transformation import DataTransformation
from datascience.components.data_validation import DataValidation
from datascience.components.model_evaluation import ModelEvaluation
from datascience.components.model_trainer import ModelTrainer
from datascience.config.configuration import ConfigurationManager
from datascience.constants import paths
from datascience.gate import QualityGate
from datascience.utils.io import read_data_version


def _cm():
    return ConfigurationManager()


def run_ingest():
    DataIngestion(_cm().ingestion()).ingest()


def run_validate():
    cm = _cm()
    if not DataValidation(cm.validation()).validate():
        raise ValueError("data validation failed (see status.txt) — aborting pipeline")


def run_transform():
    DataTransformation(_cm().transformation()).transform()


def run_train():
    ModelTrainer(_cm().trainer()).train()


def run_evaluate():
    cm = _cm()
    ModelEvaluation(cm.evaluation()).evaluate()

    metrics = json.loads(paths.METRICS_FILE.read_text())
    if QualityGate(cm.gate()).check(metrics):  # raises GateFailed if on_fail=error
        scalar = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        metadata = {
            "base_model": cm.trainer().base_model,
            "dataset": cm.ingestion().dataset_name,
            "data_version": read_data_version(),
        }
        registry.register(paths.MODEL_DIR, scalar, metadata, cm.registry())


def run_sanity():
    from datascience.sanity import run_sanity as _sanity

    _sanity(_cm().trainer())


_STAGES = [
    ("ingest", run_ingest),
    ("validate", run_validate),
    ("transform", run_transform),
    ("train", run_train),
    ("evaluate", run_evaluate),
]


def run_all(only=None, frm=None, to=None):
    names = [n for n, _ in _STAGES]
    if only:
        if only not in names:
            raise ValueError(f"unknown stage '{only}' (choices: {names})")
        selected = {only}
    else:
        i = names.index(frm) if frm else 0
        j = names.index(to) + 1 if to else len(names)
        selected = set(names[i:j])

    for name, fn in _STAGES:
        if name in selected:
            log.info(f">>>>>> {name} started <<<<<<")
            fn()
            log.info(f">>>>>> {name} done <<<<<<")
