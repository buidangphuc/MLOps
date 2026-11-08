"""Pipeline stages: the active model pack's STAGES, plus the runner.

Each stage is independently runnable (the CLI / DVC call them by name). After the
`evaluate` stage the runner gates the metrics and, on pass, versions the model.
"""

import json

from datascience import log
from datascience import pack
from datascience import registry
from datascience.config.configuration import ConfigurationManager
from datascience.constants import paths
from datascience.gate import QualityGate
from datascience.utils.io import read_data_version


def _cm(params_filepath=None):
    """`params_filepath` points at a resolved variant config (see tune/variants.py);
    None means the active model's config/<model>/params.yaml."""
    return ConfigurationManager(params_filepath=params_filepath)


def names() -> list:
    return [n for n, _ in pack.load().STAGES]


def run_stage(name, params_filepath=None, register=True, **kw):
    fn = dict(pack.load().STAGES)[name]
    cm = _cm(params_filepath)
    fn(cm, paths.active(), **kw)
    if name == "evaluate" and register:
        gate_and_register(cm)


def gate_and_register(cm):
    metrics = json.loads(paths.METRICS_FILE.read_text())
    if QualityGate(cm.gate()).check(metrics):  # raises GateFailed if on_fail=error
        scalar = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        lineage = getattr(pack.load(), "lineage", None)
        run_id = json.loads(paths.TRAIN_RUN_FILE.read_text())["run_id"] if paths.TRAIN_RUN_FILE.exists() else None
        metadata = {
            "base_model": cm.params.model.base_model,
            **(lineage(cm) if lineage else {}),
            "data_version": read_data_version(),
            **({"run_id": run_id} if run_id else {}),
        }
        return registry.register(paths.MODEL_DIR, scalar, metadata, cm.registry())
    return None


def run_all(only=None, frm=None, to=None, params_filepath=None):
    all_names = names()
    if only:
        if only not in all_names:
            raise ValueError(f"unknown stage '{only}' (choices: {all_names})")
        selected = {only}
    else:
        i = all_names.index(frm) if frm else 0
        j = all_names.index(to) + 1 if to else len(all_names)
        selected = set(all_names[i:j])

    for name in all_names:
        if name in selected:
            log.info(f">>>>>> {name} started <<<<<<")
            run_stage(name, params_filepath)
            log.info(f">>>>>> {name} done <<<<<<")
