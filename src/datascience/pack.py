"""Model packs: one plain module per model at packs/<pack>/pack.py.

A pack exposes `STAGES = [(name, fn(cm, p, **kw)), ...]` (must contain `evaluate`, which
writes p.METRICS_FILE) and `METRICS` (valid metric names), plus optionally `validate_config`,
`lineage`, `load_predictor`, `targets`, `smoke`, `COMMANDS`. The platform owns paths, the
runner, gate, registry, tune, package writers and release; the pack owns everything else.
"""

import importlib
from dataclasses import dataclass, field

import yaml

from datascience.constants import paths


def name(model_id=None) -> str:
    p = paths.for_model(model_id) if model_id else paths.active()
    return (yaml.safe_load(p.PARAMS_FILE.read_text()) or {}).get("pack") or p.MODEL_ID


def load(model_id=None):
    return importlib.import_module(f"packs.{name(model_id)}.pack")


@dataclass(frozen=True)
class Target:
    """One deployable artifact. `kind` picks the platform writer in package.py:
    `triton-python` (dir tar, optional bundled deps) or `pytorch-dlc` (deterministic flat tar).
    `files` maps archive name -> Path | bytes | str; `meta` lands in the release manifest."""
    name: str
    kind: str
    files: dict
    pip: tuple = ()
    pip_no_deps: tuple = ()
    meta: dict = field(default_factory=dict)
