"""Stage lifecycle on top of the local registry: staging/production + rollback.

Two fixed stages. State lives next to the versions in
``<root>/<name>/stages.json``::

    {"production": {"version": "v3", "history": ["v1", "v2"]}}

``assign`` pushes the previous holder onto history; ``rollback`` pops it.
Deploy tooling resolves a stage to a concrete model dir via ``stage_path``.
"""

import json
import os

from datascience import log
from datascience.constants import paths

STAGES = ("staging", "production")


def _stages_file(name: str, root: str | None = None) -> str:
    return os.path.join(root or str(paths.REGISTRY_DIR), name, "stages.json")


def _load(name: str, root: str | None = None) -> dict:
    path = _stages_file(name, root)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save(name: str, state: dict, root: str | None = None) -> None:
    path = _stages_file(name, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def _require_version_dir(name: str, version: str, root: str | None = None) -> str:
    version = version if version.startswith("v") else f"v{version}"
    version_dir = os.path.join(root or str(paths.REGISTRY_DIR), name, version)
    if not os.path.isdir(version_dir):
        raise ValueError(f"{name} has no registered version '{version}'")
    return version


def assign_stage(name: str, version: str, stage: str, root: str | None = None) -> str:
    """Point ``stage`` at ``version`` (e.g. promote v3 to production).

    The previous holder is pushed onto the stage's history for rollback.
    Returns the normalized version string.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage '{stage}' (choices: {list(STAGES)})")
    version = _require_version_dir(name, version, root)

    state = _load(name, root)
    entry = state.get(stage, {"version": None, "history": []})
    if entry["version"] == version:
        log.info(f"[stages] {name} {stage} already at {version}")
        return version
    if entry["version"]:
        entry["history"].append(entry["version"])
    entry["version"] = version
    state[stage] = entry
    _save(name, state, root)
    log.info(f"[stages] {name} {stage} -> {version}")
    return version


def rollback(name: str, stage: str, root: str | None = None) -> str:
    """Restore the previous version this stage pointed at."""
    state = _load(name, root)
    entry = state.get(stage)
    if not entry or not entry.get("history"):
        raise ValueError(f"{name} {stage} has no previous version to roll back to")
    previous = entry["history"].pop()
    log.warning(f"[stages] {name} {stage} rollback {entry['version']} -> {previous}")
    entry["version"] = previous
    state[stage] = entry
    _save(name, state, root)
    return previous


def resolve_stage(name: str, stage: str, root: str | None = None) -> str | None:
    """Current version for a stage, or None if unassigned."""
    entry = _load(name, root).get(stage)
    return entry["version"] if entry else None


def stage_path(name: str, stage: str, root: str | None = None) -> str:
    """Model dir a stage points at — what deploy tooling consumes."""
    version = resolve_stage(name, stage, root)
    if version is None:
        raise ValueError(f"{name} has nothing assigned to '{stage}'")
    return os.path.join(root or str(paths.REGISTRY_DIR), name, version)
