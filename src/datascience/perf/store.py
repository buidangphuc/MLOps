"""Tuned-result store, bound to hardware: <PERF_DIR>/<role>/<hw_id>/<workload>.json.

One file = one sweep: the fingerprint it ran on, the grid, every trial, and the chosen
config. `load_tuned` only ever reads the current box's file — a result tuned elsewhere is a
HardwareMismatch (re-tune here), never a fallback.
"""

import json
from pathlib import Path

from datascience.constants import paths
from datascience.perf.hw import ROLES, HardwareMismatch, fingerprint
from datascience.utils.io import atomic_write_json


def result_path(role: str, hw_id: str, workload: str) -> Path:
    if role not in ROLES:
        raise ValueError(f"unknown role '{role}' (choices: {list(ROLES)})")
    return paths.PERF_DIR / role / hw_id / f"{workload}.json"


def save(role: str, workload: str, fp: dict, result: dict) -> Path:
    """result: {"grid", "trials", "chosen", ...} from the runner."""
    path = result_path(role, fp["hw_id"], workload)
    atomic_write_json(path, {"role": role, "workload": workload, "hw_id": fp["hw_id"],
                             "fingerprint": fp, **result})
    return path


def load(role: str, workload: str, fp: dict | None = None) -> dict:
    fp = fp or fingerprint()
    path = result_path(role, fp["hw_id"], workload)
    if path.exists():
        return json.loads(path.read_text())
    elsewhere = [p.parent.name for p in (paths.PERF_DIR / role).glob(f"*/{workload}.json")]
    if elsewhere:
        raise HardwareMismatch(
            f"{role}/{workload} was tuned on {elsewhere}, this box is {fp['hw_id']} — "
            f"re-tune here (`datascience perf tune {workload}`), don't map configs across GPUs"
        )
    raise FileNotFoundError(f"no tuned {role}/{workload} result yet for {fp['hw_id']}")


def load_tuned(role: str, workload: str, fp: dict | None = None) -> dict:
    """The chosen config for this role/workload on THIS box."""
    return load(role, workload, fp)["chosen"]


def list_results(role: str | None = None, hw_id: str | None = None) -> list[dict]:
    rows = []
    for path in sorted(paths.PERF_DIR.glob("*/*/*.json")):
        r, h = path.parent.parent.name, path.parent.name
        if (role and r != role) or (hw_id and h != hw_id):
            continue
        data = json.loads(path.read_text())
        rows.append({"role": r, "hw_id": h, "workload": data["workload"],
                     "trials": len(data.get("trials", [])), "chosen": data.get("chosen"),
                     "path": str(path)})
    return rows
