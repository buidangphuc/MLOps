"""I/O helpers: robust JSONL read/write + atomic writes.

Atomic write = write to a temp file in the same dir then rename, so an
interrupted run never leaves a half-written (corrupt) artifact.
"""

import contextlib
import hashlib
import json
import os
import tempfile

from pathlib import Path

from src.datascience import log


@contextlib.contextmanager
def _atomic(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        yield tmp
        os.replace(tmp, path)  # atomic on the same filesystem
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_text(path, text: str) -> None:
    with _atomic(path) as tmp:
        Path(tmp).write_text(text, encoding="utf-8")


def atomic_write_json(path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False))


def data_fingerprint(path) -> dict:
    """Content hash + row count of a data file — a lightweight data 'version' for lineage.
    Remote/DVC tracking is the reserved seam; this captures WHICH data without one."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
        "n_rows": sum(1 for line in text.splitlines() if line.strip()),
    }


def read_data_version():
    """The data version written by `transform` (hash + n_rows), or None if absent."""
    from src.datascience.constants import paths

    return json.loads(paths.DATA_VERSION.read_text()) if paths.DATA_VERSION.exists() else None


def read_jsonl(path, required_fields=None) -> list:
    """Load JSONL rows; skip + count malformed/incomplete lines (don't crash on one bad row)."""
    path = Path(path)
    rows, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if required_fields and not all(k in obj for k in required_fields):
                bad += 1
                continue
            rows.append(obj)
    if bad:
        log.warning(f"{path.name}: skipped {bad} malformed/incomplete line(s)")
    if not rows:
        raise ValueError(f"no valid rows in {path}")
    return rows


def write_jsonl(path, rows) -> None:
    with _atomic(path) as tmp:
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
