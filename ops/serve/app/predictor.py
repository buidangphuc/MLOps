"""Lazy singleton around the pack's predictor; the readiness flag is set once the model has
loaded in the background.
"""

import threading

from loguru import logger

from app import pack
from app.config import Settings, get_settings
from app.loader import resolve_model_dir

_predictor = None
_ready = threading.Event()
_load_error: "str | None" = None


def load_predictor(s: Settings | None = None) -> None:
    """Build the singleton once + warm up. Runs in a background thread at startup; a failure is
    logged loudly and leaves the app not-ready (pod won't receive traffic) instead of crashing."""
    global _predictor, _load_error
    s = s or get_settings()
    try:
        p = pack.load(resolve_model_dir(s), s)
        logger.info(f"warmup: {pack.respond(p, pack.Request(text=pack.WARMUP))}")
        _predictor = p
        _ready.set()
        logger.info("model ready")
    except Exception as e:
        _load_error = str(e)
        logger.exception(f"model load FAILED: {e}")


def is_ready() -> bool:
    return _ready.is_set()


def get_predictor():
    if _predictor is None:
        raise RuntimeError(_load_error or "model not loaded yet")
    return _predictor
