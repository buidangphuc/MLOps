"""Runtime helpers: seeding, device, run-status, retry with backoff."""

import functools
import time

from datascience import log
from datascience.constants import paths
from datascience.utils.io import atomic_write_json


def set_seed(seed: int) -> None:
    """Seed all RNGs for reproducibility (uses HF's set_seed when available)."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except ImportError:
        pass


def pick_device() -> str:
    """cuda > mps > cpu — same code runs on GPU box, Mac, or CI."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_run_status(stage: str, state: str, **extra) -> None:
    """Write a small status file so a server/monitor can see where the run is."""
    atomic_write_json(paths.RUN_STATUS, {"stage": stage, "state": state, **extra})
    log.info(f"[status] {stage}: {state}")


def retry(times: int = 3, base_delay: float = 2.0, exceptions=(Exception,)):
    """Decorator: retry a flaky call (e.g. a download) with exponential backoff."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt == times:
                        raise
                    delay = base_delay * (2 ** (attempt - 1))
                    log.warning(
                        f"{fn.__name__} failed (attempt {attempt}/{times}): {e}; retry in {delay:.0f}s"
                    )
                    time.sleep(delay)

        return wrapper

    return deco


def is_oom_error(err: BaseException) -> bool:
    """Heuristic: is this exception an out-of-memory error (CUDA/MPS/generic)?"""
    msg = str(err).lower()
    return "out of memory" in msg or "oom" in msg or "mps backend out of memory" in msg
