"""Shared sweep runner: grid -> trials -> constrained pick -> hardware-bound store (+ MLflow).

A workload is any object with:
    name: str            # store key, e.g. "infer" / "train" / "kernel-layernorm"
    role: str            # "train" | "infer" — which hardware.yaml target it binds to
    space(grid) -> list[dict]     # the trial configs (workload may filter unsupported ones)
    trial(cfg)  -> dict           # metrics for one config; OOM / Unsupported -> infeasible
                                  # (a "resolved" key merges into the stored config)

Constraints are yaml-friendly: {"p99_ms": {"max": 50}, "quality_drop": {"max": 0.01}}.
"""

import importlib
import itertools
import time

from datascience import log
from datascience.constants import paths
from datascience.perf import store
from datascience.perf.hw import assert_role
from datascience.utils.runtime import is_oom_error


class Unsupported(Exception):
    """A config this model/backend can't run (e.g. bf16 on an ONNX graph) — skipped, not a bug."""


def grid_points(grid: dict) -> list[dict]:
    """{"a": [1, 2], "b": ["x"]} -> [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]."""
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def resolve_entry(spec: str):
    """'pkg.module:fn' -> fn() — how a caller plugs a model/trainer into a workload."""
    module, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"--entry must be 'module:function', got {spec!r}")
    return getattr(importlib.import_module(module), attr)()


def _cuda():
    try:
        import torch

        return torch if torch.cuda.is_available() else None
    except ImportError:
        return None


def time_ms(fn, iters: int, warmup: int) -> list[float]:
    """Wall-clock per call, GPU-synchronized so async kernels are inside the measurement."""
    torch = _cuda()
    sync = torch.cuda.synchronize if torch else (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    out = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        sync()
        out.append((time.perf_counter() - t) * 1000)
    return out


def reset_peak_mem() -> None:
    torch = _cuda()
    if torch:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_mem_gb() -> float | None:
    torch = _cuda()
    return round(torch.cuda.max_memory_allocated() / 2**30, 3) if torch else None


def violations(metrics: dict, constraints: dict) -> list[str]:
    out = []
    for key, bound in (constraints or {}).items():
        v = metrics.get(key)
        if v is None:
            continue
        if "max" in bound and v > bound["max"]:
            out.append(f"{key}={v} > {bound['max']}")
        if "min" in bound and v < bound["min"]:
            out.append(f"{key}={v} < {bound['min']}")
    return out


def pick(trials: list[dict], objective: str, mode: str = "max") -> dict | None:
    ok = [t for t in trials if t["feasible"] and t["metrics"].get(objective) is not None]
    if not ok:
        return None
    key = lambda t: t["metrics"][objective]  # noqa: E731
    return max(ok, key=key) if mode == "max" else min(ok, key=key)


def sweep(workload, grid: dict, objective: str, mode: str = "max", constraints: dict | None = None,
          fp: dict | None = None, track: bool = True) -> dict:
    """Run every config, keep the best feasible one, store it bound to this box."""
    fp = assert_role(workload.role, fp)
    trials = []
    for cfg in workload.space(grid):
        try:
            metrics = workload.trial(cfg)
        except Unsupported as e:
            trials.append({"config": cfg, "metrics": {}, "feasible": False, "reason": f"unsupported: {e}"})
            continue
        except Exception as e:  # OOM is a result (config too big), anything else is a bug
            if not is_oom_error(e):
                raise
            reset_peak_mem()
            trials.append({"config": cfg, "metrics": {}, "feasible": False, "reason": "oom"})
            log.warning(f"[perf] {workload.name} {cfg}: OOM")
            continue
        cfg = {**cfg, **metrics.pop("resolved", {})}  # e.g. an auto-fit batch size
        bad = violations(metrics, constraints)
        trials.append({"config": cfg, "metrics": metrics, "feasible": not bad,
                       "reason": "; ".join(bad) or None})
        log.info(f"[perf] {workload.name} {cfg}: {metrics}" + (f"  REJECT {bad}" if bad else ""))

    best = pick(trials, objective, mode)
    if best is None:
        raise RuntimeError(f"[perf] {workload.name}: no feasible config in {len(trials)} trials")
    result = {"objective": objective, "mode": mode, "constraints": constraints or {}, "grid": grid,
              "trials": trials, "chosen": best["config"], "chosen_metrics": best["metrics"]}
    path = store.save(workload.role, workload.name, fp, result)
    log.info(f"[perf] {workload.name} on {fp['hw_id']}: chose {best['config']} -> {path}")
    if track:
        _track(workload, fp, trials)
    return result


def _track(workload, fp: dict, trials: list[dict]) -> None:
    import os

    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", paths.MLFLOW_URI))
    mlflow.set_experiment("perf")
    tags = {"role": workload.role, "hw_id": fp["hw_id"], "workload": workload.name}
    for t in trials:
        with mlflow.start_run(tags={**tags, "feasible": str(t["feasible"])}):
            mlflow.log_params(t["config"])
            mlflow.log_metrics({k: v for k, v in t["metrics"].items() if isinstance(v, (int, float))})
