"""PROFILE — where the time goes for N steps of a train or infer workload.

torch.profiler around the steady-state steps (wait/warmup skipped) + an NVML sampler
thread. Writes, under <PERF_DIR>/<role>/<hw_id>/profile/<timestamp>/:
    trace.json        chrome://tracing / Perfetto timeline
    top_kernels.txt   top-k ops by self device time (self CPU time on a CPU-only box)
    summary.json      GPU busy split (compute vs memcpy/memset vs idle) + NVML util/mem
Uses the same `--entry` targets as `perf tune`, and the box's tuned config when one exists.
"""

import json
import threading
import time
from datetime import datetime

from datascience import log
from datascience.constants import paths
from datascience.perf.hw import fingerprint

WAIT, WARMUP = 1, 2


class NvmlSampler(threading.Thread):
    """GPU utilisation % and memory used (GB), sampled every `every` seconds on GPU 0."""

    def __init__(self, every: float = 0.1):
        super().__init__(daemon=True)
        self.every, self.util, self.mem = every, [], []
        self._stop = threading.Event()

    def run(self):
        try:
            import pynvml

            pynvml.nvmlInit()
        except Exception:
            return  # no GPU / no NVML: summary just has no nvml block
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        while not self._stop.is_set():
            self.util.append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
            self.mem.append(pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**30)
            time.sleep(self.every)
        pynvml.nvmlShutdown()

    def stop(self) -> dict:
        self._stop.set()
        self.join()
        if not self.util:
            return {}
        return {"gpu_util_mean": round(sum(self.util) / len(self.util), 1),
                "gpu_util_max": max(self.util),
                "mem_used_gb_max": round(max(self.mem), 2), "samples": len(self.util)}


def _self_device_us(evt) -> float:
    return getattr(evt, "self_device_time_total", None) or getattr(evt, "self_cuda_time_total", 0)


def busy_split(events, wall_ms: float) -> dict:
    """Self device time split into compute vs memcpy/memset; idle = wall - busy (no overlap)."""
    total = sum(_self_device_us(e) for e in events) / 1000
    memcpy = sum(_self_device_us(e) for e in events if "memcpy" in e.key.lower()
                 or "memset" in e.key.lower()) / 1000
    return {"wall_ms": round(wall_ms, 2), "compute_ms": round(total - memcpy, 2),
            "memcpy_ms": round(memcpy, 2), "idle_ms": round(max(wall_ms - total, 0.0), 2),
            "gpu_busy_frac": round(total / wall_ms, 4) if wall_ms else None}


def _out_dir(role: str, fp: dict):
    d = paths.PERF_DIR / role / fp["hw_id"] / "profile" / datetime.now().strftime("%Y%m%d-%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _profiler(active: int):
    import torch
    from torch.profiler import ProfilerActivity, profile, schedule

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    return profile(activities=acts, schedule=schedule(wait=WAIT, warmup=WARMUP, active=active),
                   record_shapes=True, profile_memory=True)


def _sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_infer_steps(target, cfg: dict, steps: int, prof) -> list[float]:
    """Step the profiler around `steps` predict calls (+ wait/warmup); returns per-step ms."""
    predict = target.predictor(cfg["backend"], cfg["precision"])
    batch = target.inputs(cfg["length"], cfg["batch_size"])
    durations = []
    for _ in range(WAIT + WARMUP + steps):
        t = time.perf_counter()
        predict(batch)
        _sync()
        durations.append((time.perf_counter() - t) * 1000)
        prof.step()
    return durations[-steps:]


def run_train_steps(target, overrides: dict, steps: int, prof) -> list[float]:
    from transformers import TrainerCallback

    durations = []

    class Step(TrainerCallback):
        t = None

        def on_step_begin(self, args, state, control, **kw):
            self.t = time.perf_counter()

        def on_step_end(self, args, state, control, **kw):
            _sync()
            durations.append((time.perf_counter() - self.t) * 1000)
            prof.step()

    trainer = target.build_trainer({**overrides, "max_steps": WAIT + WARMUP + steps,
                                    "save_strategy": "no", "eval_strategy": "no",
                                    "report_to": [], "logging_steps": 10**9})
    trainer.add_callback(Step())
    trainer.train()
    return durations[-steps:]


def profile_workload(role: str, target, cfg: dict, steps: int = 10, top_k: int = 25,
                     track: bool = True):
    """Profile `steps` steady-state steps; cfg = tuned config (or defaults) for the role."""
    fp = fingerprint()
    out = _out_dir(role, fp)
    sampler = NvmlSampler()
    sampler.start()
    with _profiler(steps) as prof:
        if role == "infer":
            durations = run_infer_steps(target, cfg, steps, prof)
        else:
            from datascience.perf.workloads.train import training_args

            durations = run_train_steps(target, training_args(cfg), steps, prof)
    nvml = sampler.stop()

    events = prof.key_averages()
    prof.export_chrome_trace(str(out / "trace.json"))
    has_device = any(_self_device_us(e) for e in events)   # False on a CPU-only box
    sort_by = ("self_device_time_total" if hasattr(events[0], "self_device_time_total")
               else "self_cuda_time_total") if has_device else "self_cpu_time_total"
    (out / "top_kernels.txt").write_text(events.table(sort_by=sort_by, row_limit=top_k))
    summary = {"role": role, "hw_id": fp["hw_id"], "config": cfg, "steps": steps,
               "step_ms_mean": round(sum(durations) / len(durations), 3),
               **(busy_split(events, sum(durations)) if has_device
                  else {"wall_ms": round(sum(durations), 2)}),
               "nvml": nvml, "sorted_by": sort_by}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"[perf] profile {role} on {fp['hw_id']} -> {out}")
    if track:
        import os

        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", paths.MLFLOW_URI))
        mlflow.set_experiment("perf")
        with mlflow.start_run(tags={"role": role, "hw_id": fp["hw_id"], "workload": f"profile-{role}"}):
            mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
            mlflow.log_artifacts(str(out))
    return out, summary
