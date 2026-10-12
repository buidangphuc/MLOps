"""TRAIN workload — training-throughput sweep that never changes the optimisation.

Searches precision × TF32 × torch.compile × gradient checkpointing × dataloader workers, and
for each combo auto-fits the largest per-device batch that doesn't OOM. The EFFECTIVE batch
is held constant (per_device × grad_accum × gpus = effective_batch), so the chosen config
changes speed only, never what the model learns.

Task-agnostic: the trainer comes from `--entry module:fn`, which returns a target with

    build_trainer(overrides: dict) -> transformers.Trainer   # apply overrides to its TrainingArguments
    flops_per_sample: float          # optional -> MFU (needs hardware.yaml targets.train.peak_tflops)

`training_args(chosen)` is the same mapping a trainer hook applies from `load_tuned("train", "train")`.
"""

import gc
import time

from datascience.perf.hw import load_targets, supports
from datascience.perf.runner import Unsupported, grid_points, peak_mem_gb, reset_peak_mem
from datascience.utils.runtime import is_oom_error


def batch_candidates(effective_batch: int, gpus: int) -> list[int]:
    """Per-device batch sizes that keep the effective batch exact (divisors of the per-GPU share)."""
    if effective_batch % gpus:
        raise ValueError(f"effective_batch {effective_batch} not divisible by {gpus} GPUs")
    share = effective_batch // gpus
    return [b for b in range(1, share + 1) if share % b == 0]


def grad_accum(effective_batch: int, per_device: int, gpus: int) -> int:
    accum, rem = divmod(effective_batch, per_device * gpus)
    if rem:
        raise ValueError(f"{per_device} x {gpus} GPUs doesn't divide effective_batch {effective_batch}")
    return accum


def max_fit(candidates: list[int], fits) -> int:
    """Largest candidate where fits(b) is True — memory is monotone in batch, so bisect."""
    lo, hi, best = 0, len(candidates) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if fits(candidates[mid]):
            best, lo = candidates[mid], mid + 1
        else:
            hi = mid - 1
    if best is None:
        raise RuntimeError("CUDA out of memory even at per-device batch 1")
    return best


def training_args(cfg: dict) -> dict:
    """Tuned config -> HF TrainingArguments kwargs."""
    out = {
        "fp16": cfg["precision"] == "fp16",
        "bf16": cfg["precision"] == "bf16",
        "tf32": cfg.get("tf32", False),
        "torch_compile": cfg.get("torch_compile", False),
        "gradient_checkpointing": cfg.get("gradient_checkpointing", False),
        "dataloader_num_workers": cfg.get("dataloader_num_workers", 0),
        "dataloader_pin_memory": True,
    }
    if "per_device_batch" in cfg:
        out["per_device_train_batch_size"] = cfg["per_device_batch"]
        out["gradient_accumulation_steps"] = cfg["grad_accum"]
    return out


def _timer(warmup_steps: int):
    import torch
    from transformers import TrainerCallback

    class StepTimer(TrainerCallback):
        t0 = t1 = None
        steps = 0

        def on_step_end(self, args, state, control, **kw):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if state.global_step == warmup_steps:          # compile/cudnn autotune are warmup
                self.t0 = time.perf_counter()
            elif state.global_step > warmup_steps:
                self.steps, self.t1 = self.steps + 1, time.perf_counter()

    return StepTimer()


class TrainWorkload:
    name = "train"
    role = "train"

    def __init__(self, target, effective_batch: int, steps: int = 30, warmup_steps: int = 5,
                 targets: dict | None = None):
        self.target, self.effective_batch = target, effective_batch
        self.steps, self.warmup_steps = steps, warmup_steps
        self.targets = targets or load_targets()
        self.gpus = self.targets["train"].get("gpu_count") or 1

    def space(self, grid: dict) -> list[dict]:
        grid = {**grid}
        grid["precision"] = [p for p in grid["precision"]
                             if p == "fp32" or supports(p, self.role, self.targets)]
        if not supports("tf32", self.role, self.targets):
            grid["tf32"] = [False]
        return grid_points(grid)

    def _run(self, cfg: dict, per_device: int, steps: int):
        overrides = {
            **training_args({**cfg, "per_device_batch": per_device,
                             "grad_accum": grad_accum(self.effective_batch, per_device, self.gpus)}),
            "max_steps": steps, "save_strategy": "no", "eval_strategy": "no",
            "report_to": [], "logging_steps": 10**9,
        }
        timer = _timer(self.warmup_steps)
        trainer = self.target.build_trainer(overrides)
        trainer.add_callback(timer)
        try:
            trainer.train()
        finally:
            del trainer
            gc.collect()
        return timer

    def _fits(self, cfg: dict, per_device: int) -> bool:
        reset_peak_mem()
        try:
            self._run(cfg, per_device, steps=2)
            return True
        except Exception as e:
            if not is_oom_error(e):
                raise
            reset_peak_mem()
            return False

    def trial(self, cfg: dict) -> dict:
        if cfg["precision"] not in ("fp32", "fp16", "bf16"):
            raise Unsupported(f"train precision {cfg['precision']}")
        per_device = cfg.get("per_device_batch") or max_fit(
            batch_candidates(self.effective_batch, self.gpus), lambda b: self._fits(cfg, b))
        reset_peak_mem()
        timer = self._run(cfg, per_device, self.steps + self.warmup_steps)
        elapsed = timer.t1 - timer.t0
        samples_s = timer.steps * self.effective_batch / elapsed
        metrics = {
            "samples_per_s": round(samples_s, 2),
            "step_ms": round(elapsed / timer.steps * 1000, 2),
            "peak_mem_gb": peak_mem_gb(),
            "resolved": {"per_device_batch": per_device,
                         "grad_accum": grad_accum(self.effective_batch, per_device, self.gpus)},
        }
        flops, peak = getattr(self.target, "flops_per_sample", None), self.targets["train"].get("peak_tflops")
        if flops and peak:
            metrics["mfu"] = round(samples_s * flops / (peak * 1e12 * self.gpus), 4)
        return metrics
