"""Hardware fingerprint + role binding — the core invariant of the perf platform.

Every tuned result is keyed by `hw_id` (GPU model, compute capability, VRAM, count) and a
role (`train` | `infer`, declared in config/hardware.yaml). Software versions are recorded
(`sw`) but don't change the key: a driver bump doesn't invalidate a batch size, a different
GPU does.

Reads NVML (pynvml, dev extra) — no torch import, so it's cheap and works on a serving box.
No GPU visible -> a CPU fingerprint, which never matches a GPU target.
"""

import hashlib
import os
import platform
import re
from importlib import metadata

from datascience.constants import paths

ROLES = ("train", "infer")

# min compute capability per feature (NVIDIA arch tables)
_MIN_CC = {"fp16": 7.0, "bf16": 8.0, "tf32": 8.0, "fp8": 8.9, "int8": 7.0, "int4": 8.0, "nf4": 7.0}


class HardwareMismatch(RuntimeError):
    """The current box is not the one a role / tuned result is bound to."""


def _version(*dists):
    for d in dists:
        try:
            return metadata.version(d)
        except metadata.PackageNotFoundError:
            continue
    return None


def _gpus() -> tuple[list[dict], dict]:
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:  # no pynvml / no driver / no GPU
        return [], {}
    try:
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
            gpus.append({
                "name": name.decode() if isinstance(name, bytes) else name,
                "compute_capability": f"{major}.{minor}",
                "vram_gb": round(pynvml.nvmlDeviceGetMemoryInfo(h).total / 2**30),
            })
        cuda = pynvml.nvmlSystemGetCudaDriverVersion()
        driver = pynvml.nvmlSystemGetDriverVersion()
        sys_info = {"driver": driver.decode() if isinstance(driver, bytes) else driver,
                    "cuda_driver": f"{cuda // 1000}.{cuda % 1000 // 10}"}
        return gpus, sys_info
    finally:
        pynvml.nvmlShutdown()


def _hw_id(gpus: list[dict]) -> str:
    if gpus:
        g = gpus[0]
        key = "|".join(f"{x['name']}|{x['compute_capability']}|{x['vram_gb']}" for x in gpus)
        slug = f"{g['name']}-{len(gpus)}x"
    else:
        cpu = platform.processor() or platform.machine()
        key = f"cpu|{cpu}|{os.cpu_count()}"
        slug = f"cpu-{cpu}-{os.cpu_count()}c"
    slug = re.sub(r"[^a-z0-9]+", "-", slug.lower()).strip("-")
    return f"{slug}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"


def fingerprint() -> dict:
    """What this box is. `hw_id` is the binding key; `sw` is recorded, never matched."""
    gpus, sys_info = _gpus()
    return {
        "hw_id": _hw_id(gpus),
        "gpus": gpus,
        "sw": {
            **sys_info,
            "torch": _version("torch"),
            "onnxruntime": _version("onnxruntime-gpu", "onnxruntime"),
            "tensorrt": _version("tensorrt"),
            "triton": _version("triton"),
        },
    }


def load_targets() -> dict:
    import yaml

    return yaml.safe_load(paths.HARDWARE_FILE.read_text())["targets"]


def mismatches(role: str, fp: dict, targets: dict | None = None) -> list[str]:
    """Why `fp` doesn't satisfy targets[role] (empty list = it does)."""
    if role not in ROLES:
        raise ValueError(f"unknown role '{role}' (choices: {list(ROLES)})")
    t = (targets or load_targets())[role]
    if not t.get("gpu_name") or not t.get("compute_capability"):
        return [f"hardware.yaml targets.{role} is not filled (gpu_name, compute_capability)"]
    if not fp["gpus"]:
        return [f"no GPU visible, targets.{role} wants {t['gpu_name']}"]
    g, out = fp["gpus"][0], []
    if t["gpu_name"].lower() not in g["name"].lower():
        out.append(f"gpu {g['name']!r} != target {t['gpu_name']!r}")
    if str(t["compute_capability"]) != g["compute_capability"]:
        out.append(f"compute capability {g['compute_capability']} != target {t['compute_capability']}")
    if t.get("gpu_count") and len(fp["gpus"]) != t["gpu_count"]:
        out.append(f"gpu count {len(fp['gpus'])} != target {t['gpu_count']}")
    if t.get("min_vram_gb") and g["vram_gb"] < t["min_vram_gb"]:
        out.append(f"vram {g['vram_gb']}GB < target {t['min_vram_gb']}GB")
    return out


def assert_role(role: str, fp: dict | None = None) -> dict:
    """Refuse to tune/apply for `role` on a box that isn't that role's target. Returns fp."""
    fp = fp or fingerprint()
    bad = mismatches(role, fp)
    if bad:
        raise HardwareMismatch(f"this box ({fp['hw_id']}) is not the {role} target: " + "; ".join(bad))
    return fp


def supports(feature: str, role: str, targets: dict | None = None) -> bool:
    """Does the role's TARGET GPU support `feature` (bf16/fp16/tf32/fp8/int8/int4/nf4)? Decided from the
    target, not from the current box — a bf16 serving config is wrong for a T4 endpoint even
    if it was measured on an A100."""
    cc = (targets or load_targets())[role].get("compute_capability")
    return cc is not None and float(cc) >= _MIN_CC[feature]
