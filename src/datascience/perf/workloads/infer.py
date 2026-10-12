"""INFER workload — backend × precision × batch sweep, quality-guarded.

Input length is TRAFFIC, not a knob: every trial is measured on each length bucket and
scored on the worst one (max p99, min throughput), so the pick holds for long inputs too.

Task-agnostic: the model comes from `--entry module:fn`, which returns a target with

    predictor(backend: str, precision: str) -> predict(batch)   # raise Unsupported if n/a
    inputs(length: str, batch_size: int)    -> batch
    score(predict) -> float          # optional; higher = better (e.g. span-F1 on a dev slice)
    backends / lengths: list[str]    # optional; override perf.yaml grid.backend / lengths

The builders below (`torch_precision`, `ort_session`, `quantize_int8`) cover the common
backends so an entry stays a few lines. Precisions the INFER TARGET GPU can't run (bf16 on
a T4) are dropped from the grid before anything is measured.
"""

import numpy as np

from datascience.perf.hw import supports
from datascience.perf.runner import Unsupported, peak_mem_gb, reset_peak_mem, time_ms

_FEATURE = {"fp16": "fp16", "bf16": "bf16"}  # precision -> hardware feature it needs


class InferWorkload:
    name = "infer"
    role = "infer"

    def __init__(self, target, lengths=("default",), iters: int = 50, warmup: int = 5,
                 targets: dict | None = None):
        self.target, self.iters, self.warmup, self.targets = target, iters, warmup, targets
        self.lengths = list(getattr(target, "lengths", None) or lengths)
        self._key, self._predict = None, None
        self._scores: dict = {}

    def space(self, grid: dict) -> list[dict]:
        from datascience.perf.runner import grid_points

        grid = {**grid}
        if getattr(self.target, "backends", None):
            grid["backend"] = list(self.target.backends)
        grid["precision"] = [p for p in grid["precision"]
                             if p not in _FEATURE or supports(_FEATURE[p], self.role, self.targets)]
        self._baseline = (grid["backend"][0], "fp32")
        return grid_points(grid)

    def _predictor(self, backend: str, precision: str):
        if self._key != (backend, precision):          # one live model at a time
            self._predict = None
            reset_peak_mem()
            self._predict = self.target.predictor(backend, precision)
            self._key = (backend, precision)
        return self._predict

    def _quality(self, backend: str, precision: str):
        score = getattr(self.target, "score", None)
        if score is None:
            return {}
        if (backend, precision) not in self._scores:
            self._scores[(backend, precision)] = score(self._predictor(backend, precision))
        if self._baseline not in self._scores:
            self._scores[self._baseline] = score(self.target.predictor(*self._baseline))
        q, base = self._scores[(backend, precision)], self._scores[self._baseline]
        return {"quality": round(q, 5), "quality_drop": round(base - q, 5)}

    def trial(self, cfg: dict) -> dict:
        predict = self._predictor(cfg["backend"], cfg["precision"])
        reset_peak_mem()                                 # peak = this batch size, not the last
        per_len = {}
        for length in self.lengths:
            batch = self.target.inputs(length, cfg["batch_size"])
            lat = np.array(time_ms(lambda: predict(batch), self.iters, self.warmup))
            per_len[length] = {
                "p50_ms": round(float(np.percentile(lat, 50)), 3),
                "p99_ms": round(float(np.percentile(lat, 99)), 3),
                "throughput": round(cfg["batch_size"] * 1000.0 / float(lat.mean()), 1),
            }
        return {
            "p50_ms": max(m["p50_ms"] for m in per_len.values()),
            "p99_ms": max(m["p99_ms"] for m in per_len.values()),
            "throughput": min(m["throughput"] for m in per_len.values()),
            "peak_mem_gb": peak_mem_gb(),
            **self._quality(cfg["backend"], cfg["precision"]),
            "by_length": per_len,
        }


# --- builders an entry can reuse ----------------------------------------------------------

def torch_precision(model, precision: str, compile: bool = False):
    """Cast an nn.Module for inference. int8 isn't a torch-eager precision here (use ONNX)."""
    import torch

    dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if precision not in dtypes:
        raise Unsupported(f"torch backend has no {precision}")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu", dtype=dtypes[precision]).eval()
    return torch.compile(model) if compile else model


def quantize_int8(onnx_path, out_path=None) -> str:
    """ORT dynamic INT8 (weights int8, activations quantized at runtime)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_path = str(out_path or str(onnx_path).replace(".onnx", ".int8.onnx"))
    quantize_dynamic(str(onnx_path), out_path, weight_type=QuantType.QInt8)
    return out_path


def ort_session(onnx_path, backend: str, precision: str):
    """onnx -> CUDA EP (fp32, or an int8 graph); tensorrt -> TensorRT EP (fp32/fp16/int8)."""
    import onnxruntime as ort

    if backend == "tensorrt":
        if precision not in ("fp32", "fp16", "int8"):
            raise Unsupported(f"tensorrt has no {precision}")
        opts = {"trt_fp16_enable": precision == "fp16", "trt_int8_enable": precision == "int8",
                "trt_engine_cache_enable": True}
        providers = [("TensorrtExecutionProvider", opts), "CUDAExecutionProvider"]
    else:
        if precision == "int8":
            onnx_path = quantize_int8(onnx_path)
        elif precision != "fp32":
            raise Unsupported(f"onnx backend serves fp32/int8 graphs, not {precision}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), providers=providers)
