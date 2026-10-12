"""KERNEL workload — launch-config autotune for one kernel at one input shape.

Shape is the input, not a knob: each shape is its own sweep, stored as
`kernel-<name>-<shape>` under the role/hw_id it ran on. Every config is checked against
the torch reference first; a config that's fast but wrong is infeasible (max_abs_err).
"""

import importlib

_DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
TOLERANCE = {"fp32": 1e-4, "fp16": 1e-2, "bf16": 5e-2}


class KernelWorkload:
    def __init__(self, kernel: str, shape: list[int], dtype: str = "fp16", role: str = "train",
                 module=None):
        self.kernel = module or importlib.import_module(f"datascience.perf.kernels.{kernel}")
        self.shape, self.dtype_name, self.role = list(shape), dtype, role
        self.name = f"kernel-{kernel}-{'x'.join(map(str, shape))}-{dtype}"
        self._inputs = self._ref = None

    def _dtype(self):
        import torch

        return getattr(torch, _DTYPES[self.dtype_name])

    def space(self, grid: dict) -> list[dict]:
        return self.kernel.space(self.shape, grid)

    def trial(self, cfg: dict) -> dict:
        import torch
        from triton.testing import do_bench

        if self._inputs is None:
            self._inputs = self.kernel.make_inputs(self.shape, self._dtype())
            self._ref = self.kernel.reference(self._inputs)
        out = self.kernel.run(cfg, self._inputs)
        err = float((out.float() - self._ref.float()).abs().max())
        ms = do_bench(lambda: self.kernel.run(cfg, self._inputs), warmup=25, rep=100)
        gbps = self.kernel.bytes_moved(self.shape, self._dtype()) / (ms * 1e-3) / 1e9
        torch.cuda.synchronize()
        return {"ms": round(ms, 5), "gbps": round(gbps, 1), "max_abs_err": err}

    def constraints(self) -> dict:
        return {"max_abs_err": {"max": TOLERANCE[self.dtype_name]}}
