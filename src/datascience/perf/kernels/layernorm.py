"""Fused LayerNorm forward in OpenAI Triton — reference kernel for the autotune harness.

One program per row; the whole row sits in one block (BLOCK_SIZE >= hidden), fp32
accumulation. Tunable launch params: BLOCK_SIZE (power of 2 >= N), num_warps, num_stages.
Copy this file to add a kernel; keep the five functions the harness calls.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_fwd(X, Y, W, B, stride, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, axis=0) / N + eps)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)
    tl.store(Y + row * stride + cols, (xc * rstd * w + b).to(Y.dtype.element_ty), mask=mask)


def space(shape, grid: dict) -> list[dict]:
    n = shape[-1]
    first = triton.next_power_of_2(n)          # the row must fit one block
    blocks = [first, first * 2]
    return [{"BLOCK_SIZE": b, "num_warps": w, "num_stages": s}
            for b in blocks for w in grid["num_warps"] for s in grid["num_stages"]]


def make_inputs(shape, dtype):
    torch.manual_seed(0)
    n = shape[-1]
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    w = torch.rand(n, device="cuda", dtype=dtype)
    b = torch.randn(n, device="cuda", dtype=dtype)
    return x, w, b


def run(cfg, inputs, eps=1e-5):
    x, w, b = inputs
    x2 = x.reshape(-1, x.shape[-1])
    y = torch.empty_like(x2)
    _layernorm_fwd[(x2.shape[0],)](x2, y, w, b, x2.stride(0), x2.shape[1], eps,
                                   BLOCK_SIZE=cfg["BLOCK_SIZE"], num_warps=cfg["num_warps"],
                                   num_stages=cfg["num_stages"])
    return y.reshape(x.shape)


def reference(inputs, eps=1e-5):
    x, w, b = inputs
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), w, b, eps)


def bytes_moved(shape, dtype) -> int:
    rows, n = math.prod(shape[:-1]), shape[-1]
    return (2 * rows * n + 2 * n) * torch.finfo(dtype).bits // 8   # read x,w,b + write y
