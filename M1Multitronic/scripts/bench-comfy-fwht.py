#!/usr/bin/env python3
"""FWHT-rotated attention overhead at DiT shapes, on Comfy's GPU0 stack.

Run with the Comfy venv python (torch 2.13.0+xpu, SYCL RT 2026.0, XMX live;
the shared afi_xpu.so is auto-picked by afi.xpu.load):

    ZE_AFFINITY_MASK=0 NEO_ReadDeviceBinaryBuiltins=0 \
    PYTHONPATH=$AFI_ROOT/python \
    ~/Documents/OlympusServer/.venv/bin/python scripts/bench-comfy-fwht.py

The rotate is attention-invariant (H orthonormal: softmax(QH(KH)^T)V
== softmax(QK^T)V), so `max diff` must stay at fp16 rounding. The
overhead column is pure cost today — the payoff is the future fused
FWHT+quant attention kernel, not the bare rotate.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from afi.xpu import fwht, load

SHAPES = [
    (24, 2048, 128),
    (24, 4096, 128),
    (32, 4096, 128),
    (24, 8192, 128),
    (16, 4096, 64),
]


def bench(fn, iters: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> int:
    load()
    torch.manual_seed(0)
    print(f"{'shape':>24} {'sdpa ms':>9} {'+fwht ms':>9} {'overhead':>9} "
          f"{'fwht q+k ms':>11} {'max diff':>9}")
    for heads, seq, hd in SHAPES:
        q = torch.randn(1, heads, seq, hd, device="xpu", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        base = lambda: F.scaled_dot_product_attention(q, k, v)  # noqa: E731
        hooked = lambda: F.scaled_dot_product_attention(fwht(q), fwht(k), v)  # noqa: E731
        t_base = bench(base)
        t_hook = bench(hooked)
        t_fwht = bench(lambda: (fwht(q), fwht(k)))  # noqa: E731
        diff = (base().float() - hooked().float()).abs().max().item()
        print(f"({heads:>2}h, {seq:>5}, {hd:>3}) fp16 {t_base:9.3f} "
              f"{t_hook:9.3f} {(t_hook / t_base - 1) * 100:8.1f}% "
              f"{t_fwht:11.3f} {diff:9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
