#!/usr/bin/env python3
"""Correctness of torch.ops.afi.fwht vs the host oracle. Needs XPU + afi_xpu.so."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afi.fwht import fwht as fwht_np
from afi.xpu import load, so_path, xmx_ok


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {msg}")


def main() -> int:
    if so_path() is None:
        print("SKIP: afi_xpu.so not built")
        return 0
    import torch

    if not torch.xpu.is_available():
        print("SKIP: no torch.xpu")
        return 0
    load()
    # Stay off Comfy (GPU0) unless ZE_AFFINITY_MASK already pinned us.
    if "ZE_AFFINITY_MASK" not in os.environ and torch.xpu.device_count() > 1:
        torch.xpu.set_device(1)
    dev = torch.device("xpu")
    print("device", torch.xpu.get_device_name(dev))
    print("xmx_ok", xmx_ok())
    rng = np.random.default_rng(7)
    for n in (16, 32, 64, 128, 256):
        x = rng.standard_normal((32, n)).astype(np.float32)
        ref = fwht_np(x, normalize=True).astype(np.float32)
        t = torch.from_numpy(x).to(dev)
        for impl in ("slm", "auto"):
            got = torch.ops.afi.fwht(t, None, impl).cpu().numpy()
            err = float(np.max(np.abs(got - ref)))
            lim = 1e-2 if impl == "auto" else 1e-3
            print(f"impl={impl} n={n} max_abs={err:.4e}")
            check(err < lim, f"{impl} n={n} err={err}")
        for dt, dtol in ((torch.float16, 3e-2), (torch.bfloat16, 6e-2)):
            th = torch.from_numpy(x).to(dev, dtype=dt)
            got_h = torch.ops.afi.fwht(th, None, "auto").cpu().float().numpy()
            err_h = float(np.max(np.abs(got_h - ref)))
            print(f"impl=auto {dt} n={n} max_abs={err_h:.4e}")
            check(err_h < dtol, f"{dt} n={n} err={err_h}")

    # Row-count split: both the small-row (decode) and large-row (prefill)
    # default configs must stay inside the correctness gate.
    for n in (16, 32, 64, 128, 256):
        for rows in (28, 2048):
            xr = rng.standard_normal((rows, n)).astype(np.float32)
            refr = fwht_np(xr, normalize=True).astype(np.float32)
            for dt, dtol in ((torch.float16, 3e-2), (torch.bfloat16, 6e-2)):
                tr = torch.from_numpy(xr).to(dev, dtype=dt)
                got_r = torch.ops.afi.fwht(tr, None, "auto").cpu().float().numpy()
                err_r = float(np.max(np.abs(got_r - refr)))
                print(f"impl=auto {dt} n={n} rows={rows} max_abs={err_r:.4e}")
                check(err_r < dtol, f"{dt} n={n} rows={rows} err={err_r}")

    def bench(n: int, rows: int, dtype, impl: str, iters: int = 50) -> float:
        x = torch.randn(rows, n, device=dev, dtype=dtype)
        for _ in range(10):
            torch.ops.afi.fwht(x, None, impl)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            torch.ops.afi.fwht(x, None, impl)
        torch.xpu.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / iters

    print("microbench (ms / call, after warmup)")
    for n, rows, note in (
        (128, 32, "GDN decode one layer Q+K"),
        (256, 28, "full-attn decode one layer Q+K"),
        (128, 2048, "cpp-bench grain"),
        (256, 2048, "cpp-bench grain"),
    ):
        for dtype, dn in ((torch.float32, "fp32"), (torch.float16, "fp16")):
            ms = bench(n, rows, dtype, "auto")
            print(f"  n={n} rows={rows} {dn}  {ms:.4f} ms  ({note})")
    print("test_fwht_xpu ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
