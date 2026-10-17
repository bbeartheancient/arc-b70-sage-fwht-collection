#!/usr/bin/env python3
"""Validate the XMX FWHT torch op against REAL Qwen3.8 Q/K activations.

Input: models/tmp/qk_real.npz (captured by scripts/capture_qk.py from the
BF16 Qwen38 checkpoint, full-attention layers 3 and 7, head_dim 256).

Checks, on the XPU kernel (impl="auto"):
  1. Kernel correctness on real heavy-tailed rows: afi.fwht(x) vs the
     fp64 host oracle, fp32 / fp16 / bf16.
  2. Rotate invariance of attention scores on real data:
     fp16 SDPA(q, k) vs fp16 SDPA(fwht(q), fwht(k)), both against the
     fp32 score reference (GQA-expanded).
  3. Per-token symmetric INT8 QK^T score error, rotated vs not — the
     accuracy question AGENTS.md deferred until real outliers existed.

Run (GPU1, vLLM's card; services may be live):

    NEO_ReadDeviceBinaryBuiltins=0 PYTHONPATH=$AFI_ROOT/python \
    ~/Documents/OlympusServer/.venv-vllm/bin/python scripts/validate_qk_fwht.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from afi.fwht import fwht as fwht_host
from afi.xpu import load

NPZ = Path(__file__).resolve().parents[1] / "models" / "tmp" / "qk_real.npz"


def int8_scores_err(q: np.ndarray, k: np.ndarray) -> tuple[float, float]:
    """Per-token symmetric INT8 QK^T vs fp32 scores. q:(24,s,d) k:(4,s,d).

    Returns (rms, max) absolute score error after GQA expand. Quantize in
    fp64, matmul in fp64 — this isolates quantizer error, not kernel noise.
    """

    def qt(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.abs(x).max(axis=-1, keepdims=True) / 127.0
        return np.round(x / s).clip(-127, 127), s

    qi, qs = qt(q)
    ki, ks = qt(k)
    # full score tensors per head group
    errs = []
    for g in range(4):
        qg = q[g * 6:(g + 1) * 6]          # (6,s,d)
        kg = k[g:g + 1]                     # (1,s,d)
        ref = np.einsum("hsd,td->hst", qg, kg[0])
        got = np.einsum("hsd,td->hst", qi[g * 6:(g + 1) * 6], ki[g])
        got = got * (qs[g * 6:(g + 1) * 6, :, 0:1] * ks[g, 0, 0])
        errs.append((got - ref).ravel())
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e**2))), float(np.abs(e).max())


def main() -> int:
    import torch
    import torch.nn.functional as F

    data = np.load(NPZ)
    print("arrays:", {k: data[k].shape for k in data.files})
    load()
    if torch.xpu.device_count() > 1:
        import os
        if "ZE_AFFINITY_MASK" not in os.environ:
            torch.xpu.set_device(1)
    dev = torch.device("xpu")

    print("\n== 1. kernel correctness on real rows (n=256) ==")
    for name in ("q3_post", "k3_post", "q7_post", "k7_post"):
        x = data[name].reshape(-1, 256).astype(np.float32)
        ref = fwht_host(x, normalize=True)
        row = f"{name:>8} max|x|={np.abs(x).max():7.2f}"
        for dt, tol in ((torch.float32, 2e-3), (torch.float16, 3e-2),
                        (torch.bfloat16, 6e-2)):
            t = torch.from_numpy(x).to(dev, dtype=dt)
            got = torch.ops.afi.fwht(t, None, "auto").cpu().float().numpy()
            err = float(np.max(np.abs(got - ref)))
            ok = "OK " if err < tol else "FAIL"
            row += f"  {str(dt).split('.')[-1]}:{err:.2e}({ok})"
            if err >= tol:
                raise SystemExit(f"FAIL: {name} {dt} err={err}")
        print(row)

    print("\n== 2. rotate invariance of fp16 SDPA on real q/k ==")
    for qn, kn in (("q3_post", "k3_post"), ("q7_post", "k7_post")):
        q = torch.from_numpy(data[qn][None]).to(dev)   # (1,24,s,256) fp32
        k = torch.from_numpy(data[kn][None]).to(dev)   # (1,4,s,256)
        kx = k.repeat_interleave(6, dim=1)
        ref = F.scaled_dot_product_attention(q.float(), kx.float(), q.float())
        out = {}
        for tag, rot in (("plain", False), ("fwht", True)):
            qh, kh = q.half(), k.half()
            if rot:
                qh = torch.ops.afi.fwht(qh, None, "auto")
                kh = torch.ops.afi.fwht(kh, None, "auto")
            o = F.scaled_dot_product_attention(
                qh, kh.repeat_interleave(6, dim=1), q.half())
            out[tag] = o
        e_plain = (out["plain"].float() - ref).abs()
        e_rot = (out["fwht"].float() - ref).abs()
        d_rots = (out["plain"].float() - out["fwht"].float()).abs().max()
        print(f"  {qn}/{kn}: fp16 plain max {e_plain.max():.4f} rms "
              f"{e_plain.pow(2).mean().sqrt():.5f} | fp16+fwht max "
              f"{e_rot.max():.4f} rms {e_rot.pow(2).mean().sqrt():.5f} | "
              f"rotate-vs-plain {d_rots:.5f}")

    print("\n== 3. per-token INT8 QK^T score error on real data ==")
    for qn, kn in (("q3_post", "k3_post"), ("q7_post", "k7_post")):
        q = data[qn].astype(np.float64)
        k = data[kn].astype(np.float64)
        rms0, mx0 = int8_scores_err(q, k)
        qr = fwht_host(q, normalize=True)
        kr = fwht_host(k, normalize=True)
        rms1, mx1 = int8_scores_err(qr, kr)
        print(f"  {qn}/{kn}: plain rms {rms0:.4f} max {mx0:.3f} | "
              f"rotated rms {rms1:.4f} max {mx1:.3f} | "
              f"rms ratio {rms0 / max(rms1, 1e-12):.2f}x")
    print("\nvalidate_qk_fwht ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
