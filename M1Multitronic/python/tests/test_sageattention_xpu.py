#!/usr/bin/env python3
"""Correctness of the sageattention API drop-in (SDPA/oneDNN backend) on XPU.

Mirrors test_fwht_xpu.py: standalone script, returns nonzero on failure,
skips (exit 0) when there is no XPU device. Needs torch.xpu (2026.0) only.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from sageattention import (  # noqa: E402
    sageattn,
    sageattn_varlen,
    sageattn_qk_int8_pv_fp16_triton,
    sageattn_qk_int8_pv_fp8_cuda,
)


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {msg}")


def sdpa_ref(q, k, v, scale, attn_mask=None, is_causal=False):
    s = (q.float() @ k.float().transpose(-2, -1)) * scale
    if is_causal:
        s = s.masked_fill(~torch.ones(s.shape[-1], s.shape[-1], device=s.device).tril().bool(), -float("inf"))
    if attn_mask is not None:
        s = s + attn_mask.float()
    return torch.softmax(s, dim=-1) @ v.float()


def main() -> int:
    if not torch.xpu.is_available():
        print("SKIP: no torch.xpu")
        return 0
    # Stay off Comfy (GPU0) unless ZE_AFFINITY_MASK already pinned us.
    if "ZE_AFFINITY_MASK" not in os.environ and torch.xpu.device_count() > 1:
        torch.xpu.set_device(1)
    dev = torch.device("xpu")
    torch.manual_seed(0)
    print("device", torch.xpu.get_device_name(dev))

    H, S, D = 8, 64, 64
    q = torch.randn(1, H, S, D, device=dev)  # (B, H, S, D) = "HND"
    k = torch.randn(1, H, S, D, device=dev)
    v = torch.randn(1, H, S, D, device=dev)
    scale = D ** -0.5
    m = torch.zeros(1, 1, S, S, device=dev)
    m[:, :, S // 2 :, : S // 2] = float("-inf")

    for dtype, tol in ((torch.float16, 3e-3), (torch.bfloat16, 1e-2)):
        qd, kd, vd = q.to(dtype), k.to(dtype), v.to(dtype)
        ref = sdpa_ref(qd, kd, vd, scale)
        for layout in ("HND", "NHD"):
            ql, kl, vl = (q, k, v)
            if layout == "NHD":
                ql, kl, vl = (x.transpose(1, 2).contiguous() for x in (ql, kl, vl))
            o = sageattn(ql.to(dtype), kl.to(dtype), vl.to(dtype), tensor_layout=layout, sm_scale=scale)
            if layout == "NHD":
                o = o.transpose(1, 2).contiguous()
            err = float((o.float() - ref).abs().max())
            print(f"dtype={dtype} layout={layout} max_abs={err:.4e}")
            check(err < tol, f"{dtype} {layout} err={err}")
        o_c = sageattn(qd, kd, vd, tensor_layout="HND", is_causal=True, sm_scale=scale)
        ref_c = sdpa_ref(qd, kd, vd, scale, is_causal=True)
        err_c = float((o_c.float() - ref_c).abs().max())
        print(f"dtype={dtype} causal max_abs={err_c:.4e}")
        check(err_c < tol, f"{dtype} causal err={err_c}")
        o_m = sageattn(qd, kd, vd, tensor_layout="HND", attn_mask=m, sm_scale=scale)
        ref_m = sdpa_ref(qd, kd, vd, scale, attn_mask=m)
        err_m = float((o_m.float() - ref_m).abs().max())
        print(f"dtype={dtype} mask max_abs={err_m:.4e}")
        check(err_m < tol, f"{dtype} mask err={err_m}")

    lse = sageattn(q.half(), k.half(), v.half(), tensor_layout="HND", sm_scale=scale, return_lse=True)[1]
    check(bool(torch.isfinite(lse).all()), "return_lse finite")

    for name, fn in (("qk_i8_pv_fp16_triton", sageattn_qk_int8_pv_fp16_triton),
                     ("qk_i8_pv_fp8_cuda", sageattn_qk_int8_pv_fp8_cuda)):
        o = fn(q.half(), k.half(), v.half(), tensor_layout="HND", pv_accum_dtype="fp32")
        err = float((o.float() - sdpa_ref(q, k, v, scale)).abs().max())
        print(f"{name} max_abs={err:.4e}")
        check(err < 3e-3, f"{name} err={err}")

    qv = torch.randn(42, H, D, device=dev, dtype=torch.float16)
    kv = torch.randn(63, H, D, device=dev, dtype=torch.float16)
    vv = torch.randn(63, H, D, device=dev, dtype=torch.float16)
    cu_q = torch.tensor([0, 3, 8, 17, 42], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, 6, 15, 31, 63], dtype=torch.int32, device=dev)
    refs = []
    for i in range(4):
        qs = qv[cu_q[i] : cu_q[i + 1]].transpose(0, 1).unsqueeze(0)
        ks = kv[cu_k[i] : cu_k[i + 1]].transpose(0, 1).unsqueeze(0)
        vs = vv[cu_k[i] : cu_k[i + 1]].transpose(0, 1).unsqueeze(0)
        refs.append(sdpa_ref(qs, ks, vs, scale).squeeze(0).transpose(0, 1))
    ov = sageattn_varlen(qv, kv, vv, cu_q, cu_k, 20, 20)
    err_v = float((ov - torch.cat(refs)).abs().max())
    print(f"varlen max_abs={err_v:.4e}")
    check(err_v < 3e-3, f"varlen err={err_v}")

    def bench():
        def one():
            for _ in range(20):
                sageattn(q.half(), k.half(), v.half(), tensor_layout="HND", sm_scale=scale)
            torch.xpu.synchronize()
        one()
        t0 = time.perf_counter()
        one()
        return (time.perf_counter() - t0) / 20 * 1000.0
    print(f"microbench fp16 (1,{H},{S},{D}) ~{bench():.3f} ms/call")
    print("test_sageattention_xpu ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())