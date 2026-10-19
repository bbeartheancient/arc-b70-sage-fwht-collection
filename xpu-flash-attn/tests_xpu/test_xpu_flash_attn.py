# SPDX-License-Identifier: Apache-2.0
"""Correctness and micro-benchmark tests for xpu_flash_attn."""
import math
import time

import torch

import xpu_flash_attn as xfa


def _manual_ref(q, k, v, causal=False, scale=None):
    """Plain PyTorch attention reference (B,S,H,D layout)."""
    d = q.size(-1)
    if scale is None:
        scale = d ** -0.5
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    # handle GQA in reference
    if k_t.size(1) != q_t.size(1) and q_t.size(1) % k_t.size(1) == 0:
        r = q_t.size(1) // k_t.size(1)
        k_t = k_t.repeat_interleave(r, dim=1)
        v_t = v_t.repeat_interleave(r, dim=1)
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
    if causal:
        sq, sk = scores.size(-2), scores.size(-1)
        q_idx = torch.arange(sq, device=q.device).view(sq, 1)
        k_idx = torch.arange(sk, device=q.device).view(1, sk)
        mask = k_idx <= q_idx + (sk - sq)
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn, v_t).transpose(1, 2)
    return out


def test_dense_equal_length():
    q = torch.randn(2, 512, 8, 64, dtype=torch.bfloat16, device="xpu")
    k = torch.randn(2, 512, 8, 64, dtype=torch.bfloat16, device="xpu")
    v = torch.randn(2, 512, 8, 64, dtype=torch.bfloat16, device="xpu")
    for causal in (False, True):
        out = xfa.flash_attn_func(q, k, v, causal=causal)
        ref = _manual_ref(q, k, v, causal=causal)
        max_err = (out - ref).abs().max().item()
        assert max_err < 0.05, f"dense causal={causal} max_err={max_err}"
    print("test_dense_equal_length passed")


def test_dense_gqa():
    q = torch.randn(2, 256, 8, 64, dtype=torch.bfloat16, device="xpu")
    k = torch.randn(2, 256, 2, 64, dtype=torch.bfloat16, device="xpu")
    v = torch.randn(2, 256, 2, 64, dtype=torch.bfloat16, device="xpu")
    out = xfa.flash_attn_func(q, k, v, causal=True)
    ref = _manual_ref(q, k, v, causal=True)
    max_err = (out - ref).abs().max().item()
    assert max_err < 0.05, f"gqa max_err={max_err}"
    print("test_dense_gqa passed")


def test_varlen():
    lens = [128, 256, 192]
    q_seqs = [torch.randn(l, 8, 64, dtype=torch.bfloat16, device="xpu") for l in lens]
    k_seqs = [torch.randn(l, 8, 64, dtype=torch.bfloat16, device="xpu") for l in lens]
    v_seqs = [torch.randn(l, 8, 64, dtype=torch.bfloat16, device="xpu") for l in lens]
    q = torch.cat(q_seqs, dim=0)
    k = torch.cat(k_seqs, dim=0)
    v = torch.cat(v_seqs, dim=0)
    cu = torch.tensor([0, *torch.cumsum(torch.tensor(lens), 0).tolist()], device="xpu", dtype=torch.int32)
    out = xfa.flash_attn_varlen_func(q, k, v, cu, cu, max(lens), max(lens), causal=True)
    # reference per sequence
    refs = []
    for qi, ki, vi in zip(q_seqs, k_seqs, v_seqs):
        refs.append(_manual_ref(qi.unsqueeze(0), ki.unsqueeze(0), vi.unsqueeze(0), causal=True).squeeze(0))
    ref = torch.cat(refs, dim=0)
    max_err = (out - ref).abs().max().item()
    assert max_err < 0.05, f"varlen max_err={max_err}"
    print("test_varlen passed")


def _bench_one(name, q, k, v, causal=True, n=20):
    for _ in range(3):
        _ = xfa.flash_attn_func(q, k, v, causal=causal)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        out = xfa.flash_attn_func(q, k, v, causal=causal)
    torch.xpu.synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) / n * 1000
    print(f"{name}: {ms:.2f} ms")


def benchmark():
    print("\n--- xpu_flash_attn benchmark on XPU ---")
    for (b, sq, sk, hq, hk, d) in [
        (1, 4096, 4096, 8, 8, 128),
        (1, 8192, 8192, 8, 8, 128),
        (2, 2048, 2048, 16, 16, 64),
    ]:
        q = torch.randn(b, sq, hq, d, dtype=torch.bfloat16, device="xpu")
        k = torch.randn(b, sk, hk, d, dtype=torch.bfloat16, device="xpu")
        v = torch.randn(b, sk, hk, d, dtype=torch.bfloat16, device="xpu")
        _bench_one(f"B={b} Sq={sq} Sk={sk} Hq={hq} Hk={hk} D={d}", q, k, v)


if __name__ == "__main__":
    test_dense_equal_length()
    test_dense_gqa()
    test_varlen()
    benchmark()
