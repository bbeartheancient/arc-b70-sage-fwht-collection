"""Core attention entry points for the XPU SageAttention drop-in.

The numerical contract is plain scaled-dot-product attention:

    out = softmax(scale * Q K^T + mask) V      scale = sm_scale

dispatching to ``torch.nn.functional.scaled_dot_product_attention``
(oneDNN SDPA on point-of-sale XPU). Layouts follow upstream SageAttention:
``"HND"`` = (batch, heads, seq, head_dim), ``"NHD"`` = (batch, seq,
heads, head_dim). ``smooth_k`` is accepted for API parity but is a no-op
here: it exists upstream to stabilize the INT8 QK^T path, which this
SDPA backend does not use.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F

_SMOOTH_WARNED = False


def _head_dim(q: torch.Tensor) -> int:
    return q.shape[-1]


def _to_bhsd(x: torch.Tensor, tensor_layout: str) -> torch.Tensor:
    """Return (batch, heads, seq, head_dim) per Sage layout conventions."""
    if tensor_layout == "NHD":
        return x.transpose(1, 2)
    if tensor_layout == "HND":
        return x
    raise ValueError(f"tensor_layout must be 'HND' or 'NHD', got {tensor_layout!r}")


def _from_bhsd(x: torch.Tensor, tensor_layout: str) -> torch.Tensor:
    if tensor_layout == "NHD":
        return x.transpose(1, 2)
    return x


def _resolve_scale(sm_scale, scale, q, is_causal) -> float:
    if sm_scale is not None:
        return float(sm_scale)
    if scale is not None:
        return float(scale)
    if is_causal:
        return 1.0 / math.sqrt(max(1, _head_dim(q)))
    return 1.0 / math.sqrt(max(1, _head_dim(q)))


def _lse(q: torch.Tensor, k: torch.Tensor, scale: float, is_causal: bool,
         attn_mask: torch.Tensor | None) -> torch.Tensor:
    """Reference log-sum-exp of the raw scores: (batch, heads, seq).

    Only computed when ``return_lse=True``; the SDPA kernel already
    produced the output tensor, this is the (rare) metadata path.
    """
    scores = q.float() @ k.float().transpose(-2, -1)
    scores = scores * scale
    if is_causal:
        s = scores.shape[-1]
        tri = torch.ones(s, s, device=scores.device, dtype=torch.bool).tril()
        scores.masked_fill_(~tri, -float("inf"))
    if attn_mask is not None:
        scores = scores + attn_mask.float()
    return torch.logsumexp(scores, dim=-1)


def _warn_smooth() -> None:
    global _SMOOTH_WARNED
    if not _SMOOTH_WARNED:
        warnings.warn(
            "sageattention: smooth_k is accepted but ignored on XPU "
            "(no INT8 QK^T path; SDPA backend does not need it)",
            stacklevel=2,
        )
        _SMOOTH_WARNED = True


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    smooth_k: bool = True,
    sm_scale: float | None = None,
    attn_mask: torch.Tensor | None = None,
    return_lse: bool = False,
    dropout_p: float = 0.0,
    scale: float | None = None,
    dtype_out: torch.dtype | None = None,
    head_first: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """SageAttention drop-in for XPU (SDPA / oneDNN backend)."""
    out_dtype = dtype_out or q.dtype
    if smooth_k:
        _warn_smooth()

    q_ = _to_bhsd(q, tensor_layout)
    k_ = _to_bhsd(k, tensor_layout)
    v_ = _to_bhsd(v, tensor_layout)
    if q_.dtype != k_.dtype:
        k_ = k_.to(q_.dtype)
    if q_.dtype != v_.dtype:
        v_ = v_.to(q_.dtype)

    sm = _resolve_scale(sm_scale, scale, q, is_causal)
    if attn_mask is not None and attn_mask.dtype.is_floating_point and attn_mask.dtype != q_.dtype:
        attn_mask = attn_mask.to(q_.dtype)
    out = F.scaled_dot_product_attention(
        q_,
        k_,
        v_,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=float(sm),
    )
    if out.dtype != out_dtype:
        out = out.to(out_dtype)

    if return_lse:
        lse = _lse(q_, k_, float(sm), is_causal, attn_mask)
        return _from_bhsd(out, tensor_layout), lse
    return _from_bhsd(out, tensor_layout)


def sageattn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    smooth_k: bool = True,
    sm_scale: float | None = None,
    scale: float | None = None,
    dtype_out: torch.dtype | None = None,
    **kwargs,
) -> torch.Tensor:
    """Variable-sequence-length attention.

    Inputs are packed along the sequence axis: Q is (total_q, heads,
    head_dim), K/V are (total_k, heads, head_dim); ``cu_seqlens_*`` are
    int32 prefix sums of length batch+1. Correct-by-construction loop
    over the packed segments (day-one drop-in; a fused varlen kernel is
    not shipped on XPU).
    """
    if smooth_k:
        _warn_smooth()
    cu_q = cu_seqlens_q.long().tolist()
    cu_k = cu_seqlens_k.long().tolist()
    batch = len(cu_q) - 1
    out_dtype = dtype_out or q.dtype
    outs = []
    for i in range(batch):
        qs = q[cu_q[i]:cu_q[i + 1]].transpose(0, 1).unsqueeze(0)
        ks = k[cu_k[i]:cu_k[i + 1]].transpose(0, 1).unsqueeze(0)
        vs = v[cu_k[i]:cu_k[i + 1]].transpose(0, 1).unsqueeze(0)
        o = sageattn(
            qs,
            ks,
            vs,
            tensor_layout="HND",
            is_causal=is_causal,
            smooth_k=False,
            sm_scale=sm_scale,
            scale=scale,
            dtype_out=out_dtype,
        )
        outs.append(o.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0) if outs else torch.empty_like(q)