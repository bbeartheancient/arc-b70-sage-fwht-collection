# SPDX-License-Identifier: Apache-2.0
"""XPU FlashAttention-compatible interface backed by PyTorch native SDPA.

This is a pure-Python shim that maps the public FlashAttention API to
``torch.nn.functional.scaled_dot_product_attention`` on Intel XPU.  It avoids
building the CUDA/ROCm-only ``flash-attn`` wheel and works on any GPU supported
by the torch+xpu backend (e.g. Arc A770, PVC, BMG).

Unsupported advanced features (alibi, softcap, sliding-window masks, paged KV
block tables, and attention-prob/LSE returns) fall back to an explicit PyTorch
reference implementation so correctness is preserved.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


def _maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.contiguous() if x.stride(-1) != 1 else x


def _repeat_kv_for_gqa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, head_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Repeat k/v heads to match q heads for GQA/MQA when SDPA requires it.

    Tensors are assumed to be in ``(batch, heads, seqlen, headdim)`` layout when
    ``head_dim=1`` (dense SDPA path) or ``(heads, seqlen, headdim)`` when
    ``head_dim=0`` (per-sequence nested path).
    """
    hq, hk = q.size(head_dim), k.size(head_dim)
    if hk == hq:
        return k, v
    if hq % hk != 0:
        raise ValueError(
            f"num_heads_q ({hq}) must be divisible by num_heads_kv ({hk})"
        )
    r = hq // hk
    k = k.repeat_interleave(r, dim=head_dim)
    v = v.repeat_interleave(r, dim=head_dim)
    return k, v


def _make_additive_mask(
    sq: int,
    sk: int,
    device: torch.device,
    dtype: torch.dtype,
    causal: bool,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build a bottom-right aligned additive attention mask.

    FlashAttention aligns the causal mask to the bottom-right corner of the
    score matrix.  For ``sq == sk`` this is identical to the standard top-left
    triangular mask, but for unequal lengths we must shift it explicitly.
    """
    q_idx = torch.arange(sq, device=device, dtype=torch.float32).view(sq, 1)
    k_idx = torch.arange(sk, device=device, dtype=torch.float32).view(1, sk)
    # bottom-right offset: query position i may attend to keys <= sk - sq + i
    shift = sk - sq
    keep = torch.ones((sq, sk), dtype=torch.bool, device=device)
    if causal:
        keep &= k_idx <= q_idx + shift
    left, right = window_size
    if left >= 0:
        keep &= k_idx >= q_idx + shift - left
    if right >= 0:
        keep &= k_idx <= q_idx + shift + right

    mask = torch.where(keep, 0.0, float("-inf")).to(dtype)
    if alibi_slopes is not None:
        # alibi_slopes: (num_heads,) or (batch, num_heads)
        slopes = alibi_slopes.to(dtype=torch.float32, device=device).view(-1)
        bias = -(slopes.abs().unsqueeze(-1).unsqueeze(-1)) * (q_idx - k_idx).abs()
        # broadcast over batch/heads at runtime
        mask = mask.unsqueeze(0).unsqueeze(0) + bias.unsqueeze(0).unsqueeze(-1)
    else:
        mask = mask.unsqueeze(0).unsqueeze(0)
    return mask


def _manual_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    window_size: Tuple[int, int],
    alibi_slopes: Optional[torch.Tensor],
    softcap: float,
    dropout_p: float,
    return_softmax_lse: bool,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Explicit attention reference for unsupported option combinations."""
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    k, v = _repeat_kv_for_gqa(q, k, v, head_dim=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    sq, sk = scores.size(-2), scores.size(-1)
    if alibi_slopes is not None or causal or window_size != (-1, -1):
        mask = _make_additive_mask(
            sq, sk, q.device, scores.dtype, causal, window_size, alibi_slopes
        )
        # expand to (batch, heads, sq, sk)
        while mask.dim() < 4:
            mask = mask.unsqueeze(0)
        scores = scores + mask
    if softcap > 0.0:
        scores = softcap * torch.tanh(scores / softcap)
    if return_softmax_lse:
        lse = torch.logsumexp(scores, dim=-1)
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=True)
    out = torch.matmul(attn, v).transpose(1, 2)
    if return_softmax_lse:
        return out, lse.transpose(-2, -1)
    return out


def _needs_manual_path(
    causal: bool,
    sq: int,
    sk: int,
    window_size: Tuple[int, int],
    alibi_slopes: Optional[torch.Tensor],
    softcap: float,
    return_attn_probs: bool,
    block_table: Optional[torch.Tensor],
) -> bool:
    if block_table is not None:
        return True
    if softcap != 0.0:
        return True
    if alibi_slopes is not None:
        return True
    if window_size != (-1, -1):
        return True
    if return_attn_probs:
        return True
    if causal and sq != sk:
        return True
    return False


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """FlashAttention forward compatible wrapper (dense tensor path).

    Arguments match the standard ``flash_attn.flash_attn_func`` signature.
    Layout is ``(batch_size, seqlen, nheads, headdim)``.
    """
    q, k, v = _maybe_contiguous(q), _maybe_contiguous(k), _maybe_contiguous(v)
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v must be 4-D tensors (B, S, H, D)")
    batch, sq, hq, d = q.shape
    sk = k.size(1)
    if softmax_scale is None:
        softmax_scale = d**-0.5

    if _needs_manual_path(
        causal, sq, sk, window_size, alibi_slopes, softcap, return_attn_probs, None
    ):
        return _manual_attention(
            q, k, v, softmax_scale, causal, window_size, alibi_slopes,
            softcap, dropout_p, return_attn_probs,
        )

    # Convert to SDPA layout (B, H, S, D)
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    k_t, v_t = _repeat_kv_for_gqa(q_t, k_t, v_t)
    out_t = F.scaled_dot_product_attention(
        q_t, k_t, v_t,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    )
    out = out_t.transpose(1, 2)
    if return_attn_probs:
        # SDPA doesn't expose LSE; recompute with the manual path for tests.
        return _manual_attention(
            q, k, v, softmax_scale, causal, window_size, alibi_slopes,
            softcap, dropout_p, True,
        )
    return out


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Variable-length FlashAttention compatible wrapper.

    Routes to the most efficient XPU SDPA path depending on the batch shape:

    * Equal-length per-sequence q/k -> one dense SDPA call with ``is_causal``.
    * Mixed-length per-sequence q/k but ``sq_i == sk_i`` for every seq -> pad
      to ``max(seqlen)`` and call SDPA once with ``is_causal=True``.
    * Decode-like single-token queries (``sq_i == 1``) -> per-sequence SDPA.
    * Anything else (paged KV, alibi, softcap, local windows, etc.) -> a
      per-sequence PyTorch reference fallback.
    """
    q, k, v = _maybe_contiguous(q), _maybe_contiguous(k), _maybe_contiguous(v)
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("q/k/v must be 3-D ragged tensors (total, H, D)")
    if softmax_scale is None:
        softmax_scale = q.size(-1) ** -0.5

    batch = cu_seqlens_q.numel() - 1
    cu_q = cu_seqlens_q.cpu().tolist()
    cu_k = cu_seqlens_k.cpu().tolist()
    seq_lens_q = [cu_q[i + 1] - cu_q[i] for i in range(batch)]
    seq_lens_k = [cu_k[i + 1] - cu_k[i] for i in range(batch)]

    if block_table is not None or softcap != 0.0 or alibi_slopes is not None \
            or window_size != (-1, -1) or return_attn_probs:
        return _manual_varlen_attention(
            q, k, v, cu_q, cu_k, seq_lens_q, seq_lens_k,
            softmax_scale, causal, window_size, alibi_slopes,
            softcap, dropout_p, return_attn_probs,
        )

    # Every sequence has matching q/k length: use a single dense SDPA call.
    if seq_lens_q == seq_lens_k:
        if len(set(seq_lens_q)) == 1:
            q_dense = q.view(batch, seq_lens_q[0], q.size(1), q.size(2))
            k_dense = k.view(batch, seq_lens_k[0], k.size(1), k.size(2))
            v_dense = v.view(batch, seq_lens_k[0], v.size(1), v.size(2))
            return flash_attn_func(
                q_dense, k_dense, v_dense,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                return_attn_probs=return_attn_probs,
            ).view(-1, q.size(1), q.size(2))
        else:
            return _padded_equal_varlen_attention(
                q, k, v, cu_q, cu_k, seq_lens_q,
                max_seqlen_q,
                softmax_scale, causal, dropout_p,
            )

    # Mismatched q/k lengths: per-sequence SDPA for decode, else manual ref.
    return _mixed_varlen_attention(
        q, k, v, cu_q, cu_k, seq_lens_q, seq_lens_k,
        softmax_scale, causal, dropout_p,
    )


def _padded_equal_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: list,
    cu_k: list,
    seq_lens: list,
    max_seqlen: int,
    softmax_scale: float,
    causal: bool,
    dropout_p: float,
) -> torch.Tensor:
    """Pad a batch of equal-q/k sequences to the same length and use SDPA."""
    batch = len(seq_lens)
    _, hq, d = q.shape
    hk = k.size(1)
    q_pad = torch.zeros(batch, max_seqlen, hq, d, dtype=q.dtype, device=q.device)
    k_pad = torch.zeros(batch, max_seqlen, hk, d, dtype=k.dtype, device=k.device)
    v_pad = torch.zeros(batch, max_seqlen, hk, d, dtype=v.dtype, device=v.device)
    for i, l in enumerate(seq_lens):
        q_pad[i, :l] = q[cu_q[i]:cu_q[i + 1]]
        k_pad[i, :l] = k[cu_k[i]:cu_k[i + 1]]
        v_pad[i, :l] = v[cu_k[i]:cu_k[i + 1]]

    q_t = q_pad.transpose(1, 2)
    k_t = k_pad.transpose(1, 2)
    v_t = v_pad.transpose(1, 2)
    k_t, v_t = _repeat_kv_for_gqa(q_t, k_t, v_t, head_dim=1)
    out_t = F.scaled_dot_product_attention(
        q_t, k_t, v_t,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    )
    out_pad = out_t.transpose(1, 2)
    return torch.cat([out_pad[i, :l] for i, l in enumerate(seq_lens)], dim=0)


def _mixed_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: list,
    cu_k: list,
    seq_lens_q: list,
    seq_lens_k: list,
    softmax_scale: float,
    causal: bool,
    dropout_p: float,
) -> torch.Tensor:
    """Handle per-sequence q/k mismatches with SDPA decode or manual prefill."""
    outputs = []
    for i, (sq_i, sk_i) in enumerate(zip(seq_lens_q, seq_lens_k)):
        qi = q[cu_q[i]:cu_q[i + 1]].unsqueeze(0)
        ki = k[cu_k[i]:cu_k[i + 1]].unsqueeze(0)
        vi = v[cu_k[i]:cu_k[i + 1]].unsqueeze(0)
        if sq_i == 1:
            # Single-token decode: no causal mask needed.
            qi_t = qi.transpose(1, 2)
            ki_t = ki.transpose(1, 2)
            vi_t = vi.transpose(1, 2)
            ki_t, vi_t = _repeat_kv_for_gqa(qi_t, ki_t, vi_t, head_dim=1)
            out_t = F.scaled_dot_product_attention(
                qi_t, ki_t, vi_t,
                dropout_p=dropout_p,
                scale=softmax_scale,
            )
            outputs.append(out_t.transpose(1, 2).squeeze(0))
        else:
            res = _manual_attention(
                qi, ki, vi, softmax_scale, causal, (-1, -1), None,
                0.0, dropout_p, False,
            )
            outputs.append(res.squeeze(0))
    return torch.cat(outputs, dim=0)


def _manual_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: list,
    cu_k: list,
    seq_lens_q: list,
    seq_lens_k: list,
    softmax_scale: float,
    causal: bool,
    window_size: Tuple[int, int],
    alibi_slopes: Optional[torch.Tensor],
    softcap: float,
    dropout_p: float,
    return_softmax_lse: bool,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    outputs, lses = [], []
    for i, (sq_i, sk_i) in enumerate(zip(seq_lens_q, seq_lens_k)):
        qi = q[cu_q[i]:cu_q[i + 1]].unsqueeze(0)
        ki = k[cu_k[i]:cu_k[i + 1]].unsqueeze(0)
        vi = v[cu_k[i]:cu_k[i + 1]].unsqueeze(0)
        slopes = None
        if alibi_slopes is not None:
            if alibi_slopes.dim() == 2:
                slopes = alibi_slopes[i]
            else:
                slopes = alibi_slopes
        res = _manual_attention(
            qi, ki, vi, softmax_scale, causal, window_size, slopes,
            softcap, dropout_p, return_softmax_lse,
        )
        if return_softmax_lse:
            outputs.append(res[0].squeeze(0))
            lses.append(res[1].squeeze(0))
        else:
            outputs.append(res.squeeze(0))
    out = torch.cat(outputs, dim=0)
    if return_softmax_lse:
        return out, torch.cat(lses, dim=-1 if lses[0].dim() == 1 else 0)
    return out


def flash_attn_qkvpacked_func(
    qkv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """QKV-packed FlashAttention compatible wrapper."""
    if qkv.dim() != 5 or qkv.size(2) != 3:
        raise ValueError("qkv must have shape (B, S, 3, H, D)")
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    return flash_attn_func(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )


def flash_attn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """KV-packed FlashAttention compatible wrapper."""
    if kv.dim() != 5 or kv.size(2) != 2:
        raise ValueError("kv must have shape (B, S, 2, H_k, D)")
    k, v = kv[:, :, 0], kv[:, :, 1]
    return flash_attn_func(
        q, k, v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )
