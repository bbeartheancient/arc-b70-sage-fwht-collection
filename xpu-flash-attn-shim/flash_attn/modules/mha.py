"""Minimal XPU-compatible replacement for flash_attn.modules.mha."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashSelfAttention(nn.Module):
    """SDPA-backed self-attention compatible with flash_attn.modules.mha.FlashSelfAttention."""

    def __init__(self, causal: bool = False, softmax_scale: float = None, attention_dropout: float = 0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = attention_dropout

    def forward(self, qkv: torch.Tensor, causal: bool = None, key_padding_mask: torch.Tensor = None):
        """qkv shape: (batch, seqlen, 3, nheads, headdim)."""
        if causal is None:
            causal = self.causal
        if qkv.dim() != 5 or qkv.size(2) != 3:
            raise ValueError("qkv must have shape (B, S, 3, H, D)")
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        scale = self.softmax_scale or q.size(-1) ** -0.5
        out = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            dropout_p=self.dropout,
            is_causal=causal,
            scale=scale,
            attn_mask=None if key_padding_mask is None else ~key_padding_mask.unsqueeze(1).unsqueeze(1),
        )
        return out.transpose(1, 2)


class FlashCrossAttention(nn.Module):
    """SDPA-backed cross-attention compatible with flash_attn.modules.mha.FlashCrossAttention."""

    def __init__(self, causal: bool = False, softmax_scale: float = None, attention_dropout: float = 0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = attention_dropout

    def forward(self, q: torch.Tensor, kv: torch.Tensor, causal: bool = None, key_padding_mask: torch.Tensor = None):
        """q shape: (B, Sq, H, D); kv shape: (B, Sk, 2, H, D)."""
        if causal is None:
            causal = self.causal
        k, v = kv[:, :, 0], kv[:, :, 1]
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        scale = self.softmax_scale or q.size(-1) ** -0.5
        out = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            dropout_p=self.dropout,
            is_causal=causal,
            scale=scale,
            attn_mask=None if key_padding_mask is None else ~key_padding_mask.unsqueeze(1).unsqueeze(1),
        )
        return out.transpose(1, 2)
