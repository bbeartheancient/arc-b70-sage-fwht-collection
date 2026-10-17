"""Qwen 3.8 27B widths. Single source for host tools and comments.

Pad rules from docs/qwen38-shapes.md: no H(5), no H(17). Walsh the
power-of-two factor or pad the whole axis to the next 2^k.
"""

from __future__ import annotations

from dataclasses import dataclass


def next_pow2(n: int) -> int:
    if n < 1:
        raise ValueError(n)
    p = 1
    while p < n:
        p <<= 1
    return p


def walsh_pad(n: int) -> int:
    """Length we actually run FWHT at. Identity when n is already 2^k."""
    return next_pow2(n)


@dataclass(frozen=True)
class LinearAttn:
    key_head_dim: int
    value_head_dim: int
    num_key_heads: int
    num_value_heads: int
    conv_kernel_dim: int


@dataclass(frozen=True)
class GatedAttn:
    head_dim: int
    num_heads: int
    num_kv_heads: int
    rope_dims: int


@dataclass(frozen=True)
class Vision:
    hidden_size: int
    intermediate_size: int
    num_heads: int
    out_hidden_size: int
    depth: int


@dataclass(frozen=True)
class Qwen38:
    name: str
    params_b: float
    hidden_size: int
    intermediate_size: int
    num_layers: int
    vocab_size: int
    full_attention_interval: int
    gated_attn: GatedAttn
    linear_attn: LinearAttn
    vision: Vision
    dtype: str

    @property
    def hidden_fwht(self) -> int:
        return walsh_pad(self.hidden_size)

    @property
    def ffn_fwht(self) -> int:
        return walsh_pad(self.intermediate_size)

    @property
    def attn_fwht(self) -> int:
        return self.gated_attn.head_dim

    @property
    def linear_fwht(self) -> int:
        return self.linear_attn.key_head_dim


QWEN38_27B = Qwen38(
    name="Qwen3.8-27B",
    params_b=27.0,
    hidden_size=5120,
    intermediate_size=17408,
    num_layers=64,
    vocab_size=248320,
    full_attention_interval=4,
    gated_attn=GatedAttn(head_dim=256, num_heads=24, num_kv_heads=4, rope_dims=64),
    linear_attn=LinearAttn(
        key_head_dim=128,
        value_head_dim=128,
        num_key_heads=16,
        num_value_heads=48,
        conv_kernel_dim=4,
    ),
    vision=Vision(
        hidden_size=1152,
        intermediate_size=4304,
        num_heads=16,
        out_hidden_size=5120,
        depth=27,
    ),
    dtype="bfloat16",
)
