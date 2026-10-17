# SPDX-License-Identifier: Apache-2.0
"""Grouped Gemma-style RMSNorm for Qwen4-Exp Gated Residual.

Tech report Eq. (30): each of the ``nr`` residual branches is RMS-normalized
independently (group size = hidden_size) with its own gain γ_i.

Zero-centered: ``y = x_hat * (1 + weight)``, same convention as GemmaRMSNorm
/ Qwen4ExpTextRMSNorm in transformers.
"""

from __future__ import annotations

import torch
from torch import nn


class Qwen4ExpGroupedRMSNorm(nn.Module):
    """RMSNorm over the last ``group_size`` dims of a ``hidden_size`` vector."""

    def __init__(self, hidden_size: int, group_size: int, eps: float = 1e-6):
        super().__init__()
        if hidden_size % group_size != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size})"
            )
        self.hidden_size = hidden_size
        self.group_size = group_size
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape
        grouped = x.reshape(*orig[:-1], -1, self.group_size)
        grouped_f = grouped.float()
        var = grouped_f.pow(2).mean(-1, keepdim=True)
        normed = grouped_f * torch.rsqrt(var + self.eps)
        scale = (1.0 + self.weight.float()).view(-1, self.group_size)
        return (normed * scale).to(x.dtype).reshape(orig)


__all__ = ["Qwen4ExpGroupedRMSNorm"]
