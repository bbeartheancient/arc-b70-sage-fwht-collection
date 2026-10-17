# SPDX-License-Identifier: Apache-2.0
"""Bicameral n-gram embedding (PLE) for Qwen3.8-Flash-Next.

Tech report §2.3 + BICAMERAL §6.3: a single PLE layer at config
``ple_layer_ids=[2]`` (1-indexed → 0-indexed layer 1).  The 128 n-gram
shards (~102 GB, each ``[2500012, 160]`` bf16) stay on NVMe / host;
they are NOT a SYCL kernel.

v1: forward is identity on the 4-wide residual.  Small PLE tensors
(key/value proj, grouped norms, dilated conv, hash metadata) still
load so the parameter set is honest.  Shards are skipped by the
sparse loader.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from vllm.model_executor.layers.linear import ReplicatedLinear

from afi.vllm_qwen4exp.config import Qwen4ExpConfig
from afi.vllm_qwen4exp.layout import ple_zero_index
from afi.vllm_qwen4exp.norm import Qwen4ExpGroupedRMSNorm

logger = logging.getLogger(__name__)


class _PLEMeta(nn.Module):
    """Hash-table metadata (no n-gram shards). Checkpoint has no .weight suffix."""

    def __init__(self, ngram_size: int, n_heads: int):
        super().__init__()
        self.layer_multipliers = nn.Parameter(
            torch.zeros(ngram_size, dtype=torch.int64), requires_grad=False
        )
        self.ngram_heads_offsets = nn.Parameter(
            torch.zeros(n_heads, dtype=torch.int64), requires_grad=False
        )
        self.ngram_heads_vocab_sizes = nn.Parameter(
            torch.zeros(n_heads, dtype=torch.int64), requires_grad=False
        )


class BicameralNgramEmbedding(nn.Module):
    """PLE module: key_proj, value_proj; n-gram shards via cache.

    Forward contract (matches the framework's intent for layer 1):
      in:  hidden_states (num_tokens, hidden_size)
      out: hidden_states + ngram_contribution (same shape)

    v1 returns ``hidden_states`` unchanged; the projections are still
    allocated so the model's parameter count is honest.
    """

    def __init__(
        self,
        config: Qwen4ExpConfig,
        prefix: str = "",
        shard_loader=None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.ple_embed_dim = config.ple_embed_dim
        self.nr = config.hc_count
        hc = self.nr * config.hidden_size
        # 0-indexed owner (config is 1-indexed).
        zeros = ple_zero_index(config.ple_layer_ids)
        self.ple_layer_id = zeros[0] if zeros else 1

        # Checkpoint: key_proj [10240, 2560] = Linear(ple_embed_dim, nr*d).
        self.key_proj = ReplicatedLinear(
            config.ple_embed_dim,
            hc,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.key_proj",
        )
        self.value_proj = ReplicatedLinear(
            config.ple_embed_dim,
            config.hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.value_proj",
        )
        self.norm_key = Qwen4ExpGroupedRMSNorm(
            hc, group_size=config.hidden_size, eps=config.rms_norm_eps
        )
        self.norm_query = Qwen4ExpGroupedRMSNorm(
            hc, group_size=config.hidden_size, eps=config.rms_norm_eps
        )
        self.norm_conv = Qwen4ExpGroupedRMSNorm(
            hc, group_size=config.hidden_size, eps=config.rms_norm_eps
        )
        k = config.ple_conv_kernel_size
        self.conv1d = nn.Conv1d(
            hc, hc, kernel_size=k, groups=hc, dilation=config.ngram_size, bias=False
        )
        n_heads = (config.ngram_size - 1) * config.heads_per_ngram
        self.ple_embedding = _PLEMeta(config.ngram_size, n_heads)

        # 128 n-gram shards stay on NVMe (BICAMERAL §6.3).  v1 identity.
        # v2 (lazy n-gram): a LazyNgramLoader mmap's shards on demand.
        # Caller may pass a pre-built loader (singleton), else we leave
        # shard_cache = None and forward returns hidden unchanged.
        self.shard_cache = shard_loader

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """v1 identity.  v2: hidden_states += value_proj(igram(token_ids))."""
        if self.shard_cache is None or token_ids is None:
            return hidden_states
        try:
            ids_np = (
                token_ids.detach().to("cpu", non_blocking=True).numpy().astype("int64")
            )
            trigrams_np = self.shard_cache.lookup(ids_np)
            trigrams = torch.from_numpy(trigrams_np).to(
                hidden_states.device, non_blocking=True
            )
            contribution = self.value_proj(trigrams)[0]
            return hidden_states + contribution
        except Exception as exc:  # noqa: BLE001
            logger.warning("BicameralNgram lookup failed: %s", exc)
            return hidden_states


__all__ = ["BicameralNgramEmbedding"]
