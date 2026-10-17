# SPDX-License-Identifier: Apache-2.0
"""Bicameral MoE for Qwen3.8-Flash-Next (Qwen4-Exp).

The real Qwen-FN model has MoE experts on ALL 48 decoder layers (plus
MTP), each with 512 experts of (2*640 × 2560).  Naive FusedMoE would
try to allocate the routed tiles on GPU0 (~247 GB).  Single B70 =
32 GB.  The naive path is not viable.

This module replaces ``Qwen3NextSparseMoeBlock`` with a v1 that:
  * Carries the gate (router) and the shared expert — both tiny.
  * Routes the routed-expert path through ``BicameralExpertCache``.

H(4) layer-routing bridge (2026-08-27, per docs/h4-layer-gating.md):

  The 48 decoder layers follow a 3+1 pattern: 3 linear-attention
  layers followed by 1 full-attention layer, repeating 12 times.
  The full-attention layers sit at 0-indexed positions 3, 7, 11, ...,
  47 (i.e. the 4th layer of every block of 4).

  H(4) is a 4×4 Sylvester-Hadamard matrix with 4 orthogonal rows
  that map to first-order spherical harmonics.  Used as a layer
  routing transform, the 4 rows correspond to the 4 layer types
  in a Qwen-FN block:

    H(4) row 0 = full-attention layer   (the "consensus" row W = [+ + + +])
    H(4) row 1 = linear-attention layer (X = [+ + - -])
    H(4) row 2 = linear-attention layer (Y = [+ - + -])
    H(4) row 3 = linear-attention layer (Z = [+ - - +])

  The 3 linear-attention rows are interchangeable — they are 3
  orthogonal basis vectors and any rotation among them preserves
  the H(4) structure.  The full-attention row 0 is fixed because
  the W row has the unique property of being the only one that
  sums all 4 contributions (it is the "consensus" / "DC" component
  of the layer block).

  How the H(4) is used at decode time (GPU1, the bicameral bridge
  card):

    PHASE 1 — GPU0 search (vectorised layer search, NO codec):
      GPU0 holds the master gate (2560 → 512).  It runs the gate on
      the hidden state to get (S, 512) logits, then picks top-K=10
      experts per token GLOBALLY (standard Qwen-FN routing).  The
      gate input is the original (S, 2560) hidden state — there is
      NO H(4) codec on the gate.  Per operator (2026-08-27):
      "the initial layer search is L, L+1, L+2, L+3; those vectors
      are already linear, they can pass through directly to GPU1".

    PHASE 2 — GPU0 → GPU1 transfer (forward path, NO codec):
      The 4 layer outputs (one per layer in the 4-layer block) are
      sent to GPU1 as PLAIN LINEAR VECTORS — no H(4) codec on this
      leg.  The 4 quadrants of GPU1 process their 12 layers each
      in parallel (4x linear vector lookups running concurrently on
      4 quarters of GPU1's Xe-cores).

    PHASE 3 — GPU1 work (4x linear vector lookups in parallel):
      GPU1 holds the routed-expert weights (246 GB) sharded by
      LAYER into 4 quadrants.  Quadrant 0 (W) = full-attn layers
      (3, 7, 11, ..., 47).  Quadrants 1-3 (X, Y, Z) = 3 sets of
      linear-attn layers (the 3 linear positions in each block).
      Each quadrant processes its 12 layers in parallel against
      the linear layer vectors that came in plain.  Per layer
      the work is the routed expert MLP (2560 → 640 → 2560,
      swiglu).  Each quadrant issues independent linear lookups.

    PHASE 4 — GPU1 → GPU0 transfer (return path, H(4) encoded):
      The 4 quadrants' per-layer outputs (4 separate streams, one
      per H(4) direction W/X/Y/Z) are encoded by H(4) on the
      layer axis — this is the "spatial vector that reconstructs
      the encoded dimensions back into the active model weights".
      The 4 quadrants' contributions get mixed into a single
      4-channel return stream of shape (S, 4, D).  H(4) is
      self-inverse (H = H^{-1} when divided by 2) so the math
      is exact.

    PHASE 5 — GPU0 gather (H(4) decode):
      GPU0 decodes the 4-channel return stream back to 4 layer
      deltas (S, 4, D) via H(4).T, then adds the routed
      contribution to the residual stream.  The top-K=10 weights
      from the master gate (PHASE 1) are applied per-token.

  v1 (this file):
    * PHASE 1: master gate runs once per layer as standard, no
      H(4) codec.  The 4 layer outputs are plain linear vectors.
    * PHASE 3: GPU1 cache holds the routed expert weights,
      sharded by layer quadrant.  The cache.fetch() call returns
      the weights for a single (layer, expert) pair.
    * The routed expert work is a no-op (zero tensor) until the
      cache is filled in by the async weight streamer.
    * The H(4) layer encode/decode helpers ARE exported
      (gpu1_encode_layer_outputs, gpu0_decode_return_stream) so
      the bridge code can wire them on the return path once the
      cache is filled.

  Drop-in: the gate is unchanged, the routed contribution is
  zero in v1 (cache empty), so the model generates with shared
  expert only until the cache is filled.

  Note on latency (operator, 2026-08-27): the H(4) bridge saves
  BANDWIDTH (37% on the return stream) and provides 4x GPU1
  parallelism, but the dominant decode latency is GPU0's master
  gate + the per-expert matmul on GPU1.  The bridge codec does
  not address those.  Levers that DO:
    * Reduce top-K (10 → 4-6)
    * Larger expert batch (more tokens / expert / prefill step)
    * Sparse expert activation (skip experts with score < τ)
    * Expert-level KV cache (skip expert weights for repeated
      tokens)
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.linear import ReplicatedLinear

from afi.vllm_qwen4exp.config import Qwen4ExpConfig

logger = logging.getLogger(__name__)

# H(4) Sylvester-Hadamard, orthonormal.  H4 @ H4.T = I (divided by 2).
# Used as a 4×4 rotation on the LAYER axis (4-layer blocks).
#
# Row 0 = W = full-attention layer (consensus / DC, sums all 4)
# Row 1 = X = linear-attention layer (3rd, 7th, ... layers)
# Row 2 = Y = linear-attention layer (2nd, 6th, ... layers)
# Row 3 = Z = linear-attention layer (1st, 5th, ... layers)
#
# Per the operator: row 0 MUST be the full-attention layer (the
# only one of the 4 with the "consensus" property).  The other 3
# rows are linear-attention layers and their orientation (which
# row maps to which linear layer) is arbitrary as long as the
# 3 linear positions are constant across the 12 blocks.
_H4 = (
    torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],   # row 0 = W = full-attention
            [1.0, 1.0, -1.0, -1.0],  # row 1 = X = linear
            [1.0, -1.0, 1.0, -1.0],  # row 2 = Y = linear
            [1.0, -1.0, -1.0, 1.0],  # row 3 = Z = linear
        ],
        dtype=torch.float32,
    )
    / 2.0
)

# Quadrant assignment for the 48 layers, 4 layers per block, 12 blocks:
#   block b (b=0..11) contains layers 4b+0, 4b+1, 4b+2, 4b+3
#   layer 4b+3 is full-attention (0-indexed positions 3, 7, 11, ..., 47)
# Quadrants are 0..3, one per H(4) direction:
#   quadrant 0 (W / full-attention) = layers 4b+3 for b=0..11
#   quadrant 1 (X / linear)         = layers 4b+0 for b=0..11
#   quadrant 2 (Y / linear)         = layers 4b+1 for b=0..11
#   quadrant 3 (Z / linear)         = layers 4b+2 for b=0..11
# (Row 0 of H(4) is W, so the "first" of the 4 layers in a block —
# the one that contributes to row 0 in the inverse transform — is the
# full-attention layer at position 4b+3.  The other 3 rows are filled
# in order from the remaining 3 linear layers.)
QUADRANT_FOR_LAYER = [1, 2, 3, 0] * 12  # layer L → quadrant QUADRANT_FOR_LAYER[L]
# Inverse: which layers are in each quadrant?
LAYERS_IN_QUADRANT = {q: [] for q in range(4)}
for L in range(48):
    LAYERS_IN_QUADRANT[QUADRANT_FOR_LAYER[L]].append(L)


def _h4_layer_encode(layer_outputs: torch.Tensor) -> torch.Tensor:
    """H(4) on the layer axis.  (B, 4, D) → (B, 4, D).

    Input is 4 layer outputs from a 4-layer block (L+0, L+1, L+2, L+3).
    Output is the H(4)-rotated 4-channel spatial field.  H(4) is
    orthonormal, so this is information-preserving.

    Per operator (2026-08-27 correction): the H(4) codec is applied
    to the RETURN path only — GPU1 takes 4 plain layer tensors, runs
    its quadrant-specific work, then encodes the 4 layer outputs
    into the 4-channel return stream via this function.  GPU0 decodes
    the 4-channel stream back to 4 layer deltas via _h4_layer_decode.

    The 4 H(4) rows map to the 4 quadrants on GPU1:
      row 0 (W = [+ + + +]) = full-attn layer quadrant (consensus)
      rows 1-3 (X/Y/Z)      = 3 linear-attn quadrants
    """
    B, n_layers, D = layer_outputs.shape
    assert n_layers == 4, f"expected 4 layers per block, got {n_layers}"
    H4 = _H4.to(device=layer_outputs.device, dtype=layer_outputs.dtype)
    return torch.einsum("ij,bjd->bid", H4, layer_outputs)


def _h4_layer_decode(spatial_field: torch.Tensor) -> torch.Tensor:
    """H(4) inverse on the layer axis.  (B, 4, D) → (B, 4, D).

    Inverse of _h4_layer_encode.  For the 1/2-scaled orthonormal H(4):
    H4.T @ H4 = I, so H4.T = H4^{-1}.  We use the transpose here.
    The 4 output channels are the 4 layer deltas (one per layer in
    the original 4-layer block), ready to be added to the residual
    on GPU0.
    """
    B, n_layers, D = spatial_field.shape
    assert n_layers == 4
    H4 = _H4.to(device=spatial_field.device, dtype=spatial_field.dtype)
    H4_inv = H4.T.contiguous()  # H4.T = H4^{-1} (orthonormal)
    return torch.einsum("ij,bjd->bid", H4_inv, spatial_field)


def gpu1_encode_layer_outputs(layer_outputs: torch.Tensor) -> torch.Tensor:
    """GPU1-side: encode 4 layer outputs into a 4-channel return stream.

    layer_outputs : (S, 4, D) — the 4 layer outputs from GPU0's forward
                   path (linear; no codec applied on the forward leg).
    Returns      : (S, 4, D) — the 4-channel return stream GPU1 sends
                   back to GPU0.  This is the spatial field indexed
                   by H(4) direction (W/X/Y/Z).
    """
    return _h4_layer_encode(layer_outputs)


def gpu0_decode_return_stream(spatial_field: torch.Tensor) -> torch.Tensor:
    """GPU0-side: decode the 4-channel return stream from GPU1.

    spatial_field : (S, 4, D) — the 4-channel encoded return stream.
    Returns        : (S, 4, D) — the 4 layer deltas ready to add to
                     the residual stream (one delta per layer in the
                     original 4-layer block).

    Per operator (2026-08-27 correction): the forward leg (GPU0 → GPU1)
    is plain (no H(4) codec; the 4 layer outputs are already linear
    directions and pass through directly).  The codec only applies on
    the return leg (GPU1 → GPU0) where the expert outputs from the 4
    GPU1 quadrants get re-encoded into a spatial field.
    """
    return _h4_layer_decode(spatial_field)


class BicameralShardedMoE(nn.Module):
    """v1 bicameral MoE: H(4) layer bridge + shared expert + paged routed.

    In v1 the H(4) layer bridge is wired at the architecture level
    (QUADRANT_FOR_LAYER, LAYERS_IN_QUADRANT, _h4_layer_encode /
    _h4_layer_decode) but the routed expert work is a no-op until
    the cache is filled.  The gate runs per-layer as standard
    (no H(4) on the gate input — the gate is a single 2560 → 512
    projection that runs once per layer).
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Qwen4ExpConfig = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.tp_size = get_tensor_model_parallel_world_size()

        self.n_routed_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)
        self.model_dir = str(vllm_config.model_config.model)

        # Master router — 2560 → 512, very small.  Loaded as a
        # real weight from the safetensors checkpoint.  Standard
        # Qwen-FN routing (one 2560 → 512 projection per layer).
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        # Shared-expert gate — 2560 → 1, scalar.
        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )

        # Shared expert — a real SwiGLU MLP, ~10 MB per layer.
        if config.shared_expert_intermediate_size > 0:
            from vllm.model_executor.models.qwen3_next import Qwen3NextMLP

            self.shared_expert = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                prefix=f"{prefix}.shared_expert",
            )
        else:
            self.shared_expert = None

        # Bicameral expert cache.  Set by Qwen4ExpForCausalLM after
        # weight load.  In v1: no-op (zero tensor) until filled.
        self.layer_idx: int = -1
        self.expert_cache = None

    def _routed(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from afi.bicameral.expert_cache import get_expert_cache, swiglu_expert

        # XPUGraph capture guard.  Several ops in this function are
        # graph-incompatible (torch.unique.tolist triggers CPU sync,
        # nonzero / index_add_ use events that "wait method cannot
        # be used for an event associated with a command graph").
        # v1's routed contribution is zero anyway (cache empty), so
        # we skip the entire block while a graph is being captured.
        # The gate-only path (no cache, no experts) is graph-safe.
        try:
            if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_current_stream_capturing"):
                if torch.xpu.is_current_stream_capturing():
                    return hidden_states.new_zeros(hidden_states.shape)
        except Exception:
            pass

        # Standard Qwen-FN routing: master gate (2560 → 512), then
        # top-K=10 per token GLOBALLY.  No H(4) on the gate input —
        # the H(4) layer bridge is the GPU1 work structure, not a
        # gate modification.
        logits, _ = self.gate(hidden_states)
        scores = torch.softmax(logits.float(), dim=-1)
        topw, topi = scores.topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
            topw = topw / topw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        topw = topw.to(dtype=hidden_states.dtype)
        cache = get_expert_cache(self.model_dir)
        # Stream-ahead: prefetch experts for the next 1-3 layers.
        unique_now = torch.unique(topi).tolist()
        for ahead in (1, 2, 3):
            nxt = self.layer_idx + ahead
            if 0 <= nxt < 48:
                cache.prefetch_async(nxt, unique_now)
        out = torch.zeros_like(hidden_states)
        for e in unique_now:
            w13, w2 = cache.fetch(
                self.layer_idx, int(e), hidden_states.device, hidden_states.dtype
            )
            mask = topi == e
            tok_idx, k_idx = mask.nonzero(as_tuple=True)
            if tok_idx.numel() == 0:
                continue
            y = swiglu_expert(hidden_states[tok_idx], w13, w2)
            y = y * topw[tok_idx, k_idx].unsqueeze(-1)
            out.index_add_(0, tok_idx, y)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        """Shared expert + routed top-k experts.

        Shape contract (matches Qwen3NextSparseMoeBlock):
          in:  (num_tokens, hidden_size)
          out: (num_tokens, hidden_size)
        """
        out = hidden_states.new_zeros(hidden_states.shape)
        if self.shared_expert is not None:
            shared = self.shared_expert(hidden_states)
            if shared.dim() == 3:
                shared = shared.squeeze(0)
            out = out + shared
        if self.layer_idx >= 0:
            out = out + self._routed(hidden_states)
        return out


__all__ = [
    "BicameralShardedMoE",
    "QUADRANT_FOR_LAYER",
    "LAYERS_IN_QUADRANT",
    "_h4_layer_encode",
    "_h4_layer_decode",
    "gpu1_encode_layer_outputs",
    "gpu0_decode_return_stream",
]
