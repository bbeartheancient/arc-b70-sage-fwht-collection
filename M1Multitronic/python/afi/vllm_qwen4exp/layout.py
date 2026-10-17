# SPDX-License-Identifier: Apache-2.0
"""Checkpoint / config layout for Qwen3.8-Flash-Next (qwen4_exp).

Pure helpers — no vLLM, no torch.  Sourced from ~/QwenFlash/config.json,
the tech report (Eq. 21–34), and the HF modeling_qwen4_exp reference.

Index landmine (BICAMERAL + tech report §2.3):
  ``ple_layer_ids`` in config.json is **1-indexed**.  The n-gram PLE
  module lives at ``model.language_model.layers.{id-1}.ple.*``.
  Config ``[2]`` → weights on layer 1 (0-indexed).
"""

from __future__ import annotations

from typing import Iterable, Sequence

# Qwen-FN card (matches ~/QwenFlash/config.json).
HIDDEN_SIZE = 2560
NUM_LAYERS = 48
FULL_ATTENTION_INTERVAL = 4
HC_COUNT = 4
HC_LOWRANK = 320  # d/8
PLE_LAYER_IDS_ONE_INDEXED = (2,)
NUM_EXPERTS = 512
NUM_EXPERTS_PER_TOK = 10
MOE_INTERMEDIATE = 640


def default_layer_types(
    n: int = NUM_LAYERS,
    interval: int = FULL_ATTENTION_INTERVAL,
) -> list[str]:
    """3× linear_attention + 1× full_attention, repeating.

    Full-attention sits at 0-indexed 3, 7, 11, …, 47 — i.e. 1-indexed
    multiples of ``interval``.  The inverted ``i % interval == 0``
    default (full on 0, 4, 8, …) is wrong for this checkpoint.
    """
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n)
    ]


def ple_zero_index(ple_layer_ids: Sequence[int] | None) -> list[int]:
    """Convert 1-indexed ``ple_layer_ids`` to 0-indexed layer indices."""
    ids = list(ple_layer_ids) if ple_layer_ids else list(PLE_LAYER_IDS_ONE_INDEXED)
    out = []
    for i in ids:
        z = int(i) - 1
        if z < 0:
            raise ValueError(f"ple_layer_ids is 1-indexed; got {i}")
        out.append(z)
    return out


def classify_weight(name: str) -> str:
    """Bucket a checkpoint key: active / ngram / moe_expert / visual / mtp.

    ``active`` is the GPU0-resident partition (GDN, GQA, GR, shared
    expert, router, embeddings, PLE *projections*).  N-gram shards and
    routed-expert tiles stay on NVMe (BICAMERAL §4 / §6.3).
    """
    if name.startswith("mtp.") or name.startswith("model.mtp."):
        return "mtp"
    if (
        name.startswith("visual.")
        or name.startswith("model.visual.")
        or ".visual." in name
    ):
        return "visual"
    if ".ngram_embedding.shard_" in name:
        return "ngram"
    if ".experts.gate_up_proj" in name or ".experts.down_proj" in name:
        return "moe_expert"
    return "active"


def should_skip_weight(name: str, *, skip_indexer: bool = True) -> bool:
    """True if the v1 loader must not feed this tensor to AutoWeightsLoader."""
    bucket = classify_weight(name)
    if bucket in ("mtp", "visual", "ngram", "moe_expert"):
        return True
    if skip_indexer and (
        ".self_attn.indexer." in name or "indexer.index_qk_proj" in name
    ):
        return True
    return False


def find_decoder_blocks(model):
    """Locate the decoder ModuleList on a vLLM runner model.

    Qwen4ExpForCausalLM wraps the trunk as ``.model.layers``; a bare
    Qwen3Next-style class exposes ``.layers`` directly.  Walk one
    ``.model`` / ``.language_model`` hop so mid-stack hidden_states
    hooks (engine.py) fire on Qwen-FN.
    """
    seen: set[int] = set()
    cur = model
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        for attr in ("layers", "decoder_layers"):
            blocks = getattr(cur, attr, None)
            if blocks is None:
                continue
            try:
                len(blocks)
            except TypeError:
                continue
            return blocks
        cur = getattr(cur, "model", None) or getattr(cur, "language_model", None)
    return None


def hidden_size_from_config(cfg, default: int = 5120) -> int:
    """Read hidden_size from a top-level or text-sub-config object."""
    for obj in (cfg, getattr(cfg, "text_config", None)):
        if obj is None:
            continue
        hs = getattr(obj, "hidden_size", None)
        if hs:
            return int(hs)
    return int(default)


def strip_language_model_prefix(name: str) -> str:
    """``model.language_model.X`` → ``model.X`` (vLLM has no LM wrapper)."""
    prefix = "model.language_model."
    if name.startswith(prefix):
        return "model." + name[len(prefix) :]
    return name


def hc_shapes(hidden: int = HIDDEN_SIZE, nr: int = HC_COUNT, rank: int = HC_LOWRANK):
    """Checkpoint shapes for one GR module (tech report Eq. 31–33)."""
    hc = nr * hidden
    return {
        "input_mix_weight_down": (rank, hc),   # Wd
        "input_mix_weight_up": (hc, rank),     # Wu
        "block_inject_weight": (nr, hc),       # Ww
        "hc_norm": (hc,),
    }


def expand_to_hc(embed, nr: int = HC_COUNT):
    """Tile the token embed across all ``nr`` residual branches.

    Matches ``transformers`` ``Qwen4ExpTextModel.forward``:
    ``hidden_states = hidden_states.repeat(1, 1, hc_count)``.  Zero-filling
    branches 1..nr-1 (AltUp-style) is wrong for this checkpoint.
    """
    out = []
    for row in embed:
        row = list(row)
        out.append(row * nr)
    return out


__all__ = [
    "HIDDEN_SIZE",
    "NUM_LAYERS",
    "FULL_ATTENTION_INTERVAL",
    "HC_COUNT",
    "HC_LOWRANK",
    "PLE_LAYER_IDS_ONE_INDEXED",
    "classify_weight",
    "default_layer_types",
    "expand_to_hc",
    "find_decoder_blocks",
    "hc_shapes",
    "hidden_size_from_config",
    "ple_zero_index",
    "should_skip_weight",
    "strip_language_model_prefix",
]
