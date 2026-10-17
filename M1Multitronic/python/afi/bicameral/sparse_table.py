# SPDX-License-Identifier: Apache-2.0
"""Sparse weight table for Qwen3.8-Flash-Next (Qwen4-Exp).

Reads the safetensors index.json once at startup and classifies every
weight tensor into one of three buckets:

  * **active**   — small (11 GB total), loaded into GPU0 once at boot via
                   vLLM's standard ``AutoWeightsLoader`` path.  This is
                   the GDN recurrence, full-attention, layer norms,
                   embeddings, MoE gates and shared-expert weights.
  * **ngram**    — 128 shards at layer 1, 800 MB each, 102.4 GB total.
                   Resides on NVMe.  Paged into host RAM (8 GB page
                   pool) on demand.  See ``bicameral/igram.py``.
  * **moe_expert** — 12 layers × (gate_up_proj + down_proj) = 24
                   tensors, ~10 GB each, ~247 GB total.  Resides on
                   NVMe.  Paged into GPU0 (4 GB cache) per the
                   expert-activation change.  See
                   ``bicameral/expert_cache.py``.

The shape and dtype for every tensor is also recorded so the cache
modules can ``safe_open`` the right file and slice the right tensor
without re-reading the index.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TensorLoc:
    """One tensor's location + shape."""

    name: str  # full key from the index, e.g. "model.language_model.layers.0.mlp.experts.gate_up_proj"
    file: str  # safetensors file basename, e.g. "model-00002-of-00131.safetensors"
    shape: Tuple[int, ...]
    dtype: str  # string form, e.g. "torch.bfloat16"
    size_bytes: int

    @property
    def nbytes(self) -> int:
        return self.size_bytes


@dataclass
class SparseTable:
    """Sparse weight table for one Qwen-FN model directory."""

    model_dir: str
    active: List[TensorLoc] = field(default_factory=list)
    ngram: List[TensorLoc] = field(default_factory=list)
    moe_expert: List[TensorLoc] = field(default_factory=list)
    ngram_ple_layer: int = 1  # 0-indexed owner (config ple_layer_ids=[2])

    def by_name(self) -> Dict[str, TensorLoc]:
        out: Dict[str, TensorLoc] = {}
        for t in self.active:
            out[t.name] = t
        for t in self.ngram:
            out[t.name] = t
        for t in self.moe_expert:
            out[t.name] = t
        return out

    def summary(self) -> str:
        def gb(tensors: List[TensorLoc]) -> float:
            return sum(t.size_bytes for t in tensors) / 1e9

        return (
            f"active={len(self.active)} ({gb(self.active):.2f} GB), "
            f"ngram={len(self.ngram)} ({gb(self.ngram):.2f} GB), "
            f"moe_expert={len(self.moe_expert)} ({gb(self.moe_expert):.2f} GB), "
            f"ngram_ple_layer={self.ngram_ple_layer}"
        )


def _shape_size_bytes(shape: Tuple[int, ...], dtype: str) -> int:
    """Multiply elements × dtype size."""
    elem_bytes = {
        "torch.bfloat16": 2,
        "torch.float16": 2,
        "torch.float32": 4,
        "torch.float64": 8,
        "torch.int8": 1,
        "torch.int16": 2,
        "torch.int32": 4,
        "torch.int64": 8,
        "torch.uint8": 1,
        "torch.bool": 1,
    }.get(dtype, 2)
    n = 1
    for d in shape:
        n *= d
    return n * elem_bytes


def _classify(name: str) -> str:
    """Return one of 'active', 'ngram', 'moe_expert', 'visual', 'mtp'."""
    from afi.vllm_qwen4exp.layout import classify_weight

    return classify_weight(name)


def _extract_ple_layer(name: str) -> int | None:
    """If this is a PLE/ngram tensor, return the layer that owns it."""
    if ".ngram_embedding" in name or ".ple." in name:
        # model.language_model.layers.{N}.ple...
        import re

        m = re.search(r"layers\.(\d+)\.ple", name)
        if m:
            return int(m.group(1))
    return None


def build(model_dir: str) -> SparseTable:
    """Build the sparse weight table for ``model_dir``.

    Reads ``model.safetensors.index.json`` and probes the shape of every
    unique safetensors file once via ``safetensors.safe_open``.  For
    131 files this takes ~10 seconds on the B70 NVMe.
    """
    import safetensors.torch as st

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        # Some checkpoints ship without an index; build one from the
        # directory listing (slower — opens each file to enumerate keys).
        raise FileNotFoundError(
            f"missing {index_path}; this loader requires a sharded "
            "safetensors index.  Qwen-FN ships one."
        )

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    # Group by file (so we open each file once).
    files_to_keys: Dict[str, List[str]] = {}
    for k, fn in weight_map.items():
        files_to_keys.setdefault(fn, []).append(k)

    table = SparseTable(model_dir=model_dir)

    # ngram_ple_layer: derive from any ngram tensor
    for name in weight_map:
        layer = _extract_ple_layer(name)
        if layer is not None and "ngram_embedding" in name:
            table.ngram_ple_layer = layer
            break

    for fn, keys in sorted(files_to_keys.items()):
        path = os.path.join(model_dir, fn)
        with st.safe_open(path, framework="pt") as sf:
            # Build a name→tensor map for this file.
            for k in keys:
                t = sf.get_tensor(k)
                shape = tuple(t.shape)
                dtype = str(t.dtype)
                loc = TensorLoc(
                    name=k,
                    file=fn,
                    shape=shape,
                    dtype=dtype,
                    size_bytes=_shape_size_bytes(shape, dtype),
                )
                bucket = _classify(k)
                if bucket == "ngram":
                    table.ngram.append(loc)
                elif bucket == "moe_expert":
                    table.moe_expert.append(loc)
                elif bucket == "active":
                    table.active.append(loc)
                # visual / mtp are not GPU0-resident in the text-only loader

    logger.info("Sparse table built for %s: %s", model_dir, table.summary())
    return table


__all__ = ["TensorLoc", "SparseTable", "build"]
