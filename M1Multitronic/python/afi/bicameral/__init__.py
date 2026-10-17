"""Bicameral split: sparse weight loading for Qwen3.8-Flash-Next.

Per BICAMERAL_FRAMEWORK.md §4, the 180 B parameter model lives across
host RAM (cold shards on NVMe) and GPU0 (the 11 GB active partition).
MoE experts (247 GB) and n-gram shards (102 GB) are paged on demand
through the sparse loader and the cache modules.
"""

from .expert_cache import BicameralExpertCache, get_expert_cache
from .lazy_ngram import LazyNgramLoader
from .sharded_moe import (
    BicameralShardedMoE,
    QUADRANT_FOR_LAYER,
    LAYERS_IN_QUADRANT,
    gpu0_decode_return_stream,
    gpu1_encode_layer_outputs,
)
from .sharded_ngram import BicameralNgramEmbedding
from .sparse_table import SparseTable, TensorLoc, build

__all__ = [
    "BicameralExpertCache",
    "BicameralNgramEmbedding",
    "BicameralShardedMoE",
    "LazyNgramLoader",
    "LAYERS_IN_QUADRANT",
    "QUADRANT_FOR_LAYER",
    "SparseTable",
    "TensorLoc",
    "build",
    "get_expert_cache",
    "gpu0_decode_return_stream",
    "gpu1_encode_layer_outputs",
]
