# SPDX-License-Identifier: Apache-2.0
"""Qwen27bPLE — writable n-gram PLE wired into the 27B forward pass.

Architecture (docs/27b-writable-ngram-plan.md § Step 2):

  The Qwen3.8-27B has 64 layers in a 3+1 hybrid pattern: 16 blocks of
  4 layers, full-attention at 0-indexed positions 3, 7, 11, ..., 63.
  The H(4) gate maps each 4-layer block to 4 directional components
  (W/X/Y/Z); row 0 (W = [+ + + +]) is the full-attention consensus.

  Qwen27bPLE holds 16 WritableNgramStore instances — one per
  full-attention layer.  At each full-attention layer, the n-gram
  table is read for the current token ids, the stored 160-dim edges
  are combined weighted by the H(4) directional magnitudes, projected
  through a per-layer value_proj (160 → hidden_size), and added to
  the residual stream.

  The table is initialized EMPTY and grows at inference time via
  ngram_store (Step 3 wires the EEL2 verbs; Step 5 trains the
  value_proj + tap gates per the Whittle pattern).  An empty table
  is an exact no-op: lookups return zeros, the projected delta is
  zero, and the 27B forward is bit-identical to baseline.

  H(4) directional magnitudes: for a block's 4-channel spatial field
  (S, 4, D), the magnitude per direction is the per-token L2 norm
  (S, 4), normalized to sum 1.  With a single store row per
  (token, trigram_pos, layer) the 4 edges collapse to one shared
  edge modulated by these weights — the per-edge split (4 stored
  vectors per node) is a capacity upgrade, not a wiring change.

  numpy-first (fabric venv has no torch); torch tensors are accepted
  by duck-typing (detach/cpu/numpy) and deltas returned in the
  caller's framework.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np

import writable_ngram as wn  # sibling module (direct-import pattern)

logger = logging.getLogger(__name__)

# 27B layer topology: 64 layers, 16 blocks of 4, full-attn at 4b+3.
N_LAYERS_27B = 64
FULL_ATTN_LAYERS_27B = [4 * b + 3 for b in range(16)]  # 3, 7, ..., 63
QUADRANT_FOR_LAYER_27B = [1, 2, 3, 0] * 16  # same pattern as Qwen-FN

# H(4) Sylvester-Hadamard, orthonormal (H4 @ H4.T = I).  Row 0 = W =
# full-attention consensus; rows 1-3 = X/Y/Z linear directions.
_H4_NP = np.array(
    [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0, 1.0],
    ],
    dtype=np.float32,
) / 2.0

_NGRAM_DIM = wn._NGRAM_DIM          # 160
_DEFAULT_HIDDEN = 5120               # Qwen3.8-27B hidden_size


def _to_numpy(x) -> np.ndarray:
    """Accept numpy arrays or torch tensors; return numpy."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


def h4_encode_block(layer_outputs: np.ndarray) -> np.ndarray:
    """H(4) on the layer axis.  (S, 4, D) → (S, 4, D).

    Pure-numpy mirror of sharded_moe._h4_layer_encode (which is
    torch-only and lives behind the vllm import).  Bit-identical at
    fp32 noise: H4 @ x per token.
    """
    lo = _to_numpy(layer_outputs)
    assert lo.shape[1] == 4, f"expected 4 layers per block, got {lo.shape[1]}"
    return np.einsum("ij,sjd->sid", _H4_NP, lo.astype(np.float32))


def h4_direction_magnitudes(spatial_field: np.ndarray) -> np.ndarray:
    """Per-token L2 norm of each H(4) direction, normalized to sum 1.

    spatial_field : (S, 4, D) — the H(4)-encoded block field.
    Returns        : (S, 4) float32, rows sum to 1 (uniform 1/4 when
                     the field is all zeros — a degenerate block
                     weights all directions equally).
    """
    sf = _to_numpy(spatial_field).astype(np.float32)
    mag = np.linalg.norm(sf, axis=-1)              # (S, 4)
    total = mag.sum(axis=-1, keepdims=True)
    uniform = np.full_like(mag, 0.25)
    return np.where(total > 0, mag / np.maximum(total, 1e-20), uniform)


class Qwen27bPLE:
    """Writable n-gram PLE for the 27B: 16 per-layer MV2 stores.

    Parameters
    ----------
    shard_dir   : directory holding qwen27b_ple_layer_{N}.mv2 files
                  (default: wn._default_shard_dir(), env NGRAM_SHARD_DIR)
    hidden_size : 27B hidden size (5120)
    layers      : full-attention layer ids that carry PLE stores
    read_only   : open shards without write locks (inference-only)

    Usage in the forward pass (per full-attention layer L):

        delta = ple.forward(hidden, token_ids, h4_field, layer_id=L)
        hidden = hidden + delta

    where ``h4_field`` is the (S, 4, D) H(4)-encoded spatial field of
    the block containing L (h4_encode_block of the block's 4 layer
    outputs).  ``delta`` is zero until the table is populated.
    """

    def __init__(
        self,
        shard_dir: Path | None = None,
        hidden_size: int = _DEFAULT_HIDDEN,
        layers: list[int] | None = None,
        read_only: bool = False,
    ):
        self.shard_dir = Path(shard_dir) if shard_dir else wn._default_shard_dir()
        self.hidden_size = hidden_size
        self.layers = list(layers) if layers else list(FULL_ATTN_LAYERS_27B)
        self.read_only = read_only
        self._stores: dict[int, wn.WritableNgramStore] = {}
        self._lock = threading.RLock()
        # Per-layer value projection: (NGRAM_DIM, hidden_size) fp32.
        # Initialized to ZEROS — an untrained PLE is an exact no-op on
        # the residual even after the table is populated.  Step 5
        # (scripts/train_ngram_ple.py) learns these; a trained
        # value_proj_layer{L}.npz in the shard dir is auto-loaded.
        self.value_proj = {
            L: self._load_value_proj(L, hidden_size) for L in self.layers
        }

    def _load_value_proj(self, layer_id: int,
                         hidden_size: int) -> np.ndarray:
        path = self.shard_dir / f"value_proj_layer{layer_id}.npz"
        if path.exists():
            try:
                vp = np.load(path)["value_proj"].astype(np.float32)
                if vp.shape == (wn._NGRAM_DIM, hidden_size):
                    logger.info("loaded trained value_proj for layer %d",
                                layer_id)
                    return vp
                logger.warning("value_proj shape mismatch for layer %d: "
                               "%s", layer_id, vp.shape)
            except Exception as exc:  # noqa: BLE001
                logger.warning("value_proj load failed for layer %d: %s",
                               layer_id, exc)
        return np.zeros((wn._NGRAM_DIM, hidden_size), dtype=np.float32)

    # ------------------------------------------------------------------
    # Stores
    # ------------------------------------------------------------------
    def _store(self, layer_id: int) -> wn.WritableNgramStore:
        with self._lock:
            st = self._stores.get(layer_id)
            if st is None:
                st = wn.open_store(
                    layer_id, shard_dir=self.shard_dir, read_only=self.read_only
                )
                self._stores[layer_id] = st
            return st

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states,
        token_ids,
        h4_field,
        layer_id: int,
        trigram_pos: int = 0,
    ):
        """N-gram lookup + H(4)-weighted combine + value projection.

        hidden_states : (S, hidden_size) numpy or torch
        token_ids     : (S,) int
        h4_field      : (S, 4, hidden_size) H(4) spatial field for the
                        block containing layer_id
        layer_id      : full-attention layer index (must be in
                        self.layers)
        trigram_pos   : which of the 8 n-gram buckets (default 0)

        Returns a delta of the same shape/framework as hidden_states
        to add to the residual stream.
        """
        if layer_id not in self.layers:
            raise ValueError(f"layer {layer_id} has no PLE store")
        tids = _to_numpy(token_ids).astype(np.int64).reshape(-1)
        S = tids.shape[0]

        edges = self._store(layer_id).lookup_batch(tids, trigram_pos)  # (S,160)

        # H(4) directional weight: the full-attention layer reads the
        # W (row 0 / consensus) component of its block's field.  Its
        # normalized magnitude gates the edge contribution per token.
        w = h4_direction_magnitudes(h4_field)[:, 0]  # (S,) W weights

        proj = self.value_proj[layer_id]             # (160, hidden)
        delta = (edges.astype(np.float32) @ proj) * w[:, None]  # (S,hidden)

        if hasattr(hidden_states, "detach"):  # torch in → torch out
            import torch
            return torch.from_numpy(delta).to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
        return delta.astype(np.asarray(hidden_states).dtype, copy=False)

    # ------------------------------------------------------------------
    # Write path (inference-time learning)
    # ------------------------------------------------------------------
    def store_turn(
        self,
        token_ids,
        vectors,
        layer_id: int,
        trigram_pos: int = 0,
        metadata: dict | None = None,
    ) -> None:
        """Write one turn's n-gram rows into a layer's store (WAL).

        token_ids : (S,) int
        vectors   : (S, 160) float16 — the edge vectors to persist
                    (e.g. a projection of the layer's hidden state)
        """
        if layer_id not in self.layers:
            raise ValueError(f"layer {layer_id} has no PLE store")
        tids = _to_numpy(token_ids).astype(np.int64).reshape(-1)
        vecs = _to_numpy(vectors).astype(np.float16)
        rows = [(int(t), trigram_pos, vecs[i]) for i, t in enumerate(tids)]
        self._store(layer_id).store(rows, metadata=metadata)

    def flush(self) -> None:
        """Commit all open stores (call at session end)."""
        with self._lock:
            for st in self._stores.values():
                st.commit()

    def close(self) -> None:
        self.flush()
        with self._lock:
            for st in self._stores.values():
                st.close()
            self._stores.clear()

    def stats(self) -> dict:
        with self._lock:
            return {L: st.stats() for L, st in self._stores.items()}


__all__ = [
    "Qwen27bPLE",
    "FULL_ATTN_LAYERS_27B",
    "QUADRANT_FOR_LAYER_27B",
    "N_LAYERS_27B",
    "h4_encode_block",
    "h4_direction_magnitudes",
]
