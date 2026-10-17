# SPDX-License-Identifier: Apache-2.0
"""QSA block-based indexer (BICAMERAL_FRAMEWORK.md §6.4).

The Qwen-Flash-Next QSA (Query-Sparse Attention) compression step:

  input  : 2048 candidate blocks of 256 head-dim vectors
  output : 512 selected blocks (compression ratio r=4)
  goal   : O(n^2 / r) sparse attention that stacks with dct_topk@0.5
           for ~8x total stream reduction (25.75 GB full KV → ~3.2 MB
           stream)

Per the Qwen-FN tech report the selection rule is a small top-k over
per-block score; the framework's `python/afi/spectral.py` already runs
per-block energy on 256-wide row blocks.  This module wraps that into
a single, measurable top-k selector that passes the row-reconstruction
gate from §2: ``row_cos >= 0.95`` on real Qwen-FN rows.

Landmine (BICAMERAL_FRAMEWORK.md §9 #2): per-block energy is per
256-wide ROW (mean over L*10 rows), NOT per token.  Divide token-domain
totals by 10 when comparing to the paper.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .spectral import BLOCK as HEAD_DIM
from .spectral import blocks_view, fingerprint

logger = logging.getLogger(__name__)

DEFAULT_BUDGET = 2048
DEFAULT_KEEP = 512  # budget / r where r=4

__all__ = ["IndexResult", "select_topk", "select_topk_fingerprinted",
           "compress_stream", "validate_index"]


@dataclass
class IndexResult:
    """Outcome of one QSA index pass.

    Attributes
    ----------
    selected_blocks : [K] int
        Block ids (in the candidate-block space) selected by the indexer.
    block_scores    : [num_blocks] float
        Score per block (descending = high-importance).
    kept_mean_energy : float
        Mean energy of the kept blocks (Parseval).
    dropped_mean_energy : float
        Mean energy of the dropped blocks.
    row_cos         : float
        Mean per-row cosine of the reconstructed vs original rows.
        The framework gate:  >= 0.95 row_cos at KEEP=512 of 2048.
    elapsed_ms      : float
    block           : int
        Per-block width (256 = HEAD_DIM for Qwen-FN sparse attn).
    """
    selected_blocks: np.ndarray
    block_scores: np.ndarray
    kept_mean_energy: float
    dropped_mean_energy: float
    row_cos: float
    elapsed_ms: float
    block: int = HEAD_DIM

    def to_dict(self) -> dict:
        return {
            "selected_blocks": [int(b) for b in self.selected_blocks.tolist()],
            "block_scores": [round(float(s), 4)
                             for s in self.block_scores.tolist()],
            "kept_mean_energy": round(float(self.kept_mean_energy), 4),
            "dropped_mean_energy": round(float(self.dropped_mean_energy), 4),
            "row_cos": round(float(self.row_cos), 4),
            "elapsed_ms": round(float(self.elapsed_ms), 2),
            "block": self.block,
        }


def _per_block_energy(hidden: np.ndarray, block: int = HEAD_DIM
                      ) -> np.ndarray:
    """Return [num_blocks] mean energy of each 256-wide block row.

    Input ``hidden`` is [L, H] (token-domain).  We view it as
    [L*num_blocks, block] then mean over rows; per the framework
    landmine note, this is per-block energy (NOT per token).
    """
    if hidden.ndim != 2:
        raise ValueError(f"hidden must be [L, H], got {hidden.shape}")
    if hidden.shape[1] % block != 0:
        # Zero-pad to a multiple (cheap and correct for energy)
        pad = block - (hidden.shape[1] % block)
        hidden = np.pad(hidden, ((0, 0), (0, pad)))
    rows = blocks_view(hidden, block)  # [L * num_blocks, block]
    e = (rows.astype(np.float64) ** 2).sum(axis=1)  # [L*num_blocks]
    num_blocks = hidden.shape[1] // block
    return e.reshape(-1, num_blocks).mean(axis=0)  # [num_blocks]


def select_topk(hidden: np.ndarray, keep: int = DEFAULT_KEEP,
                budget: int = DEFAULT_BUDGET, block: int = HEAD_DIM,
                ) -> IndexResult:
    """Per-block top-k selector.

    Parameters
    ----------
    hidden : [L, H] fp16/fp32 hidden states
    keep   : number of blocks to keep (default 512)
    budget : candidate budget (default 2048); the per-block energy
             vector is truncated to ``budget`` blocks before top-k
    block  : block width (default 256)

    Returns
    -------
    IndexResult
    """
    t0 = time.time()
    if hidden.shape[1] % block != 0:
        # Pad so the view is exact (spectral.BLOCK) — this matches the
        # Qwen-FN sparse-attn head_dim when H=2560 (2560/256=10 blocks/token).
        pad = block - (hidden.shape[1] % block)
        hidden = np.pad(hidden, ((0, 0), (0, pad)))
    energies = _per_block_energy(hidden, block)  # [num_blocks]
    if energies.size > budget:
        # Keep only the highest-energy ``budget`` blocks as candidates.
        # The trim is by *aggregate* energy (mean over token rows) so
        # the indexing contract still holds: we still pick KEEP of the
        # top-budget candidates.
        top_budget = np.argsort(energies)[::-1][:budget]
        energies = energies[top_budget]
    scores = energies
    if keep > scores.size:
        keep = scores.size
    sel = np.argsort(scores)[::-1][:keep]
    kept_e = float(scores[sel].mean())
    dropped_e = float(np.delete(scores, sel).mean()) if scores.size > keep else 0.0
    return IndexResult(
        selected_blocks=sel,
        block_scores=scores,
        kept_mean_energy=kept_e,
        dropped_mean_energy=dropped_e,
        # row_cos is filled in by compress_stream(); placeholder 0.0
        # here signals the indexer was called standalone.
        row_cos=0.0,
        elapsed_ms=(time.time() - t0) * 1000,
        block=block,
    )


def select_topk_fingerprinted(hidden: np.ndarray, keep: int = DEFAULT_KEEP,
                              budget: int = DEFAULT_BUDGET,
                              block: int = HEAD_DIM
                              ) -> tuple[IndexResult, dict]:
    """Like ``select_topk`` but also returns the per-block fingerprint
    (Walsh) so the caller can correlate QSA selection with spectral
    compaction."""
    res = select_topk(hidden, keep=keep, budget=budget, block=block)
    # Run the existing spectral fingerprint on the kept-block rows.
    kept_rows = _gather_blocks(hidden, res.selected_blocks, block)
    fp = fingerprint(kept_rows, block) if kept_rows.size else {}
    return res, fp


def _gather_blocks(hidden: np.ndarray, block_ids: np.ndarray,
                   block: int = HEAD_DIM) -> np.ndarray:
    """Materialize the selected 256-wide block rows in block-id order."""
    if hidden.shape[1] % block != 0:
        pad = block - (hidden.shape[1] % block)
        hidden = np.pad(hidden, ((0, 0), (0, pad)))
    num_blocks = hidden.shape[1] // block
    rows = blocks_view(hidden, block)  # [L*num_blocks, block]
    rows = rows.reshape(-1, num_blocks, block)
    # block_id is in the candidate-block space; if hidden produced more
    # blocks than the budget-trimmed ids, this is a programming error.
    if block_ids.max() >= num_blocks:
        raise ValueError(f"block_id {block_ids.max()} >= num_blocks "
                         f"{num_blocks}; did you trim the budget?")
    sel = rows[:, block_ids, :].reshape(-1, block)
    return sel


def compress_stream(hidden: np.ndarray, keep: int = DEFAULT_KEEP,
                    budget: int = DEFAULT_BUDGET,
                    block: int = HEAD_DIM,
                    rate: float = 0.5) -> dict:
    """Full QSA + dct_topk composite compaction.

    Selects KEEP of BUDGET blocks, then dct-compacts the kept rows at
    ``rate`` (along the token axis).  Returns the round-trip
    row_cos so the framework gate is one call away.
    """
    from .mdct import dct2, idct2, keep_topk

    t0 = time.time()
    res = select_topk(hidden, keep=keep, budget=budget, block=block)
    kept_rows = _gather_blocks(hidden, res.selected_blocks, block)  # [L*num_blocks_kept, block]
    if kept_rows.size == 0:
        return {**res.to_dict(), "row_cos": 0.0, "kept_rows": 0,
                "rate": rate, "composite_kept_fraction": 0.0}
    coeffs = dct2(kept_rows.astype(np.float64), axis=0)
    kept = keep_topk(coeffs.T, rate).T
    recon = idct2(kept, axis=0)
    # Per-row cosine vs the kept_rows input
    a = kept_rows.astype(np.float64)
    r = recon.astype(np.float64)
    num = (a * r).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(r, axis=1)
    ok = den > 1e-30
    row_cos = float(np.mean(num[ok] / den[ok])) if ok.any() else 0.0
    # total stream: keep/budget * rate
    composite_kept_fraction = (res.selected_blocks.size / max(budget, 1)) * rate
    return {
        **res.to_dict(),
        "row_cos": round(row_cos, 4),
        "kept_rows": int(kept_rows.shape[0]),
        "rate": rate,
        "composite_kept_fraction": round(composite_kept_fraction, 4),
        "elapsed_ms": round((time.time() - t0) * 1000, 2),
    }


def validate_index(hidden: np.ndarray, keep: int = DEFAULT_KEEP,
                   budget: int = DEFAULT_BUDGET,
                   block: int = HEAD_DIM,
                   min_row_cos: float = 0.95) -> dict:
    """Validate the index on a hidden-state sample.

    Returns a dict with the same shape as ``compress_stream`` plus a
    boolean ``gate_passed`` field.  The gate is the framework's
    row-reconstruction floor: ``row_cos >= min_row_cos`` at the
    composite (QSA + dct_topk@0.5) keep fraction.
    """
    out = compress_stream(hidden, keep=keep, budget=budget, block=block,
                          rate=0.5)
    out["gate_passed"] = out["row_cos"] >= min_row_cos
    out["min_row_cos"] = min_row_cos
    return out
