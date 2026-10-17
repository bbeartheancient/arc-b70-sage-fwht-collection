"""Spectral fingerprints of hidden states via orthonormal FWHT.

Shared by the looping-transformer stack: scripts/latent-fwht-worker.py
(B70, torch.ops.afi.fwht) and fabric/latent.py (host fallback). The
transform is orthogonal (Parseval), so energies reported here compare
directly with token-domain squared norms.

Per-block energy is per BLOCK-wide row, NOT per token.  The number of
blocks per token depends on the model:
  - Qwen 3.8 27B (H=5120, head_dim=128): nb = 40  (BLOCK=128)
  - Qwen-FN     (H=2560, head_dim=256): nb = 10  (BLOCK=256)
Divide token-domain totals by nb when comparing to papers.
"""

from __future__ import annotations

import numpy as np

BLOCK = 256  # Qwen-FN sparse/full attn head_dim; hidden rows viewed in blocks
TOP_BINS = 6


def blocks_view(x: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Zero-pad the last dim to a multiple of ``block``; return
    ``[rows * nb, block]`` (per-token block rows, order preserved)."""
    rows, width = x.shape
    nb = -(-width // block)
    if nb * block != width:
        x = np.pad(x, ((0, 0), (0, nb * block - width)))
    return x.reshape(rows * nb, block)


def fingerprint(coeffs: np.ndarray, block: int = BLOCK) -> dict:
    """Summarize Walsh coefficients ``[rows, block]`` from an
    ORTHONORMAL transform.

    - mean/p50/p95 row energy (sum of squared coefficients per row;
      Parseval makes these equal token-domain energies)
    - compaction_90: mean fraction of coefficients holding 90% of a
      row's energy — lower means a more compressible spectrum
      (the Phase 3 spectral-compaction signal)
    - top_bins: Walsh bins with the highest mean normalized share
    """
    c = np.asarray(coeffs, dtype=np.float64)
    e = (c * c).sum(axis=1)
    out = {
        "block": int(block),
        "rows": int(c.shape[0]),
        "mean_energy": round(float(e.mean()), 4),
        "energy_p50": round(float(np.percentile(e, 50)), 4),
        "energy_p95": round(float(np.percentile(e, 95)), 4),
    }
    if not e.size or float(e.sum()) <= 0.0:
        out.update({"compaction_90": 1.0, "top_bins": []})
        return out
    sq = (c * c).astype(np.float32)
    vals = np.sort(sq, axis=1)[:, ::-1]
    totals = np.maximum(vals.sum(axis=1, keepdims=True), 1e-12)
    cum = np.cumsum(vals / totals, axis=1)
    k90 = np.minimum((cum < 0.9).sum(axis=1) + 1, sq.shape[1])
    out["compaction_90"] = round(float(k90.mean()) / block, 4)
    share = (sq / np.maximum(e[:, None].astype(np.float32),
                             1e-12)).mean(axis=0)
    top = np.argsort(share)[::-1][:TOP_BINS]
    out["top_bins"] = [{"bin": int(b), "share": round(float(share[b]), 4)}
                       for b in top]
    return out
