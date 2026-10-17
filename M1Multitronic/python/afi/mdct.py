"""MDCT / DCT-II spectral compaction of hidden-state sequences.

Phase 3 of the looping transformer: compress CONTEXT along the token
axis. Each channel of an [L, H] hidden sequence is a 1-D signal over
token position; DCT-II (whole sequence) or windowed MDCT (framed,
TDAC overlap-add) produces a ~critically-sampled spectral copy.
Compaction keeps a fraction of the coefficients, zeroes the rest,
and inverts; error curves measure how much context survives.

The MDCT follows the EEL2/jsfx convention already verified in-repo
(fabric/web/jsfx.js mdct/imdct + jsfx_conformance.js TDAC gate):
window sin(pi/n (m+1/2)) over the full frame n, kernel phase
(m + n/4 + 1/2)(2k+1) pi/n, forward x 1/sqrt(2), inverse x 4 sqrt(2)/n
then windowed, overlap-add at hop n/2. Frames here are edge-padded by
one hop so every token position is doubly covered and the round-trip
is exact over the whole sequence.

Host reference only (numpy). Boring on purpose: cached basis matrices,
dense matmuls — L is at most a couple thousand here.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = [
    "dct2",
    "idct2",
    "mdct_analyze",
    "mdct_synthesize",
    "keep_low",
    "keep_topk",
    "compact_sequence",
    "rel_err",
    "row_cos_mean",
]


@lru_cache(maxsize=32)
def _dct_basis(n: int) -> np.ndarray:
    """Orthonormal DCT-II basis [n, n]: B @ x == DCT(x), B.T @ X == IDCT."""
    k = np.arange(n)[:, None]
    m = np.arange(n)[None, :]
    c = np.full(n, 1.0)
    c[0] = np.sqrt(0.5)
    return np.sqrt(2.0 / n) * c[:, None] * np.cos(np.pi * (2 * m + 1) * k / (2 * n))


def dct2(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.moveaxis(
        _dct_basis(x.shape[axis]) @ np.moveaxis(x, axis, 0), 0, axis)


def idct2(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.moveaxis(
        _dct_basis(x.shape[axis]).T @ np.moveaxis(x, axis, 0), 0, axis)


@lru_cache(maxsize=8)
def _mdct_plan(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EEL2-convention analysis/synthesis matrices for frame length n."""
    m = np.arange(n)
    k = np.arange(n // 2)
    win = np.sin(np.pi / n * (m + 0.5))
    ker = np.cos(np.pi / n * (m[:, None] + (n // 2 + 1) / 2)
                 * (2 * k[None, :] + 1))
    analysis = win[:, None] * ker / np.sqrt(2.0)        # [n, n/2]
    synthesis = ker * (4.0 / n) * np.sqrt(2.0)          # [n, n/2] pre-window
    return analysis, synthesis, win


def mdct_analyze(seq: np.ndarray, n: int = 128) -> tuple[np.ndarray, tuple]:
    """[T, H] sequence -> (coeffs [F, n/2, H], plan-meta for synthesize).

    The token axis is edge-padded by one hop (n/2) first, so every
    position is doubly covered on synthesis."""
    seq = np.asarray(seq, dtype=np.float64)
    t = seq.shape[0]
    hop = n // 2
    # Enough frames that every original position keeps a full COLA pair:
    # the last frame start must reach >= t - hop.
    starts = np.arange(-(-t // hop) + 1) * hop
    pad_right = max(hop, int(starts[-1]) + n - t - hop)
    xp = np.pad(seq, ((hop, pad_right), (0, 0)), mode="edge")
    analysis, _, _ = _mdct_plan(n)
    coeffs = np.stack([
        (analysis.T @ xp[s:s + n]) for s in starts
    ])
    return coeffs, (seq.shape[0], n)


def mdct_synthesize(coeffs: np.ndarray, meta: tuple) -> np.ndarray:
    """[F, n/2, H] MDCT coefficients -> [T, H] via windowed overlap-add."""
    t_len, n = meta
    hop = n // 2
    _, synthesis, win = _mdct_plan(n)
    starts = np.arange(coeffs.shape[0]) * hop
    out = np.zeros((int(starts[-1]) + n, coeffs.shape[2]))
    for i in range(coeffs.shape[0]):
        s = int(starts[i])
        out[s:s + n] += (synthesis @ coeffs[i]) * win[:, None]
    return out[hop:hop + t_len]


def keep_low(coeffs: np.ndarray, rate: float, axis: int = -1) -> np.ndarray:
    """Keep the lowest `rate` fraction of coefficients along `axis`
    (causal lowpass, zero side information)."""
    n = coeffs.shape[axis]
    keep = max(1, int(round(rate * n)))
    sl = [slice(None)] * coeffs.ndim
    sl[axis] = slice(keep, None)
    out = coeffs.copy()
    out[tuple(sl)] = 0.0
    return out


def keep_topk(coeffs: np.ndarray, rate: float) -> np.ndarray:
    """Keep the largest-|.| `rate` fraction of coefficients per row of a
    [rows, K] array (adaptive; indices are side information)."""
    c = coeffs.reshape(coeffs.shape[0], -1)
    k = c.shape[1]
    keep = max(1, int(round(rate * k)))
    thr = np.partition(np.abs(c), k - keep, axis=1)[:, k - keep:k - keep + 1]
    return np.where(np.abs(c) >= thr, c, 0.0)


def _pool_up(seq: np.ndarray, rate: float) -> np.ndarray:
    """Average-pool token blocks of 1/rate and hold — naive baseline."""
    t = seq.shape[0]
    b = max(1, int(round(1.0 / rate)))
    pad = (-t) % b
    if pad:
        seq = np.concatenate([seq, np.repeat(seq[-1:], pad, axis=0)])
    pooled = seq.reshape(-1, b, seq.shape[1]).mean(axis=1)
    return np.repeat(pooled, b, axis=0)[:t]


def compact_sequence(seq: np.ndarray, rate: float,
                     mode: str = "dct_low", block: int = 128) -> tuple[np.ndarray, dict]:
    """Compress-reconstruct [L, H] along the token axis.

    Modes: dct_low | dct_topk (whole-sequence DCT-II) and
    mdct_low | mdct_topk (framed MDCT; falls back to the dct variant
    when the sequence is too short to frame) and pool (average-pool
    baseline). Returns (reconstruction [L, H] float64, stats dict)."""
    seq = np.asarray(seq, dtype=np.float64)
    t, h = seq.shape
    used = mode
    if mode == "pool":
        recon = _pool_up(seq, rate)
        eff = 1.0 / max(1, int(round(1.0 / rate)))
    elif mode.startswith("mdct") and t >= 2 * block:
        coeffs, meta = mdct_analyze(seq, block)          # [F, n/2, H]
        flat = coeffs.reshape(-1, h)                     # [F*n/2, H]
        kept = (keep_topk if mode.endswith("topk") else keep_low)(
            flat.T, rate).T                              # per-channel rows
        recon = mdct_synthesize(kept.reshape(coeffs.shape), meta)
        keep_frac = float(np.count_nonzero(kept) / max(kept.size, 1))
        eff = min(1.0, keep_frac * kept.shape[0] / t)
    else:
        used = {"mdct_topk": "dct_topk", "mdct_low": "dct_low"}.get(mode, mode)
        coeffs = dct2(seq, axis=0)                       # [T, H]
        kept = (keep_topk if used.endswith("topk") else keep_low)(
            coeffs.T, rate).T
        recon = idct2(kept, axis=0)
        keep_frac = float(np.count_nonzero(kept) / max(kept.size, 1))
        eff = min(1.0, keep_frac)
    stats = {
        "mode": used,
        "requested_rate": rate,
        "effective_rate": round(float(eff), 4),
        "rel_err": round(rel_err(recon, seq), 5),
        "row_cos": round(row_cos_mean(recon, seq), 5),
    }
    return recon, stats


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a - b)
    return float(d / max(np.linalg.norm(b), 1e-30))


def row_cos_mean(a: np.ndarray, b: np.ndarray) -> float:
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = den > 1e-30
    return float(np.mean(num[ok] / den[ok])) if ok.any() else 0.0
