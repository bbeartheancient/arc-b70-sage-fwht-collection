# SPDX-License-Identifier: Apache-2.0
"""Qwen-Flash-Next n-gram embedding store (I64 quantized, host-side).

BICAMERAL_FRAMEWORK.md §6.3: the 51B n-gram table is a *store*, not a
compute kernel.  We do NOT write a SYCL I64 dequant kernel — the right
shape is:

  - I64 shard on NVMe (cold; ~360 GB BF16+I64 across 131 shards)
  - Prefetch one shard ahead into host RAM
  - On-demand dequant to fp16, return the requested token block
  - Optional push to GPU1 (per-framework layout: GPU1 is the preprocessor
    + store slot)

I64 layout (per Qwen-FN tech report):
  - 20M n-gram vocabulary × 2560 hidden dim × 8 bytes (int64) = 409.6 TB.
    The "51B params" number is the EFFECTIVE store size after sparse
    n-gram bucketing + per-row I64 codes (Q4-K-style block quant).  For
    framework purposes we treat the table as a function:
        ngram_table[token_block, layer, d] -> fp16[H]
    and the shard files as opaque blake2h-tagged blobs we do not parse
    byte-by-byte.

This module ships:

  1.  ``NgramShard``  — one shard's metadata + lazy-mmap buffer (I64).
  2.  ``NgramStore``  — LRUCache of open shards; prefetch callback API.
  3.  ``dequant_i64_to_fp16``  — vectorized I64 -> fp16 (the only kernel).
  4.  ``fetch_token_block``  — public API: given (ngram_ids, layer),
      return fp16 [N, H] numpy.

Tests (``python/tests/test_igram.py``) cover round-trip on synthetic
shards; real Qwen-FN I64 is verified when the download finishes.

Landmine (BICAMERAL_FRAMEWORK.md §9 #4): do NOT write a SYCL I64
dequant kernel. Host side is the right place.
"""

from __future__ import annotations

import hashlib
import logging
import mmap
import os
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# I64 header magic (little-endian 4-byte ASCII) — markers a Qwen-FN
# I64 shard.  We only honor files starting with this magic so the
# wildcard prefetch does not pick up BF16 vocab shards.
_SHARD_MAGIC = b"N64S"  # "N64 Shard"
_SHARD_HEADER_FMT = "<4sIIIq"  # magic, version, vocab, hidden, layer
_SHARD_HEADER_LEN = struct.calcsize(_SHARD_HEADER_FMT)  # 24 bytes
_DEFAULT_HIDDEN = 2560  # Qwen-FN hidden_size

__all__ = [
    "NgramShard",
    "NgramStore",
    "dequant_i64_to_fp16",
    "fetch_token_block",
    "quantize_fp16_to_i64",
]


# ---------------------------------------------------------------------------
# Dequant kernel
# ---------------------------------------------------------------------------

def dequant_i64_to_fp16(arr_i64: np.ndarray, scale: np.ndarray,
                         zero: np.ndarray) -> np.ndarray:
    """Block-quantized I64 -> fp16 dequant.

    Per-row block quant (Q4-K / Q5-K style):
        x_fp = (x_i64 - zero) * scale

    For Qwen-FN we treat the table as INT8-quantized latent codes:
    each row is stored as I64 (8x storage) to skip the unpack kernel.
    The dequant formula is:
        fp = (int64 - zero) * scale
    where ``scale`` and ``zero`` are per-row fp16/fp32 vectors.  This
    module supports both shared ``scale/zero`` (per-shard) and per-row
    vectors.  The common case is shared.

    Parameters
    ----------
    arr_i64 : [N, H] int64
        Quantized rows.
    scale : [N] or [H] or scalar
        Per-row / per-col multiplier.
    zero : [N] or [H] or scalar
        Per-row / per-col zero point.

    Returns
    -------
    fp : [N, H] float16 numpy
        Dequantized rows.  Caller owns the memory.
    """
    if arr_i64.dtype != np.int64:
        arr_i64 = arr_i64.astype(np.int64, copy=False)
    # Promote to float32 for the math (int64*float32 can overflow
    # in pathological ranges; fp32 has 24 mantissa bits which matches
    # the int8 quant grid comfortably).
    out = arr_i64.astype(np.float32)
    z = np.asarray(zero, dtype=np.float32)
    s = np.asarray(scale, dtype=np.float32)
    # Broadcast scale/zero over the row axis when they are 1D.
    # Per-row [N]: subtract row vector of [N,1]; per-col [H]: subtract [1,H].
    if z.ndim == 1:
        if z.shape[0] == out.shape[0]:
            z = z[:, None]
        elif z.shape[0] == out.shape[1]:
            z = z[None, :]
    if s.ndim == 1:
        if s.shape[0] == out.shape[0]:
            s = s[:, None]
        elif s.shape[0] == out.shape[1]:
            s = s[None, :]
    out = (out - z) * s
    return out.astype(np.float16)


def quantize_fp16_to_i64(arr: np.ndarray, scale: np.ndarray | None = None,
                         zero: np.ndarray | None = None
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """fp16 -> I64 quant for tests / synthetic shards.

    Computes per-row symmetric INT8 quant (scale = max(|row|)/127,
    zero = 0) when no scale/zero provided.  Returns (int64, scale, zero)
    such that ``dequant_i64_to_fp16`` round-trips.
    """
    if arr.dtype != np.float16:
        arr = arr.astype(np.float16)
    a = arr.astype(np.float32)
    N = a.shape[0]
    if scale is None:
        m = np.maximum(np.abs(a).max(axis=1), 1e-12)  # [N]
        scale = (m / 127.0).astype(np.float32)
    if zero is None:
        zero = np.zeros(N, dtype=np.float32)
    # scale/zero are [N]; broadcast to [N, H] for the quant op
    q = np.clip(
        np.round(a / scale[:, None] + zero[:, None]),
        -128, 127
    )
    return q.astype(np.int64), scale, zero


# ---------------------------------------------------------------------------
# Shard I/O
# ---------------------------------------------------------------------------

@dataclass
class NgramShard:
    """One I64 shard on disk.  Lazily mmapped.

    Header layout (24 bytes, little-endian):
        magic[4]   "N64S"
        version    u32
        vocab      u32   # rows in this shard
        hidden     u32   # H
        layer      i64   # target layer id (0..47; -1 = shared)
    Data:  vocab × hidden int64 codes (LE) immediately after the header.
    """

    path: Path
    version: int
    vocab: int
    hidden: int
    layer: int

    @classmethod
    def open(cls, path: Path | str) -> "NgramShard":
        path = Path(path)
        with path.open("rb") as f:
            hdr = f.read(_SHARD_HEADER_LEN)
        if len(hdr) != _SHARD_HEADER_LEN or hdr[:4] != _SHARD_MAGIC:
            raise ValueError(f"not an I64 ngram shard: {path}")
        magic, version, vocab, hidden, layer = struct.unpack(
            _SHARD_HEADER_FMT, hdr)
        return cls(path=path, version=version, vocab=vocab,
                   hidden=hidden, layer=layer)

    def mmap(self) -> mmap.mmap:
        """Read-only mmap of the entire shard (caller must skip header)."""
        fd = self.path.open("rb")
        try:
            return mmap.mmap(fd.fileno(), 0, prot=mmap.PROT_READ)
        finally:
            fd.close()

    @property
    def nbytes(self) -> int:
        return self.vocab * self.hidden * 8 + _SHARD_HEADER_LEN

    def __repr__(self) -> str:
        return (f"NgramShard(path={self.path.name}, layer={self.layer}, "
                f"vocab={self.vocab}, hidden={self.hidden})")


# ---------------------------------------------------------------------------
# LRU store with prefetch
# ---------------------------------------------------------------------------

class NgramStore:
    """LRU cache of open n-gram shards.

    The store keeps at most ``max_shards`` shards resident in host RAM.
    Callers register a ``prefetch`` callback that takes (shard_id) and
    materializes the shard on disk (e.g. from NVMe).  Once materialized,
    the store opens the shard on first access and serves subsequent
    requests from cache.

    The fetch API is the dequantized fp16 block; raw int64 access is
    not exposed.
    """

    def __init__(self, max_shards: int = 4,
                 prefetch: Callable[[int], Path] | None = None) -> None:
        self._cache: OrderedDict[int, tuple[NgramShard, mmap.mmap]] = OrderedDict()
        self._max = max_shards
        self._prefetch = prefetch
        self._hits = self._misses = self._evicts = 0

    def open(self, shard_id: int) -> tuple[NgramShard, mmap.mmap]:
        if shard_id in self._cache:
            self._hits += 1
            self._cache.move_to_end(shard_id)
            return self._cache[shard_id]
        self._misses += 1
        if self._prefetch is None:
            raise FileNotFoundError(f"no prefetch wired for shard {shard_id}")
        path = self._prefetch(shard_id)
        info = NgramShard.open(path)
        new_buf = info.mmap()
        self._cache[shard_id] = (info, new_buf)
        self._cache.move_to_end(shard_id)
        while len(self._cache) > self._max:
            evicted_id, (ev_info, ev_buf) = self._cache.popitem(last=False)
            ev_buf.close()
            self._evicts += 1
            logger.debug("igram: evicted shard %d (%s)", evicted_id, ev_info.path)
        return info, new_buf

    def fetch(self, shard_id: int, row_ids: np.ndarray,
              layer: int = -1) -> np.ndarray:
        """Fetch + dequant one row block to fp16.

        row_ids : [N] int64 token / ngram ids
        Returns : [N, H] float16 numpy
        """
        info, buf = self.open(shard_id)
        if layer >= 0 and info.layer >= 0 and layer != info.layer:
            raise ValueError(f"shard {shard_id} is for layer {info.layer}, "
                             f"requested {layer}")
        H = info.hidden
        N = row_ids.shape[0]
        out = np.empty((N, H), dtype=np.int64)
        for i, rid in enumerate(row_ids):
            off = _SHARD_HEADER_LEN + int(rid) * H * 8
            row = np.frombuffer(buf, dtype="<i8", count=H, offset=off)
            out[i] = row
        # Per-row scales from the shard tail; index only the requested rows.
        tail = self._maybe_read_scales(info, buf)
        if tail is not None:
            raw_scale, raw_zero = tail
            # Index by the actual requested row ids so scale/zero align with out
            rids = row_ids.astype(np.intp).ravel()
            scale = np.asarray(raw_scale[rids], dtype=np.float32)
            zero = np.asarray(raw_zero[rids], dtype=np.float32)
        else:
            scale = np.full(N, 1.0 / 127.0, dtype=np.float32)
            zero = np.zeros(N, dtype=np.float32)
        return dequant_i64_to_fp16(out, scale, zero)

    def _maybe_read_scales(self, info: NgramShard, buf: mmap.mmap
                           ) -> tuple[np.ndarray, np.ndarray] | None:
        """If the shard carries per-row scale/zero at the tail, return
        them; else return None.  Layout: scales[vocab] float32 then
        zeros[vocab] float32 (4*vocab + 4*vocab = 8*vocab extra bytes)."""
        expected_data = _SHARD_HEADER_LEN + info.vocab * info.hidden * 8
        total = len(buf)
        tail = total - expected_data
        if tail >= 8 * info.vocab:
            scale = np.frombuffer(buf, dtype="<f4", count=info.vocab,
                                  offset=expected_data).copy()
            zero = np.frombuffer(buf, dtype="<f4", count=info.vocab,
                                 offset=expected_data + 4 * info.vocab).copy()
            return scale, zero
        return None

    def stats(self) -> dict:
        return {
            "hits": self._hits, "misses": self._misses,
            "evicts": self._evicts,
            "open": list(self._cache.keys()),
        }

    def close(self) -> None:
        for _id, (_info, buf) in self._cache.items():
            try:
                buf.close()
            except Exception:
                pass
        self._cache.clear()


# ---------------------------------------------------------------------------
# Convenience: write a synthetic I64 shard (used by tests + bootstrap
# when real Qwen-FN n-gram shards are not yet on disk)
# ---------------------------------------------------------------------------

def write_synthetic_shard(path: Path | str, vocab: int, hidden: int,
                          layer: int = -1, seed: int = 0,
                          quant: bool = True) -> Path:
    """Generate a synthetic I64 n-gram shard (header + int64 codes +
    optional scale/zero tail).  Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if quant:
        # Quantize random fp16 rows to int8-grid int64 (round-trip safe)
        rows = rng.standard_normal((vocab, hidden)).astype(np.float16)
        codes, scale, zero = quantize_fp16_to_i64(rows)
    else:
        codes = rng.integers(-128, 128, size=(vocab, hidden), dtype=np.int64)
        scale = np.full(vocab, 1.0 / 127.0, dtype=np.float32)
        zero = np.zeros(vocab, dtype=np.float32)
    with path.open("wb") as f:
        f.write(struct.pack(_SHARD_HEADER_FMT, _SHARD_MAGIC, 1,
                            vocab, hidden, layer))
        f.write(codes.astype("<i8").tobytes())
        if quant:
            f.write(scale.astype("<f4").tobytes())
            f.write(zero.astype("<f4").tobytes())
    return path


# ---------------------------------------------------------------------------
# Public API: fetch_token_block
# ---------------------------------------------------------------------------

_DEFAULT_STORE: NgramStore | None = None


def _default_prefetch(shard_id: int) -> Path:
    """Default prefetch: look in FABRIC_IGRAM_DIR or ~/QwenFlash/ngram."""
    base = Path(os.environ.get("FABRIC_IGRAM_DIR",
                                str(Path.home() / "QwenFlash" / "ngram")))
    return base / f"shard-{shard_id:05d}.n64s"


def fetch_token_block(ngram_ids: np.ndarray, layer: int = -1,
                      shard_id: int = 0) -> np.ndarray:
    """Fetch one n-gram token block as fp16.

    Parameters
    ----------
    ngram_ids : [N] int (token / ngram ids)
    layer : target layer id (-1 = shared vocab)
    shard_id : which shard to open (default 0; real Qwen-FN will
        require token-id → shard routing)

    Returns
    -------
    fp16 [N, H] numpy
    """
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = NgramStore(max_shards=4, prefetch=_default_prefetch)
    ids = np.asarray(ngram_ids, dtype=np.int64).ravel()
    return _DEFAULT_STORE.fetch(shard_id, ids, layer=layer)
