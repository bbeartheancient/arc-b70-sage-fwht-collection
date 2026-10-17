# SPDX-License-Identifier: Apache-2.0
"""Lazy BF16 n-gram shard loader for Qwen3.8-Flash-Next PLE.

The full n-gram table is 128 shards × (2,500,012, 160) BF16 = 102.4 GB
on NVMe (model.language_model.layers.1.ple.ple_embedding.ngram_embedding.
shard_{0..127}.weight).  This loader:

  * mmaps each shard lazily (no upfront read)
  * keeps a small LRU of in-RAM shards (default 10 × 800 MB = 8 GB)
  * returns (ngram_ids → fp16[B, 160]) numpy arrays via gather
  * on miss, evicts the coldest shard and mmap's the new one (mmap is
    cheap; the OS pages in on first touch)

This drops the HBM cost from 102 GB → ~0 GB (only the host page cache
holds hot rows; GPU sees the gathered rows only after the projection
in ``BicameralNgramEmbedding.forward``).

Lookup math
-----------
ngram_vocab_size_base = 20,000,000 (vocab the model EXPECTS to look up)
stored rows per shard = 2,500,012 = 20M / 8 (so 12.5% of n-grams
physically have weights; the rest return zero).  Splitting:
  shard_id = token_id // 2_500_012           (128 shards, 0..127)
  row_id   = token_id %  2_500_012           (per-shard row)

The forward in ``BicameralNgramEmbedding`` does
  ``trigrams[token_id, :] = shard[shard_id][row_id, :]``  if row_id
  < 2_500_012 else zero.  12.5% zero-return is the published
  trade-off (sparse n-gram table; see tech report §2.3).
"""
from __future__ import annotations

import logging
import mmap
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# Layout constants (from config.json):
#   ngram_vocab_size_base = 20_000_000
#   split_ngram_parts     = 128
#   ngram_embed_dim       = 160
#
# Actual stored rows = 2_500_012 = 20M // 8 (rounded up to 128).
_NGRAM_VOCAB_BASE = 20_000_000
_SPLIT_PARTS = 128
_ROWS_PER_SHARD = 2_500_012          # observed on disk
_NGRAM_EMBED_DIM = 160
_DTYPE_BYTES = 2                      # bf16
_SHARD_BYTES = _ROWS_PER_SHARD * _NGRAM_EMBED_DIM * _DTYPE_BYTES  # 800_003_840


def _bf16_to_fp16(arr: np.ndarray) -> np.ndarray:
    """Convert bf16 bit-pattern array (int16) to fp16 float array.

    Two-step via torch (single int16 reinterpret, no Python arithmetic):
      int16 bit-pattern → torch bfloat16 → torch float16 → numpy float16.
    Vectorised; one shot per buffer slice.
    """
    import torch
    return (
        torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
        .to(torch.float16)
        .numpy()
    )


@dataclass
class NgramShardInfo:
    """One shard's location."""
    shard_id: int
    safetensors_file: str
    safetensors_key: str
    path: Path
    rows: int = _ROWS_PER_SHARD
    dim: int = _NGRAM_EMBED_DIM
    dtype: str = "bfloat16"
    data_offset: int = 0           # byte offset into safetensors file
    data_size: int = 0             # byte size of this tensor's data

    @property
    def nbytes(self) -> int:
        if self.data_size:
            return self.data_size
        return self.rows * self.dim * _DTYPE_BYTES


class LazyNgramLoader:
    """mmap-based lazy loader for the 128 n-gram BF16 shards.

    Parameters
    ----------
    model_dir : path
        Directory containing the 131 safetensors files.  We discover the
        n-gram shard paths from ``model.safetensors.index.json``.
    max_resident_shards : int
        LRU size in shards (each ~800 MB).  Default 10 = 8 GB.
    mmap_huge : bool
        If True, use MAP_HUGETLB on Linux.  Requires CAP_IPC_LOCK or
        enough hugepages.  Default False (mmap on file pages is fine;
        OS pages in on first touch).
    """

    def __init__(
        self,
        model_dir: str | os.PathLike,
        max_resident_shards: int = 10,
        mmap_huge: bool = False,
    ):
        self.model_dir = Path(model_dir)
        self.max_resident = max_resident_shards
        self.mmap_huge = mmap_huge
        self._infos: dict[int, NgramShardInfo] = {}
        self._resident: OrderedDict[int, tuple[NgramShardInfo, mmap.mmap]] = OrderedDict()
        self._load_index()
        logger.info(
            "LazyNgramLoader: %d shards, max_resident=%d (%.1f GB)",
            len(self._infos),
            self.max_resident,
            self.max_resident * _SHARD_BYTES / 1e9,
        )

    # ------------------------------------------------------------------
    # index discovery
    # ------------------------------------------------------------------
    def _load_index(self) -> None:
        """Parse ``model.safetensors.index.json`` and register n-gram shards."""
        import json

        index_path = self.model_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"missing {index_path}")
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        # Discover per-shard byte offsets by reading each safetensors
        # header once.  We only need data_offsets for n-gram tensors;
        # group by file so we open each file's header at most once.
        files_seen: dict[str, dict[str, tuple[int, int]]] = {}
        for name, shard_file in weight_map.items():
            if "ngram_embedding.shard_" not in name:
                continue
            shard_id = int(name.split("shard_")[1].split(".")[0])
            if shard_id in self._infos:
                continue
            if shard_file not in files_seen:
                files_seen[shard_file] = self._read_ngram_offsets(
                    self.model_dir / shard_file
                )
            offset_infos = files_seen[shard_file]
            if name not in offset_infos:
                continue
            data_offset, data_size = offset_infos[name]
            self._infos[shard_id] = NgramShardInfo(
                shard_id=shard_id,
                safetensors_file=shard_file,
                safetensors_key=name,
                path=self.model_dir / shard_file,
                data_offset=data_offset,
                data_size=data_size,
            )

    @staticmethod
    def _read_ngram_offsets(path: Path) -> dict[str, tuple[int, int]]:
        """Return {ngram_tensor_name: (data_offset_abs, data_size)} for the
        n-gram tensors in a safetensors file.  Opens the file once to
        read the JSON header; we do NOT touch the data region.

        Offsets in the header are RELATIVE to the start of the data
        section (after ``8 + header_size`` bytes), so we add that base.
        Returns (absolute_offset, byte_size).
        """
        import json
        out: dict[str, tuple[int, int]] = {}
        with open(path, "rb") as f:
            header_size_bytes = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(header_size_bytes))
        base = 8 + header_size_bytes
        for name, info in header.items():
            if not isinstance(info, dict):
                continue
            if "ngram_embedding.shard_" not in name:
                continue
            data_offsets = info.get("data_offsets")
            if data_offsets is None:
                continue
            start, end = data_offsets
            abs_start = base + start
            size = end - start
            out[name] = (abs_start, size)
        return out

    # ------------------------------------------------------------------
    # resident management
    # ------------------------------------------------------------------
    def _ensure_resident(self, shard_id: int) -> tuple[mmap.mmap, int]:
        """Make sure ``shard_id`` is mmapped; return (buffer, data_offset).

        We mmap the WHOLE safetensors file (not a slice); the buffer
        can then be sliced at ``data_offset`` to reach the n-gram
        tensor's bytes.  mmap is a private view; the OS pages in only
        the pages we actually touch.
        """
        if shard_id in self._resident:
            self._resident.move_to_end(shard_id)
            return self._resident[shard_id][1]
        info = self._infos[shard_id]
        if not info.path.exists():
            raise FileNotFoundError(f"shard {shard_id} not at {info.path}")
        # mmap the whole file; data_offset is the byte where the
        # n-gram tensor's bf16 rows begin.
        fd = os.open(info.path, os.O_RDONLY)
        try:
            buf = mmap.mmap(
                fd,
                length=0,                 # 0 = whole file
                access=mmap.ACCESS_READ,
                offset=0,
            )
        finally:
            os.close(fd)
        # Evict cold shards
        while len(self._resident) >= self.max_resident:
            cold_id, (cold_info, cold_pair) = self._resident.popitem(last=False)
            cold_buf = cold_pair[0]  # (buf, data_offset)
            cold_buf.close()
            logger.debug("evicted n-gram shard %d (%s)", cold_id, cold_info.path.name)
        self._resident[shard_id] = (info, (buf, info.data_offset))
        logger.debug("resident +ngram shard %d (%s, off=%d)", shard_id, info.path.name, info.data_offset)
        return (buf, info.data_offset)

    def _read_row(self, info: NgramShardInfo, row: int) -> bytes:
        """Read ONE 320-byte row from disk via os.pread.  Avoids
        mmap page-cache pollution when we only need a few rows.

        We open the file per-shard (cached fd) and use os.pread
        (positional read) which doesn't move the file offset.
        """
        fd = self._open_fd(info)
        offset = info.data_offset + row * (_NGRAM_EMBED_DIM * _DTYPE_BYTES)
        return os.pread(fd, _NGRAM_EMBED_DIM * _DTYPE_BYTES, offset)

    def _open_fd(self, info: NgramShardInfo) -> int:
        """Open (and cache) a file descriptor for ``info.path``."""
        if not hasattr(self, "_fds"):
            self._fds = {}
        fd = self._fds.get(info.path)
        if fd is not None:
            return fd
        fd = os.open(info.path, os.O_RDONLY)
        self._fds[info.path] = fd
        return fd

    def close(self) -> None:
        for _, pair in self._resident.values():
            buf = pair[0]
            buf.close()
        self._resident.clear()
        if hasattr(self, "_fds"):
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def lookup(
        self,
        token_ids: np.ndarray,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Look up n-gram rows for ``token_ids`` (uint32 / int64).

        Returns an fp16 array of shape (len(token_ids), NGRAM_EMBED_DIM).
        Tokens outside the stored range (>= 2_500_012 * 128 = 320_001_536)
        are zero-filled (sparse n-gram table).
        """
        if not isinstance(token_ids, np.ndarray):
            token_ids = np.asarray(token_ids)
        token_ids = token_ids.astype(np.int64, copy=False).reshape(-1)
        n = token_ids.shape[0]
        if out is None:
            out = np.zeros((n, _NGRAM_EMBED_DIM), dtype=np.float16)
        else:
            assert out.shape == (n, _NGRAM_EMBED_DIM), f"bad out shape {out.shape}"
            assert out.dtype == np.float16
            out.fill(0)

        # Group by shard to minimize mmap faults.
        # shard_id = token_id // ROWS_PER_SHARD  (clip to 0..127)
        shard_of_token = token_ids // _ROWS_PER_SHARD
        in_range = shard_of_token < _SPLIT_PARTS
        if not in_range.any():
            return out
        valid_ids = token_ids[in_range]
        valid_shards = shard_of_token[in_range]
        row_in_shard = (valid_ids % _ROWS_PER_SHARD).astype(np.int64)

        # Per-shard gather
        out_indices_all = np.where(in_range)[0]
        unique_shards = np.unique(valid_shards)
        row_bytes = _NGRAM_EMBED_DIM * _DTYPE_BYTES
        for shard_id in unique_shards:
            mask = valid_shards == shard_id
            rows_here = row_in_shard[mask]
            out_indices = out_indices_all[mask]
            info = self._infos[int(shard_id)]
            # Build a contiguous buffer of just the rows we need via
            # one os.pread per row.  This bypasses the page cache for
            # the rows we don't touch and avoids the 320 MB minimum
            # fault that mmap slicing incurs.
            fd = self._open_fd(info)
            n = rows_here.shape[0]
            buf_int16 = np.empty((n, _NGRAM_EMBED_DIM), dtype=np.int16)
            for i, r in enumerate(rows_here):
                off = info.data_offset + int(r) * row_bytes
                got = os.pread(fd, row_bytes, off)
                if len(got) != row_bytes:
                    # OOB: zero out this row
                    buf_int16[i, :] = 0
                else:
                    buf_int16[i, :] = np.frombuffer(got, dtype=np.int16)
            flat = _bf16_to_fp16(buf_int16)
            out[out_indices] = flat
        return out

    def resident_shard_ids(self) -> list[int]:
        return list(self._resident.keys())

    def resident_bytes(self) -> int:
        return sum(pair[0].nbytes for pair in self._resident.values())

    def summary(self) -> str:
        return (
            f"LazyNgramLoader: {len(self._infos)} shards total, "
            f"{len(self._resident)} resident ({self.resident_bytes()/1e9:.2f} GB)"
        )


__all__ = [
    "LazyNgramLoader",
    "NgramShardInfo",
    "_NGRAM_VOCAB_BASE",
    "_SPLIT_PARTS",
    "_ROWS_PER_SHARD",
    "_NGRAM_EMBED_DIM",
]
