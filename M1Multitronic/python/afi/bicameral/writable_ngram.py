# SPDX-License-Identifier: Apache-2.0
"""Writable n-gram store backed by memvid .mv2 shards.

Architecture (docs/27b-writable-ngram-plan.md § Step 1):

  Each n-gram shard is one .mv2 file, one per full-attention layer
  that has PLE.  The 27B has 16 full-attention layers (at positions
  3, 7, 11, ..., 63 in 0-indexed layers); Qwen-FN has 1.

  Frame layout per n-gram row (one MV2 frame):
    uri       : mv2://{layer_id}/{token_id}
    title    : f"l{layer_id}_t{token_id}_p{trigram_pos}"
    text     : base64(bf16_vector)           — 320 bytes → ~430 chars base64
    tags     : ["layer:{layer}", "token:{token}", "pos:{pos}"]
    metadata : {write_count, seen_at, graph_id?}   ("last_seen" is a
                 reserved memvid key that breaks the lex index)

  Indexes:
    lex  — on uri + tags (token prefix lookup: O(1))
    vec  — on the 160-dim bf16 embedding (k-NN neighbors)

  Write path (EEL2 ngram_store verb):
    put(..., uri, text=base64, tags, metadata)
    WAL append → session_commit() flushes to disk

  Read path (EEL2 ngram_lookup verb):
    find(tags=["layer:N", "token:M"]) → lex hit → decode base64 → np.float16

  MV2 file path:
    ~/runtime/.ngram_shards/qwen27b_ple_layer_{N}.mv2
  (separate from the model dir; gitignored)

Dependencies:
  memvid-sdk 2.x (pip install memvid-sdk)
  numpy, torch (already available in vLLM venv)

Shape contract:
  n_gram_dim = 160  (matches Qwen-FN PLE; the 27B uses the same
    ple_embed_dim=160 in Qwen-FN config; for the 27B the n-gram
    projection is trained separately so this is a design choice)
  n_trigram_pos = 8  (8 buckets per token_id, same as Qwen-FN)
  Stored rows: token_id in [0, 248320)  (27B vocab size)

Key decisions:
  - base64 text for the bf16 vector (no custom binary parser needed;
    the text frame IS the payload; MV2 WAL handles it)
  - One .mv2 per layer (16 files for the 27B) so lex lookups are
    scoped to one file at a time
  - write_count / seen_at tracked in metadata for cache eviction
  - vec index for k-NN neighbor lookups (for graph traversal)
"""
from __future__ import annotations

import base64
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Layout constants
_NGRAM_DIM = 160          # ple_embed_dim (Qwen-FN spec; fixed for now)
_N_TRIGRAM_POS = 8         # n-gram bucket per token position
_VOCAB_SIZE = 248320        # Qwen3.8-27B vocab size
_BF16_BYTES = _NGRAM_DIM * 2  # 320 bytes per row
_LITERAL_FRAME_SIZE = 640   # 320 bytes × 2 (W+X) = 640 for h4 2-edge store


def _is_capacity(exc: Exception) -> bool:
    """memvid MV001 (CapacityExceededError or the string code)."""
    return type(exc).__name__ == "CapacityExceededError" or "MV001" in str(exc)


@dataclass
class NgramRow:
    """One n-gram row's decoded contents."""
    token_id: int
    trigram_pos: int        # 0..7 (which of the 8 buckets)
    layer_id: int
    vector: np.ndarray       # (NGRAM_DIM,) float16
    write_count: int = 1
    last_seen: float = field(default_factory=time.time)
    graph_id: str | None = None

    @property
    def uri(self) -> str:
        return f"mv2://{self.layer_id}/{self.token_id}/{self.trigram_pos}"

    def to_base64(self) -> str:
        arr = np.asarray(self.vector, dtype=np.float16)
        return base64.b64encode(arr.tobytes()).decode("ascii")

    @classmethod
    def from_base64(cls, token_id: int, trigram_pos: int, layer_id: int,
                     b64: str, metadata: dict) -> "NgramRow":
        raw = base64.b64decode(b64)
        vec = np.frombuffer(raw, dtype=np.float16).copy()
        return cls(
            token_id=token_id,
            trigram_pos=trigram_pos,
            layer_id=layer_id,
            vector=vec,
            write_count=metadata.get("write_count", 1),
            last_seen=metadata.get("seen_at", time.time()),
            graph_id=metadata.get("graph_id"),
        )


# ---------------------------------------------------------------------------
# Singleton per-layer store
# ---------------------------------------------------------------------------

_lock = threading.RLock()
_handles: dict[int, "WritableNgramStore"] = {}


def _default_shard_dir() -> Path:
    """~/.runtime/ngram_shards/ (gitignored, NVMe-preferred)."""
    base = Path(os.environ.get(
        "NGRAM_SHARD_DIR",
        Path.home() / ".runtime" / "ngram_shards",
    ))
    base.mkdir(parents=True, exist_ok=True)
    return base


def open_store(layer_id: int, *, shard_dir: Path | None = None,
               read_only: bool = False) -> "WritableNgramStore":
    """Open (or create) the MV2 shard for ``layer_id``.

    Thread-safe singleton per layer — subsequent calls with the same
    layer_id return the cached handle.
    """
    shard_dir = shard_dir or _default_shard_dir()
    with _lock:
        if layer_id in _handles:
            return _handles[layer_id]
        store = WritableNgramStore(layer_id, shard_dir, read_only=read_only)
        _handles[layer_id] = store
        return store


# ---------------------------------------------------------------------------
# Core store
# ---------------------------------------------------------------------------

class WritableNgramStore:
    """One MV2 shard for a single PLE layer's n-gram table.

    Parameters
    ----------
    layer_id   : full-attention layer index (0..63 for 27B; 0..47 for Qwen-FN)
    shard_dir  : directory for .mv2 files
    read_only  : if True, open existing shard without write locks

    Write path (per EEL2 ngram_store call):
      store([(token_id, trigram_pos, vector)], metadata)
      → put(..., uri=text(...), text=base64, tags=[...], metadata={...})
      → WAL append (no flush until session_commit())

    Read path (per EEL2 ngram_lookup call):
      lookup([token_id, ...]) → find(tags=[layer:N, token:M])
      → lex hit → decode base64 → NgramRow(vector)
      → value_proj(vector) → residual delta
    """

    def __init__(self, layer_id: int, shard_dir: Path, read_only: bool = False):
        self.layer_id = layer_id
        self.shard_dir = Path(shard_dir)
        self.read_only = read_only
        self._mv: object | None = None
        # Session write cache: rows stored since open, keyed
        # (token_id, trigram_pos).  Read-your-writes must NOT depend
        # on memvid's lex index — WAL rows are visible immediately,
        # and the lex index degrades in-process after many shard
        # open/close cycles (MV004 / silent misses; on-disk fine).
        self._cache: dict[tuple[int, int], NgramRow] = {}
        self._open()

    def _open(self) -> None:
        import memvid_sdk as m
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        path = self.shard_dir / f"qwen27b_ple_layer_{self.layer_id}.mv2"
        # LANDMINE: enable_vec MUST stay False.  Calling bare
        # enable_lex() (or create/use with defaults) also turns the
        # vec index on with an "unknown" embedder; embedding attempts
        # on arbitrary base64 payloads then poison the shard's lex
        # index IN-PROCESS (find raises MV004 LexIndexDisabledError,
        # unrecoverable until process restart; on-disk data is fine).
        if not path.exists():
            self._mv = m.create(str(path), enable_vec=False, enable_lex=True)
            logger.info("created n-gram shard %s", path)
        else:
            self._mv = m.use("basic", str(path),
                             enable_vec=False, enable_lex=True)
            logger.info("opened n-gram shard %s", path)

    def _uri(self, token_id: int, trigram_pos: int) -> str:
        return f"mv2://{self.layer_id}/{token_id}/{trigram_pos}"

    def _tags(self, token_id: int, trigram_pos: int) -> list[str]:
        return [
            f"layer:{self.layer_id}",
            f"token:{token_id}",
            f"pos:{trigram_pos}",
        ]

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------
    def store(self, rows: list[tuple[int, int, np.ndarray]],
               metadata: dict | None = None,
               ) -> None:
        """Write one or more n-gram rows (WAL append, no flush).

        ``rows`` is [(token_id, trigram_pos, vector)], where vector is
        (NGRAM_DIM,) float16.

        Args
        ----
        rows     : list of (token_id, trigram_pos, vector)
        metadata : extra fields written into each frame's metadata dict
        """
        if self.read_only or self._mv is None:
            return
        meta = dict(metadata or {})
        write_count = int(meta.get("write_count", 1))
        graph_id = meta.get("graph_id")
        # LANDMINE: ANY non-empty memvid ``metadata`` dict intermittently
        # breaks the lex index (find() raises LexIndexDisabledError /
        # MV004 and never recovers, even after reopen + enable_lex;
        # ~20-100% depending on keys — "last_seen" and string-valued
        # timestamps are the worst).  Bookkeeping goes in TAGS instead
        # (wc:/seen:/graph:), which are indexed cleanly and returned
        # verbatim on find hits.
        bookkeeping = [f"wc:{write_count}", f"seen:{int(time.time())}"]
        if graph_id:
            bookkeeping.append(f"graph:{graph_id}")
        for token_id, trigram_pos, vec in rows:
            row = NgramRow(
                token_id=token_id,
                trigram_pos=trigram_pos,
                layer_id=self.layer_id,
                vector=np.asarray(vec, dtype=np.float16),
                write_count=write_count,
                graph_id=graph_id,
            )
            self._cache[(token_id, trigram_pos)] = row
            b64 = row.to_base64()
            try:
                self._put(b64, token_id, trigram_pos, bookkeeping)
            except Exception as exc:  # noqa: BLE001
                if _is_capacity(exc) and self._rotate():
                    # Fresh shard with reset capacity; retry this row once.
                    try:
                        self._put(b64, token_id, trigram_pos, bookkeeping)
                        continue
                    except Exception as exc2:  # noqa: BLE001
                        exc = exc2
                logger.warning("ngram store failed for t=%d p=%d: %s",
                               token_id, trigram_pos, exc)

    def _put(self, b64: str, token_id: int, trigram_pos: int,
             bookkeeping: list[str]) -> None:
        self._mv.put(
            title=f"l{self.layer_id}_t{token_id}_p{trigram_pos}",
            text=b64,
            uri=self._uri(token_id, trigram_pos),
            tags=self._tags(token_id, trigram_pos) + bookkeeping,
            metadata={},
        )

    def _rotate(self) -> bool:
        """Archive the full shard and start a fresh one (MV001 recovery).

        memvid's 50 MB capacity is baked into the native core (free
        tier; raising it needs a dashboard ticket).  Rotation is the
        local fix: close, rename to ``*.full-N``, create a new shard.
        Rows stranded in the archive are acceptable — the npz sidecars
        are the operational read cache and the PLE buffer rebuilds its
        table per span (replace mode)."""
        path = self.path()
        try:
            self.close()
            n = 1
            while path.with_suffix(f".mv2.full-{n}").exists():
                n += 1
            path.rename(path.with_suffix(f".mv2.full-{n}"))
            wal = path.parent / (path.stem + ".manifest.wal")
            if wal.exists():
                wal.rename(wal.with_suffix(f".wal.full-{n}"))
            self._open()
            logger.warning("ngram shard %d rotated (MV001) -> .full-%d",
                           self.layer_id, n)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("ngram shard %d rotation failed: %s",
                           self.layer_id, exc)
            return False

    def increment_write_count(self, token_id: int, trigram_pos: int) -> int:
        """Increment the write_count for an existing row; returns new count."""
        hit = self._find_one(token_id, trigram_pos)
        if hit is None:
            return 1
        new_count = (hit.write_count or 1) + 1
        # Re-write with updated count
        self.store([(token_id, trigram_pos, hit.vector)],
                   metadata={"write_count": new_count})
        return new_count

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------
    def _find_one(self, token_id: int,
                   trigram_pos: int) -> NgramRow | None:
        """Lex-lookup one specific row; returns None if not found."""
        # Session write cache first: read-your-writes independent of
        # the lex index (see the cache note in __init__).
        cached = self._cache.get((token_id, trigram_pos))
        if cached is not None:
            return cached
        if self._mv is None:
            return None
        # LANDMINE: memvid lex engine mishandles multi-term tag
        # queries ("layer:N token:M pos:P" returns duplicate/missing
        # hits).  URI-prefix queries are exact and reliable — the
        # URI is unique per (layer, token, pos).
        try:
            res = self._mv.find(self._uri(token_id, trigram_pos), k=5)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ngram lookup failed for t=%d p=%d: %s",
                           token_id, trigram_pos, exc)
            return None
        hits = res.get("hits", []) if isinstance(res, dict) else []
        hit = None
        for h in hits:
            if h.get("uri") == self._uri(token_id, trigram_pos):
                hit = h
                break
        if hit is None:
            return None
        b64 = hit.get("text", "")
        if not b64:
            return None
        try:
            raw = base64.b64decode(b64)
            vec = np.frombuffer(raw, dtype=np.float16).copy()
        except Exception:  # noqa: BLE001
            return None
        # Bookkeeping lives in tags (see the LANDMINE note in store()).
        write_count = 1
        last_seen = time.time()
        graph_id = None
        for tag in hit.get("tags", []):
            if tag.startswith("wc:"):
                try:
                    write_count = int(tag[3:])
                except ValueError:
                    pass
            elif tag.startswith("seen:"):
                try:
                    last_seen = float(tag[5:])
                except ValueError:
                    pass
            elif tag.startswith("graph:"):
                graph_id = tag[6:]
        row = NgramRow(
            token_id=token_id,
            trigram_pos=trigram_pos,
            layer_id=self.layer_id,
            vector=vec,
            write_count=write_count,
            last_seen=last_seen,
            graph_id=graph_id,
        )
        self._cache[(token_id, trigram_pos)] = row
        return row

    def lookup(self, token_ids: np.ndarray,
               trigram_pos: int = 0) -> np.ndarray:
        """Look up n-gram vectors for ``token_ids`` (S,).

        Returns (S, NGRAM_DIM) float16 array.  OOV tokens (token_id
        outside [0, VOCAB_SIZE)) or missing rows → zeros.

        Performance: one lex find per token (k=1).  For batch access
        (many tokens), use lookup_batch() which uses tag filtering.
        """
        S = token_ids.shape[0]
        out = np.zeros((S, _NGRAM_DIM), dtype=np.float16)
        for i, tid in enumerate(token_ids):
            row = self._find_one(int(tid), trigram_pos)
            if row is not None:
                out[i] = row.vector
        return out

    def lookup_batch(self, token_ids: np.ndarray,
                     trigram_pos: int = 0) -> np.ndarray:
        """Look up many n-gram vectors at once.

        LANDMINE: the broad tag-filter find ("layer:N pos:P") is broken
        in memvid's lex index (duplicate hits, missing frames).  URI
        prefix queries are exact, so we issue one per unique token.
        The lex lookup is ~0.3 ms/hit, so 2048 tokens ≈ a few hundred
        ms worst case — acceptable for the write-through path.
        """
        S = token_ids.shape[0]
        out = np.zeros((S, _NGRAM_DIM), dtype=np.float16)
        cache: dict[int, NgramRow | None] = {}
        for i, tid in enumerate(token_ids):
            tid = int(tid)
            if tid not in cache:
                cache[tid] = self._find_one(tid, trigram_pos)
            row = cache[tid]
            if row is not None:
                out[i] = row.vector
        return out

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def commit(self) -> None:
        """Flush WAL to disk (call at session end)."""
        if self._mv is None:
            return
        try:
            self._mv.commit()
            logger.debug("ngram shard %d committed", self.layer_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ngram commit failed for layer %d: %s",
                          self.layer_id, exc)

    def close(self) -> None:
        """Flush + close the shard."""
        if self._mv is None:
            return
        self.commit()
        try:
            self._mv.close()
        except Exception:  # noqa: BLE001
            pass
        self._mv = None

    def stats(self) -> dict:
        """Return shard stats (frame count, size)."""
        if self._mv is None:
            return {}
        try:
            st = self._mv.stats()
            return dict(st) if st else {}
        except Exception:  # noqa: BLE001
            return {}

    def path(self) -> Path:
        return self.shard_dir / f"qwen27b_ple_layer_{self.layer_id}.mv2"

    def __repr__(self) -> str:
        st = self.stats()
        frames = st.get("frame_count", "?")
        size = st.get("size_bytes", 0)
        return (f"WritableNgramStore(layer={self.layer_id}, "
                f"frames={frames}, size={size})")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# EEL2-friendly batch interface
# ---------------------------------------------------------------------------

def ngram_lookup_via_mv2(
    token_ids: np.ndarray,
    layer_id: int,
    trigram_pos: int = 0,
    shard_dir: Path | None = None,
) -> np.ndarray:
    """EEL2 ngram_lookup: read n-gram rows from MV2.

    Thin wrapper around WritableNgramStore.lookup_batch for use as
    an EEL2 builtin.  The token_ids are (S,) int64; returns (S, 160)
    float16.

    Performance: ~1-3 ms for 2048 tokens (lex find over ~2k rows).
    """
    with _lock:
        store = open_store(layer_id, shard_dir=shard_dir)
    return store.lookup_batch(token_ids, trigram_pos)


def ngram_store_via_mv2(
    token_ids: np.ndarray,
    vectors: np.ndarray,
    layer_id: int,
    trigram_pos: int = 0,
    shard_dir: Path | None = None,
    metadata: dict | None = None,
) -> None:
    """EEL2 ngram_store: write n-gram rows to MV2 (WAL append).

    Writes one MV2 frame per token_id.  No flush until session end.
    """
    rows = [
        (int(t), trigram_pos, vectors[i])
        for i, t in enumerate(token_ids)
    ]
    with _lock:
        store = open_store(layer_id, shard_dir=shard_dir)
    store.store(rows, metadata=metadata)


def session_commit(shard_dir: Path | None = None) -> None:
    """Flush all open n-gram stores."""
    with _lock:
        for store in _handles.values():
            store.commit()


def close_all() -> None:
    """Close every open store and clear the singleton cache.

    LANDMINE: leaving many .mv2 handles open in one process (e.g. a
    pytest session creating per-test shards) intermittently breaks
    memvid's lex index on NEW shards — find() raises MV004
    (LexIndexDisabledError) even though enable_lex succeeded.
    Tests must close_all() on teardown, not just clear _handles.
    """
    with _lock:
        for store in _handles.values():
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass
        _handles.clear()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "WritableNgramStore",
    "NgramRow",
    "open_store",
    "ngram_lookup_via_mv2",
    "ngram_store_via_mv2",
    "session_commit",
    "close_all",
    "_NGRAM_DIM",
    "_N_TRIGRAM_POS",
    "_VOCAB_SIZE",
]
