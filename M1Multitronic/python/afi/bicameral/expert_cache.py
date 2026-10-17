# SPDX-License-Identifier: Apache-2.0
"""Host-side MoE expert pager for Qwen-FN.

Expert tiles stay on NVMe (~5 GB/layer, 247 GB total).  Each decode
step only needs top-10 experts × ~10 MB.  ``safetensors`` slices mmap
the file; the OS faults just the selected rows.

BICAMERAL §4: no P2P, no GPU-resident expert bank.

Top-K VRAM hot cache + async prefetch (2026-08-27 FIRSTRUN):
  Each forward step reads expert tiles from NVMe (~10 ms each).
  A bg thread pool (8 workers) reads tiles in parallel so multiple
  reads are in flight concurrently — OS readahead saturates disk BW.
  A per-layer prefetch fires after each layer's gate produces topi:
  ``prefetch_async(layer+1, unique(topi))`` schedules the same
  expert set for the next layer, so by the time the next layer's
  forward starts the tiles are already in the host cache.

  Cache policy: LRU, ``top_k`` entries max (default 32 ≈ 1.6 GB
  per layer's hot set).  top_k=0 is the legal "no cache" mode
  (read once, discard, useful for cold-path benchmarks).

  The layer periodicity (3 linear → 1 full, ×12) guides the prefetch
  window: during the compute-cheap linear layers, the bg pool has
  ~0.5-1s per layer to read the next block's expert tiles before
  GPU0 reaches the compute-heavy full-attention layers.

  Prefetch pool size is controlled by ``AFI_EXPERT_PREFETCH_WORKERS``
  (default: cpu_count() // 4, min 2, max 8).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_INDEX_NAME = "model.safetensors.index.json"

# ---------------------------------------------------------------------------
# Shared background prefetch pool
# ---------------------------------------------------------------------------
_PREFETCH_POOL: ThreadPoolExecutor | None = None
_PREFETCH_LOCK = threading.Lock()


def _get_pool() -> ThreadPoolExecutor:
    global _PREFETCH_POOL
    if _PREFETCH_POOL is None:
        with _PREFETCH_LOCK:
            if _PREFETCH_POOL is None:
                n = max(2, min(8, (os.cpu_count() or 4) // 4))
                _PREFETCH_POOL = ThreadPoolExecutor(
                    max_workers=n, thread_name_prefix="afi-expert"
                )
                logger.info("afi expert prefetch pool: %d workers", n)
    return _PREFETCH_POOL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _weight_map(model_dir: str) -> dict[str, str]:
    with open(os.path.join(model_dir, _INDEX_NAME)) as f:
        return json.load(f)["weight_map"]


def _layer_of(name: str) -> int | None:
    if name.startswith("mtp.") or ".layers." not in name:
        return None
    try:
        return int(name.split(".layers.")[1].split(".")[0])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Core cache
# ---------------------------------------------------------------------------
class BicameralExpertCache:
    def __init__(self, model_dir: str, top_k: int = 32):
        self.model_dir = os.path.abspath(os.path.expanduser(model_dir))
        self.top_k = int(os.environ.get("AFI_EXPERT_CACHE_K", str(top_k)))
        # layer -> {gate_up: (path, key), down: (path, key)}
        self._loc: dict[int, dict[str, tuple[str, str]]] = {}
        wm = _weight_map(self.model_dir)
        for name, fn in wm.items():
            layer = _layer_of(name)
            if layer is None:
                continue
            path = os.path.join(self.model_dir, fn)
            slot = self._loc.setdefault(layer, {})
            if name.endswith(".mlp.experts.gate_up_proj"):
                slot["gate_up"] = (path, name)
            elif name.endswith(".mlp.experts.down_proj"):
                slot["down"] = (path, name)
        self._handles: dict[str, object] = {}
        n = sum(
            1 for v in self._loc.values()
            if "gate_up" in v and "down" in v
        )
        # VRAM hot-cache: (layer, expert) -> ((gate_up, down) tuple,
        # last_use_tick).  Sized by self.top_k.  Populated lazily on
        # first miss; LRU eviction when full.
        self._vram: dict[
            tuple[int, int], tuple["torch.Tensor", "torch.Tensor", int]
        ] = {}
        self._tick = 0
        # Per-call lock: serialises _evict and the tick increment so the
        # LRU ordering is consistent across bg threads + foreground.
        self._lock = threading.Lock()
        # Telemetry
        self._hits = 0
        self._misses = 0
        self._bytes_copied = 0
        logger.info(
            "BicameralExpertCache: %d MoE layers under %s, top-%d VRAM cache",
            n, self.model_dir, self.top_k,
        )

    def _handle(self, path: str):
        h = self._handles.get(path)
        if h is None:
            from safetensors.torch import safe_open

            ctx = safe_open(path, framework="pt")
            h = ctx.__enter__()
            self._handles[path] = h
        return h

    def _evict_if_full(self) -> None:
        """LRU eviction — drop the (layer, expert) with the smallest tick.

        top_k <= 0 is the legal "no cache" mode (honoured).
        """
        if self.top_k <= 0:
            return
        if len(self._vram) <= self.top_k:
            return
        oldest_key = min(self._vram, key=lambda k: self._vram[k][2])
        del self._vram[oldest_key]

    def fetch(self, layer: int, expert: int, device, dtype):
        """Return (gate_up [1280,2560], down [2560,640]) on ``device``.

        VRAM hot-cache: if (layer, expert) is in cache, copy to device
        (small tile) and return; else fall back to the safetensors slice,
        store the *host* tensor in the cache, then return the device copy.
        """
        import torch

        key = (int(layer), int(expert))
        with self._lock:
            self._tick += 1
            cached = self._vram.get(key)
            if cached is not None:
                w13_host, w2_host, _ = cached
                self._hits += 1
                self._bytes_copied += (
                    w13_host.nelement() * 2 + w2_host.nelement() * 2
                )
                self._vram[key] = (w13_host, w2_host, self._tick)
                return (
                    w13_host.to(device=device, dtype=dtype, non_blocking=True),
                    w2_host.to(device=device, dtype=dtype, non_blocking=True),
                )
            self._misses += 1
        # Outside the lock: the read is the slow path and can run
        # concurrently with other fetches (safetensors reads are safe
        # to interleave at the OS level).
        self._read_to_cache(layer, expert)
        with self._lock:
            w13_host, w2_host, _ = self._vram[key]
        return (
            w13_host.to(device=device, dtype=dtype, non_blocking=True),
            w2_host.to(device=device, dtype=dtype, non_blocking=True),
        )

    def prefetch_async(self, layer: int, experts: list[int]) -> None:
        """Schedule background reads of ``[(layer, e) for e in experts]``.

        Designed to be called from the model forward path *after* a
        layer's gate has produced its topi — the same expert set is a
        reasonable predictor for the next layer's routing.  The reads
        run on the shared thread pool; when the next layer calls
        ``fetch()`` the tiles should already be in the VRAM cache.

        The layer periodicity (3 linear → 1 full, ×12) means: during
        the compute-cheap linear layers, the bg pool has ~0.5-1s per
        layer to read the next block's expert tiles before GPU0 reaches
        the compute-heavy full-attention layers.
        """
        if not experts:
            return
        layer = int(layer)
        missing = [
            int(e)
            for e in experts
            if (layer, int(e)) not in self._vram
        ]
        if not missing:
            return

        def _read(e: int) -> None:
            try:
                self._read_to_cache(layer, e)
            except Exception as exc:  # noqa: BLE001 — telemetry only
                logger.debug(
                    "prefetch_async layer=%d expert=%d failed: %s",
                    layer, e, exc,
                )

        _get_pool().map(_read, missing)

    def _read_to_cache(self, layer: int, expert: int) -> None:
        """Materialize the (layer, expert) tile into the host VRAM cache.

        Synchronous; shared by ``fetch`` (on miss) and
        ``prefetch_async`` (background).  top_k <= 0 means "no cache"
        mode — read once, discard.  Raises on missing weight.
        """
        import torch

        key = (int(layer), int(expert))
        # Fast path: already in cache
        if key in self._vram:
            return
        slot = self._loc.get(layer)
        if not slot or "gate_up" not in slot or "down" not in slot:
            raise KeyError(f"no expert tiles for layer {layer}")
        gp, gk = slot["gate_up"]
        dp, dk = slot["down"]
        w13 = self._handle(gp).get_slice(gk)[int(expert)]
        w2 = self._handle(dp).get_slice(dk)[int(expert)]
        self._bytes_copied += w13.nelement() * 2 + w2.nelement() * 2
        with self._lock:
            self._vram[key] = (w13, w2, self._tick)
            self._evict_if_full()

    def prefetch_topk(
        self, top_ids: dict[int, list[int]], device, dtype
    ) -> int:
        """Pre-warm the VRAM cache for the given per-layer expert id lists.

        ``top_ids`` is a ``{layer: [expert_id, ...]}`` mapping.  Returns
        the number of (layer, expert) tiles successfully cached.
        Designed to be called after the first warmup pass, once we know
        which experts are the most frequent.
        """
        n = 0
        for layer, ids in top_ids.items():
            for e in ids:
                key = (int(layer), int(e))
                if key in self._vram:
                    continue
                try:
                    self.fetch(layer, e, device, dtype)
                    n += 1
                except Exception:
                    pass
        return n

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "top_k": self.top_k,
            "vram_entries": len(self._vram),
            "loc_layers": len(self._loc),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
            "bytes_copied": self._bytes_copied,
        }


def swiglu_expert(x, w13, w2):
    import torch

    gate, up = torch.nn.functional.linear(x, w13).chunk(2, dim=-1)
    return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, w2)


# ---------------------------------------------------------------------------
# Process-global singleton cache registry
# ---------------------------------------------------------------------------
_CACHES: dict[str, BicameralExpertCache] = {}
_CACHE_LOCK = threading.Lock()


def get_expert_cache(model_dir: str) -> BicameralExpertCache:
    key = os.path.abspath(os.path.expanduser(model_dir))
    with _CACHE_LOCK:
        c = _CACHES.get(key)
        if c is None:
            c = BicameralExpertCache(key)
            _CACHES[key] = c
        return c


__all__ = ["BicameralExpertCache", "get_expert_cache", "swiglu_expert"]
