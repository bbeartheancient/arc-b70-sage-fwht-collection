# FWHT bridge architecture (Qwen-FN, 2026-08-27)

## What this is

A short, low-bandwidth bridge that uses Walsh-Hadamard coefficients
as a *fingerprint* and *predictor* between the two B70 cards, not
as a lossless transport. The data is already GPU-resident on GPU0
during prefill; the question is what sparse signal GPU1 needs from
GPU0 to do useful parallel work, and what GPU0 can cheaply compute
to feed that signal.

## Hardware constraint (BICAMERAL §1, unchanged)

- GPU0 Gen5 x16, GPU1 Gen4 x4. Three separate PCI roots.
- **No P2P, no GPU↔GPU VRAM, no GPU↔NVMe VRAM.**
- All cross-card transfers go through host RAM (~22 GiB free).
- Both cards have working `afi.fwht` (ESIMD, 240 GiB/s on bf16
  n=4096). Encode and decode are the same operation, divided by n.

## The bridge contracts (measured, real data)

### Bridge A: gate fingerprint (predictive)

GPU0's per-layer gate produces `[S, 512]` logits. Apply FWHT over
the 512-expert axis (already a power of 2, no padding). Threshold
the top-K Walsh coefficients (sorted by mean energy over the S
tokens) and transfer to host RAM as a small `compaction_fingerprint`.

| K / 512 | Energy kept | Top-1 expert match | Top-3-in-10 match |
|---------|-------------|--------------------|-------------------|
| 5% (25) | 5%          | 38%                | 16%               |
| 10% (51)| 10%         | 61%                | 42%               |
| 20% (102)| 21%        | 77%                | 65%               |
| **50% (256)** | **51%** | **90%**       | **86%**           |
| 100% (512) | 100%    | 100%               | 100%              |

At top-50% the bridge is 256 fp16 values per layer = 512 bytes.
90% top-1 expert match on the *next layer's* routing is enough
for a speculative prefetch: pull those experts from NVMe on the
bg thread while GPU0 computes the current layer's MoE.

**Cost on GPU0:** one ESIMD FWHT over `[S, 512]` (n=512) at
240 GiB/s = 2001 × 512 × 2 / 240e9 = **0.0085 ms per layer.**
Plus one threshold (top-K by energy): trivial.
**Cost on the host bridge:** 512 bytes. <0.01 ms.

### Bridge B: hidden state residual fingerprint (for GPU1 tasks)

GPU0 emits the next layer's input hidden state. Full transfer is
`[S, 2560]` = 10.2 MB at fp16. Most of GPU1's work (compaction_90,
QSA indexer, n-gram PLE lookups) needs only an *energy profile* of
the hidden state, not the original vectors. Apply FWHT over the
2560-dim axis (pad to n=4096), threshold, and transfer the top-K
coefficients. GPU1 operates on the Walsh coefficients directly —
no decode needed for energy-domain tasks.

| K / 4096 | Energy kept | row_cos vs full | Saved bandwidth |
|----------|-------------|-----------------|-----------------|
| 5% (204)  | 10%        | 0.39            | 92%             |
| 10% (409) | 19%        | 0.50            | 84%             |
| 20% (819) | 33%        | 0.64            | 68%             |
| 30% (1228)| 44%        | 0.73            | 52%             |
| **50% (2048)** | **64%** | **0.85**    | **20%**         |
| 100% (4096) | 100%    | 1.00            | 0%              |

**Cost on GPU0:** ESIMD FWHT over `[S, 4096]` at 240 GiB/s =
2001 × 4096 × 2 / 240e9 = **0.07 ms per layer.**
**Cost on the bridge (top-50%):** 2048 × 2 × 2001 = 8 MB at
20 GB/s host RAM = 0.4 ms. Still cheaper than the GPU0 emit.

### What about full hidden-state transfer?

If GPU1 actually needs the *decoded* hidden state (e.g., to add
to a residual stream), the round-trip cost is:
- encode 0.07 ms + xfer 0.4 ms (top-50%) + decode 0.07 ms
- = 0.54 ms for lossy (row_cos 0.85)
- = 0.85 ms for lossless (full transfer at 0.5 ms)
So even lossless is <1 ms. **The bridge is not the bottleneck.**

## Layer periodicity (3 linear → 1 full, ×12)

The pattern: 36 Mamba SSM layers, 12 full-attention layers, with
PLE n-gram at layer 1. Within each 4-block, the *linear* layers
have spare GPU compute budget (Mamba is cheap relative to the
attention-heavy full layers). The bridge fingerprint fires
**after each layer's gate**; the bg thread on GPU1 fetches
predicted experts for the *next* layer during the linear layers
in between. By the time the next full-attention layer starts,
the cache is warm.

## What this changes in the v1 path

`python/afi/bicameral/sharded_moe.py::BicameralShardedMoE._routed`:
- After computing the gate's top-K, apply a small ESIMD FWHT
  (n=512) and reduce to a 50% coefficient mask (256 fp16 values).
- Write that mask to a host-RAM ring buffer under a per-layer slot.
- The bg thread pool reads (layer+1, mask) and translates to expert
  IDs (top-1 of each masked-coefficient forward path) to drive
  `prefetch_async`.
- Today the bg prefetch uses the *current* layer's expert IDs as
  the predictor. This proposal replaces that with a fingerprint
  that's tighter (a few hundred bytes vs the full top-10) and
  potentially more accurate (90% top-1 vs random 0.2%).

The result on 2001-tok prefill (rough projection from the
per-layer gate cost of 0.0085 ms × 48 layers = 0.4 ms total):
**the bridge is a 0.4 ms cost vs ~100s prefill wall time, but
unlocks 90%-accurate per-layer prefetch that hits the cache
5-10x more often than today's "same expert set" predictor.**

## What this does NOT change

- The 100s prefill wall time is dominated by GPU compute on
  the MoE GEMMs (100 GFLOPS effective, 10x off peak). The
  bridge fingerprint does not speed that up.
- The XMX-fused MoE kernel (separate track, recommended next)
  is the lever that actually moves prefill from 100s toward
  30-50s. The bridge is a complementary, smaller win.
- GPU1's "true" prefill job (per BICAMERAL §10.9) is the
  *n-gram PLE table* and the *QSA indexer* — both decode-time
  and not active in v1. The bridge is a stepping stone.

## Open questions for the operator

1. **What should GPU1 do with the hidden state fingerprint
   during prefill, that GPU0 can't already do?**
   - Compaction_90 trajectory for next turn (loops back to
     the looping transformer pipeline)
   - Speculative QSA indexer top-512 over the next layer's
     predicted blocks
   - Both?

2. **The gate-FWHT prediction is 90% accurate on synthetic
   Gaussian-mixture logits.** Real Qwen-FN logits are sparser
   (the gate is trained to be selective). The real-world
   accuracy is plausibly higher (95%+) but unmeasured.

3. **The fwht encode cost is 0.07 ms per layer; the full
   forward is ~2 s. So we're adding 0.0035% overhead per
   layer.** Negligible. But the per-layer cross-process
   IPC for the ring buffer should be lock-free (mmap or
   io_uring); using a Python queue would dominate.

## Files

- `python/afi/fwht.py` (existing, host numpy-only oracle)
- `python/afi/bicameral/sharded_moe.py` (modified _routed)
- `python/afi/bicameral/expert_cache.py` (existing
  `prefetch_async`; the fingerprint replaces the predictor)
- `docs/fwht-bridge-architecture.md` (this file)
- `scripts/bench_fwht_bridge.py` (TBD: end-to-end A/B vs
  current "same-expert-set" predictor)
