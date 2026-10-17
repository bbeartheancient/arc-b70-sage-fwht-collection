"""Training-free QSA (Quest-style block-sparse attention) decode patch.

Monkey-patches ``FlashAttentionImpl.forward`` in the vLLM XPU fork with a
decode-only block-sparse branch, gated by ``AFI_QSA_DECODE=1``.  The sparse
branch engages only when all of:

  * ``attn_metadata`` is not None (profiling passes fall through),
  * exactly one live request (``seq_lens.shape[0] == 1``; spec-decode steps
    do not register as decode_reqs because their q_len > 1),
  * ``max_query_len <= 8`` (MTP draft layers run q_len = 1 + num_spec = 5),
  * ``max_seq_len > AFI_QSA_MIN_CTX`` (default 4096),
  * no alibi / sinks / sliding window on the impl,
  * KV cache is bf16/fp16 or fp8 (``fp8`` / ``fp8_e4m3`` with the layer's
    per-tensor ``_k_scale`` / ``_v_scale`` buffers present; other
    quantizations still fall back),
  * layer is not an MTP draft layer when ``AFI_QSA_SKIP_MTP=1`` (default).

Anything unexpected -> original forward, warning logged once.

Sparse path (fixed shapes, no ``.item()``, no host sync — safe under
XPUGraph FULL_DECODE_ONLY).  Selection granularity is the fixed S-token
span (``AFI_QSA_SPAN``, default 64), NOT the physical block: on this
hybrid GDN + full-attn model the paged cache uses inflated physical
blocks (1664 = 26 * 64 tokens, uniform across KV cache groups).  Logical
token t lives at flat index ``block_table[t // P] * P + t % P``.

  1. mean-pool K per S-token span in logical order (block_table gather,
     chunked to bound the dequantized intermediate), AMORTIZED across
     decode steps in a per-layer persistent buffer (``_POOL_CACHE``):
     only the newest span(s) and the growing tail block are re-pooled
     per step (``qsa_span_pool_update``); resets on sequence shrink or
     a >2-span jump (new request / prefill while gated off)
     -> compressed keys [max_blocks * P // S, kv_heads, head_dim]
  2. score with the LAST query token: per KV head, sum of ReLU(q . k_bar)
     over its grouped Q heads (``qsa_block_scores``)
  3. fixed topk of ``AFI_QSA_BUDGET // S`` spans, with span 0 (sink) and
     the 2 most recent valid spans forced in via +inf scores; invalid
     tail spans padded to -inf (``qsa_select_blocks``)
  4. gather selected K/V spans (two-level indexing: block id + in-block
     offset) into a preallocated scratch of [AFI_QSA_BUDGET, kv_heads,
     head_dim] in the model compute dtype; fp8 spans are dequantized
     (fp8 -> f32 x scale -> compute dtype) during pooling and the gather;
     valid tail length derived from ``seq_lens`` as a device tensor
  5. dense varlen attention over the gathered contiguous KV via the same
     ``flash_attn_varlen_func`` (no block_table).

Known approximation: the gathered KV is not in positional order, so with
MTP q_len > 1 the causal mask over the gathered window is inexact (the
forced-in recent spans carry the strictly-ordered tail, so the error is
bounded by budget coverage).  Decode q_len == 1 is exact up to selection.

Env knobs: AFI_QSA_DECODE (0/1), AFI_QSA_BUDGET (tokens, default 2048),
AFI_QSA_SPAN (tokens, default 64; must divide both the budget and the
physical block size), AFI_QSA_MIN_CTX (default 4096),
AFI_QSA_SKIP_MTP (default 1).
"""

from __future__ import annotations

import logging
import os
import sys

import torch

logger = logging.getLogger("afi.qsa_decode")

_INSTALLED = False
_DISABLED = False  # set on first unexpected-shape fallback
_WARNED: set[str] = set()
_ENGAGED = 0  # sparse-branch executions (custom op runs eagerly on XPU)
_SKIPPED = 0  # gate rejections (logged rate-limited with the reason)
_DBG_DONE = False  # one-shot qsa-vs-stock output comparison

# Preallocated K/V scratch, keyed by (budget, kv_heads, head_dim, dtype,
# device-string).  Shared across layers: each layer's kernel consumes the
# scratch before the next layer's forward runs (also true under graph
# replay, which fixes the same addresses).
_SCRATCH: dict = {}


def _log(msg: str) -> None:
    logger.info(msg)
    print(f"[afi-qsa] {msg}", file=sys.stderr, flush=True)


def _warn_once(key: str, msg: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(msg)
    print(f"[afi-qsa] WARNING: {msg}", file=sys.stderr, flush=True)


def _env_on(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Pure scoring / selection (CPU-testable, fullgraph-traceable)
# ---------------------------------------------------------------------------


def qsa_block_scores(q_last, pooled_k):
    """Quest-style page scores.

    q_last:   [num_q_heads, head_dim] — the last query token.
    pooled_k: [max_blocks, num_kv_heads, head_dim] — mean-pooled keys.

    Returns [max_blocks]: per KV head, sum of ReLU(q . k_bar) over that
    head's grouped Q heads, then summed over KV heads.
    """
    num_q_heads, head_dim = q_last.shape
    max_blocks, num_kv_heads, _ = pooled_k.shape
    group = num_q_heads // num_kv_heads
    q_grouped = q_last.view(num_kv_heads, group, head_dim)
    dots = torch.einsum("bhd,hgd->bhg", pooled_k.to(q_last.dtype), q_grouped)
    return dots.clamp_min(0).sum(dim=(1, 2))


def qsa_select_blocks(scores, budget_blocks: int, valid_blocks, n_recent: int = 2):
    """Fixed-shape top-k block selection.

    scores:       [max_blocks] block scores.
    budget_blocks: fixed python int (topk width).
    valid_blocks: scalar int tensor — number of blocks holding real KV.
    n_recent:     force-include this many most-recent valid blocks.

    Block 0 (sink) and the last ``n_recent`` valid blocks are forced in by
    setting their scores to +inf; blocks >= valid_blocks are padded to
    -inf so topk still returns a fixed [budget_blocks] index tensor.
    """
    max_blocks = scores.shape[0]
    ar = torch.arange(max_blocks, device=scores.device)
    valid = ar < valid_blocks
    recent = valid & (ar >= valid_blocks - n_recent)
    sink = ar == 0
    inf = torch.full((), float("inf"), dtype=scores.dtype, device=scores.device)
    neg_inf = torch.full((), float("-inf"), dtype=scores.dtype, device=scores.device)
    masked = torch.where(valid, scores, neg_inf)
    masked = torch.where(sink | recent, inf, masked)
    return torch.topk(masked, budget_blocks).indices


def qsa_dequant(x, scale):
    """Dequant fp8 (or passthrough bf16/fp16) cache values to float32.

    x:     any-dtype tensor (fp8_e4m3 when the KV cache is quantized).
    scale: None (already float) or a 0-dim fp32 tensor (per-tensor scale,
           ``layer._k_scale`` / ``layer._v_scale`` in this fork).
    """
    y = x.to(torch.float32)
    if scale is not None:
        y = y * scale
    return y


def qsa_pool_k(key_cache, block_ids, k_scale=None):
    """Mean-pool K per physical block -> [num_ids, kv_heads, head_dim] f32.

    Dequantizes first (scale commutes with the mean) so scoring matches
    what the dense kernel computes internally on fp8 caches.
    """
    return qsa_dequant(key_cache.index_select(0, block_ids), k_scale).mean(dim=1)


def qsa_span_pool_k(key_cache, block_table, span: int, k_scale=None,
                    out_dtype=None, chunk_blocks: int = 0):
    """Mean-pool K per ``span``-token span in LOGICAL sequence order.

    key_cache:   [num_blocks, P, kv_heads, head_dim] physical paged cache
                 (P = physical block size, e.g. 1664 on hybrid GDN models).
    block_table: [max_blocks] long — logical block position -> physical id.
    span:        S; requires P % S == 0.
    chunk_blocks: process this many logical blocks at a time to bound the
                 dequantized intermediate (0 = all at once).

    Returns [max_blocks * P // S, kv_heads, head_dim] in out_dtype (f32
    default).  Dequantizes before pooling so fp8 scores match the kernel's
    internal math (scale commutes with the mean).
    """
    num_blocks, phys_block, kv_heads, head_dim = key_cache.shape
    max_blocks = block_table.shape[0]
    per_block = phys_block // span
    if chunk_blocks <= 0:
        chunk_blocks = max_blocks
    outs = []
    for i in range(0, max_blocks, chunk_blocks):
        phys = block_table[i:i + chunk_blocks]
        k = qsa_dequant(key_cache.index_select(0, phys), k_scale)
        outs.append(k.view(phys.shape[0] * per_block, span, kv_heads, head_dim).mean(1))
    pooled = torch.cat(outs)
    return pooled.to(out_dtype) if out_dtype is not None else pooled


def qsa_span_flat_ids(block_table, span_ids, span: int, phys_block: int):
    """Flat physical cache indices for the given logical span ids.

    Logical token t lives at flat index ``block_table[t // P] * P + t % P``
    in the cache flattened to [num_blocks * P, kv_heads, head_dim].
    Returns [span_ids.numel() * span] long — fixed-shape tensor math only.
    """
    starts = span_ids * span  # logical token offset of each span start
    phys = block_table.index_select(0, starts // phys_block)
    flat_start = phys * phys_block + starts % phys_block
    offs = torch.arange(span, device=block_table.device, dtype=flat_start.dtype)
    return (flat_start.unsqueeze(1) + offs.unsqueeze(0)).flatten()


def qsa_gather_kv(key_cache, value_cache, phys_ids, k_scale=None, v_scale=None,
                  out_dtype=None):
    """Gather physical blocks -> contiguous [budget_tokens, kv_heads, D].

    key_cache/value_cache: [num_blocks, block_size, kv_heads, head_dim]
    (the post-transpose/split layout of FlashAttentionImpl.forward; fp8
    viewed when the KV cache is quantized).
    phys_ids: [budget_blocks] long tensor of physical block ids.
    k_scale/v_scale: per-tensor dequant scales (None = cache already float).
    out_dtype: cast result to this dtype (model compute dtype); defaults to
    float32 when dequantizing, cache dtype otherwise.
    """
    budget_blocks = phys_ids.shape[0]
    block_size = key_cache.shape[1]
    kv_heads = key_cache.shape[2]
    head_dim = key_cache.shape[3]
    if out_dtype is None:
        out_dtype = torch.float32 if k_scale is not None else key_cache.dtype
    k = key_cache.index_select(0, phys_ids).view(budget_blocks * block_size, kv_heads, head_dim)
    v = value_cache.index_select(0, phys_ids).view(budget_blocks * block_size, kv_heads, head_dim)
    if k_scale is not None or v_scale is not None:
        k = qsa_dequant(k, k_scale).to(out_dtype)
        v = qsa_dequant(v, v_scale).to(out_dtype)
    return k, v


# ---------------------------------------------------------------------------
# Forward wrapper
# ---------------------------------------------------------------------------

# Per-layer incremental span-pool cache: layer key ->
# {"pooled": [max_spans, kv_heads, head_dim] compute-dtype buffer,
#  "upto": int spans pooled so far}.  Amortizes pooling across decode
# steps: only the newest span(s) (and the growing tail block) are pooled
# per step instead of the whole cache.  Safe because attention runs eager
# (AFI_GRAPH_MODE=PIECEWISE); python ints only, no host sync.
_POOL_CACHE: dict = {}


def _pool_state(layer, max_spans: int, kv_heads: int, head_dim: int, ref):
    key = getattr(layer, "layer_name", None) or f"id{id(layer)}"
    st = _POOL_CACHE.get(key)
    if (
        st is None
        or st["pooled"].shape != (max_spans, kv_heads, head_dim)
        or st["pooled"].dtype != ref.dtype
        or st["pooled"].device != ref.device
    ):
        st = {
            "pooled": torch.empty(
                max_spans, kv_heads, head_dim, dtype=ref.dtype, device=ref.device
            ),
            "upto": 0,
        }
        _POOL_CACHE[key] = st
    return st


def qsa_span_pool_update(pooled_buf, upto: int, key_cache, block_table,
                         span: int, valid_spans: int, k_scale=None,
                         out_dtype=None, chunk_blocks: int = 8) -> int:
    """Incrementally refresh ``pooled_buf`` up to ``valid_spans`` spans.

    pooled_buf: [max_spans, kv_heads, head_dim] persistent buffer (logical
                span order).
    upto:       spans pooled so far (python int, from the layer state).
    valid_spans: python int — ceil(current seq_len / span).

    Re-pools the block range containing span ``upto - 1`` onward every
    call: the tail span grows between steps even when valid_spans is
    unchanged, and physical-block-aligned re-pooling covers it.  Resets
    (full re-pool from span 0) when the sequence shrank (new request) or
    jumped by more than 2 spans (a prefill happened while the branch was
    gated off).  Returns the new ``upto``.
    """
    per_block = key_cache.shape[1] // span
    if valid_spans < upto or valid_spans > upto + 2:
        upto = 0
    if valid_spans == 0:
        return 0
    lo_span = max(upto - 1, 0)
    lo_blk = lo_span // per_block
    hi_blk = (valid_spans + per_block - 1) // per_block
    pooled_chunk = qsa_span_pool_k(
        key_cache, block_table[lo_blk:hi_blk], span, k_scale,
        out_dtype=out_dtype, chunk_blocks=chunk_blocks,
    )
    pooled_buf[lo_blk * per_block:hi_blk * per_block].copy_(pooled_chunk)
    return valid_spans


def _mtp_layer(layer) -> bool:
    name = str(
        getattr(layer, "layer_name", "")
        or getattr(layer, "prefix", "")
        or getattr(layer, "layer_prefix", "")
    )
    return "mtp" in name.lower()


def _get_scratch(budget: int, kv_heads: int, head_dim: int, ref,
                 span: int = 0):
    # paged layout [budget_spans, span, kv_heads, head_dim] when span > 0 so
    # the attention call can take the compiled block-table kernel path
    key = (budget, kv_heads, head_dim, ref.dtype, str(ref.device), span)
    pair = _SCRATCH.get(key)
    if pair is None:
        shape = (budget // span, span, kv_heads, head_dim) if span \
            else (budget, kv_heads, head_dim)
        pair = (
            torch.empty(shape, dtype=ref.dtype, device=ref.device),
            torch.empty(shape, dtype=ref.dtype, device=ref.device),
        )
        _SCRATCH[key] = pair
    return pair


_ID_BT: dict = {}


def _get_identity_block_table(budget_spans: int, device):
    bt = _ID_BT.get((budget_spans, str(device)))
    if bt is None:
        # [num_seqs=1, budget_spans] — the paged kernel indexes per sequence
        bt = torch.arange(budget_spans, dtype=torch.int32,
                          device=device).unsqueeze(0)
        _ID_BT[(budget_spans, str(device))] = bt
    return bt


def _engage_reason(self, layer, attn_metadata, kv_cache, budget: int, min_ctx: int,
                   span: int):
    """None = engage; otherwise the human-readable skip reason."""
    if attn_metadata is None:
        return "no attn_metadata (profiling)"
    if getattr(attn_metadata, "use_cascade", False):
        return "cascade attention"
    # spec-decode (MTP) steps report q_len = 1 + num_spec > 1, so they do not
    # count as decode_reqs; the request count lives in seq_lens instead.
    if attn_metadata.seq_lens.shape[0] != 1:
        return f"batch size {attn_metadata.seq_lens.shape[0]} != 1"
    if attn_metadata.max_query_len > 8:
        return "max_query_len > 8"
    if attn_metadata.max_seq_len <= min_ctx:
        return "below AFI_QSA_MIN_CTX"
    if self.num_heads % self.num_kv_heads != 0:
        return "q heads not a multiple of kv heads"
    if getattr(self, "alibi_slopes", None) is not None:
        return "alibi"
    if getattr(self, "sinks", None) is not None:
        return "sinks"
    sw = getattr(self, "sliding_window", None)
    if sw is not None and tuple(sw) != (-1, -1):  # (-1, -1) = FA's "off" sentinel
        return "sliding window"
    kvdt = getattr(self, "kv_cache_dtype", "auto")
    if kvdt in ("fp8", "fp8_e4m3"):
        if not (
            torch.is_tensor(getattr(layer, "_k_scale", None))
            and torch.is_tensor(getattr(layer, "_v_scale", None))
        ):
            return "fp8 kv cache without per-tensor scales on the layer"
    elif isinstance(kvdt, str) and kvdt not in ("auto", "float16", "bfloat16"):
        return f"unsupported kv cache dtype ({kvdt})"
    # kv_cache: [num_blocks, num_kv_heads, P, 2 * head_size]; P is inflated
    # on hybrid GDN models (1664 = 26 * 64) — selection granularity is the
    # fixed S-token span, not the physical block.
    if kv_cache.dim() != 4 or kv_cache.shape[-1] != 2 * self.head_size:
        return "unexpected kv_cache layout"
    phys_block = kv_cache.shape[2]
    if phys_block < span or phys_block % span != 0 or budget % span != 0:
        return f"physical block {phys_block} / span {span} / budget {budget} incompatible"
    if _env_on("AFI_QSA_SKIP_MTP", "1") and _mtp_layer(layer):
        return "mtp layer"
    return None


def _qsa_forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                 budget: int, span: int):
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    num_actual_tokens = attn_metadata.num_actual_tokens
    # (B, H, P, 2*D) -> ([B, P, H, D], [B, P, H, D]); P = physical block size
    key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
    phys_block = key_cache.shape[1]
    budget_spans = budget // span

    # fp8 KV cache: view as e4m3 and dequant with the layer's per-tensor
    # scales (same values the stock path passes as k_descale/v_descale).
    fp8 = getattr(self, "kv_cache_dtype", "auto") in ("fp8", "fp8_e4m3")
    if fp8:
        from vllm.platforms import current_platform

        fp8_dtype = current_platform.fp8_dtype()
        key_cache = key_cache.view(fp8_dtype)
        value_cache = value_cache.view(fp8_dtype)
        k_scale = layer._k_scale
        v_scale = layer._v_scale
    else:
        k_scale = v_scale = None

    block_table = attn_metadata.block_table[0].long()  # [max_blocks]
    seq_len = attn_metadata.seq_lens[0]  # scalar device tensor

    # 1. mean-pool (dequantized) K per S-token span in logical order,
    #    amortized: only the newest span(s) + the growing tail block are
    #    pooled per step into a persistent per-layer buffer.
    #    valid_spans bookkeeping uses attn_metadata.max_seq_len, already a
    #    python int (single request — the gate enforces batch == 1), so no
    #    host sync is introduced; the tensor seq_len is used below only
    #    where the kernel needs it.
    kv_heads = key_cache.shape[2]
    per_block = phys_block // span
    max_spans = block_table.shape[0] * per_block
    valid_spans_i = (attn_metadata.max_seq_len + span - 1) // span
    st = _pool_state(layer, max_spans, kv_heads, self.head_size, query)
    st["upto"] = qsa_span_pool_update(
        st["pooled"], st["upto"], key_cache, block_table, span, valid_spans_i,
        k_scale, out_dtype=query.dtype, chunk_blocks=8,
    )
    pooled_k = st["pooled"]

    # 2. score with the last query token
    q_last = query[num_actual_tokens - 1]  # [num_q_heads, head_dim]
    scores = qsa_block_scores(q_last, pooled_k)

    # 3. fixed topk over spans with sink + recency forced in
    valid_spans = (seq_len + span - 1) // span
    if _env_on("AFI_QSA_IDENTITY_SEL", "0"):
        # debug: bypass scoring — first budget_spans spans in logical order
        sel = torch.arange(budget_spans, device=scores.device,
                           dtype=torch.long)
    else:
        sel = qsa_select_blocks(scores, budget_spans, valid_spans, n_recent=2)

    # 4. gather selected spans (dequantized into the compute-dtype scratch).
    # Two-level indexing (block id + in-block offset) instead of flattening
    # the whole cache: equivalent to index_select on the flattened cache
    # (qsa_span_flat_ids) but avoids a full-cache contiguous copy.
    starts = sel * span
    phys_b = block_table.index_select(0, starts // phys_block)  # [budget_spans]
    offs = (starts % phys_block).unsqueeze(1) + torch.arange(
        span, device=sel.device, dtype=sel.dtype
    )  # [budget_spans, S]
    gk = key_cache[phys_b.unsqueeze(1), offs].reshape(budget, kv_heads, self.head_size)
    gv = value_cache[phys_b.unsqueeze(1), offs].reshape(budget, kv_heads, self.head_size)
    if fp8:
        gk = qsa_dequant(gk, k_scale).to(query.dtype)
        gv = qsa_dequant(gv, v_scale).to(query.dtype)
    scratch_k, scratch_v = _get_scratch(budget, kv_heads, self.head_size, query,
                                        span=span)
    scratch_k.copy_(gk.view(budget // span, span, kv_heads, self.head_size))
    scratch_v.copy_(gv.view(budget // span, span, kv_heads, self.head_size))

    # 5. dense attention over the gathered KV, presented as a PAGED cache so
    # the call takes the compiled kernel path (the non-paged path is not
    # compiled for this head_dim on XPU and silently falls back to a slow
    # reference that returns a NEW tensor instead of writing `out`).
    # scratch: [budget_spans, S, kv_heads, D]; identity block table; the
    # valid tail is seqused_k (mutually exclusive with cu_seqlens_k).
    budget_t = torch.full_like(seq_len, budget)
    kv_len = torch.minimum(seq_len, budget_t).to(torch.int32).view(1)
    bt = _get_identity_block_table(budget_spans, seq_len.device)
    ret = flash_attn_varlen_func(
        q=query[:num_actual_tokens],
        k=scratch_k,
        v=scratch_v,
        out=output[:num_actual_tokens],
        cu_seqlens_q=attn_metadata.query_start_loc,
        max_seqlen_q=attn_metadata.max_query_len,
        seqused_k=kv_len,
        max_seqlen_k=budget,
        softmax_scale=self.scale,
        causal=True,
        block_table=bt,
        fa_version=self.vllm_flash_attn_version,
    )
    if torch.is_tensor(ret) and ret.data_ptr() != output[:num_actual_tokens].data_ptr():
        # fallback path returned a fresh tensor instead of writing `out`
        output[:num_actual_tokens].copy_(ret)
    return output


def _make_wrapped(orig, budget: int, min_ctx: int, span: int):
    def wrapped(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        global _DISABLED
        if _DISABLED:
            return orig(self, layer, query, key, value, kv_cache, attn_metadata,
                        output, output_scale, output_block_scale)
        reason = _engage_reason(self, layer, attn_metadata, kv_cache, budget,
                                min_ctx, span)
        if reason is not None:
            global _SKIPPED
            _SKIPPED += 1
            if (reason != "no attn_metadata (profiling)" and _SKIPPED <= 20) \
                    or _SKIPPED % 2048 == 1:
                _log(f"sparse branch skipped x{_SKIPPED}: last reason: {reason}")
            return orig(self, layer, query, key, value, kv_cache, attn_metadata,
                        output, output_scale, output_block_scale)
        global _ENGAGED
        _ENGAGED += 1
        if _ENGAGED <= 3 or _ENGAGED % 512 == 0:
            _log(f"sparse branch engaged x{_ENGAGED}")
        global _DBG_DONE
        if _env_on("AFI_QSA_DEBUG", "0") and not _DBG_DONE:
            _DBG_DONE = True
            try:
                ref = orig(self, layer, query, key, value, kv_cache, attn_metadata,
                           output.clone(), output_scale, output_block_scale)
                out = _qsa_forward(self, layer, query, key, value, kv_cache,
                                   attn_metadata, output, budget, span)
                a = ref[:attn_metadata.num_actual_tokens].float().flatten()
                b = out[:attn_metadata.num_actual_tokens].float().flatten()
                cos = torch.nn.functional.cosine_similarity(a, b, dim=0)
                # independent gather check: logical K straight from the cache
                # vs the span-gather the sparse path built (identity
                # selection => scratch rows == first kv_len logical tokens)
                kc, vc = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
                P = kc.shape[1]
                nblk = (int(attn_metadata.seq_lens[0]) + P - 1) // P
                logical_k = kc[attn_metadata.block_table[0, :nblk].long()] \
                    .reshape(-1, kc.shape[2], self.head_size)
                L = int(min(attn_metadata.seq_lens[0], budget))
                want = logical_k[:L]
                if getattr(self, "kv_cache_dtype", "auto") in ("fp8", "fp8_e4m3"):
                    from vllm.platforms import current_platform as _cp
                    want = qsa_dequant(want.view(_cp.fp8_dtype()),
                                       layer._k_scale).to(query.dtype)
                else:
                    want = want.to(query.dtype)
                scr_k, _sv = _get_scratch(budget, kc.shape[2], self.head_size,
                                          query, span=span)
                got = scr_k.reshape(budget, kc.shape[2], self.head_size)[:L].float()
                dcos = torch.nn.functional.cosine_similarity(
                    got.flatten(), want.float().flatten(), dim=0)
                maxdiff = (got - want.float()).abs().max()
                _log(f"DEBUG qsa-vs-stock output cos={cos.item():.4f} "
                     f"gather cos={dcos.item():.4f} maxdiff={maxdiff.item():.4f} "
                     f"kv_cache{tuple(kv_cache.shape)} {kv_cache.dtype} "
                     f"q{tuple(query.shape)} {query.dtype} "
                     f"block_table{tuple(attn_metadata.block_table.shape)} "
                     f"seq_lens={attn_metadata.seq_lens.tolist()} "
                     f"max_q={attn_metadata.max_query_len} "
                     f"num_actual={attn_metadata.num_actual_tokens} "
                     f"k_scale={getattr(layer, '_k_scale', None)}")
                return out
            except Exception as exc:  # noqa: BLE001
                _log(f"DEBUG block failed: {exc!r}")
                return orig(self, layer, query, key, value, kv_cache,
                            attn_metadata, output, output_scale,
                            output_block_scale)
        try:
            return _qsa_forward(self, layer, query, key, value, kv_cache,
                                attn_metadata, output, budget, span)
        except Exception as exc:  # noqa: BLE001
            _DISABLED = True
            _warn_once(
                "qsa-fallback",
                f"QSA decode branch failed ({exc!r}); permanently falling back "
                "to the stock FlashAttention path",
            )
            return orig(self, layer, query, key, value, kv_cache, attn_metadata,
                        output, output_scale, output_block_scale)

    return wrapped


def install() -> bool:
    """Monkey-patch FlashAttentionImpl.forward.  Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return True
    budget = _env_int("AFI_QSA_BUDGET", 2048)
    min_ctx = _env_int("AFI_QSA_MIN_CTX", 4096)
    span = _env_int("AFI_QSA_SPAN", 64)
    if span <= 0 or budget % span != 0:
        _warn_once("qsa-config", f"invalid AFI_QSA_SPAN={span} for budget {budget}")
        return False
    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    except Exception as exc:  # noqa: BLE001
        _warn_once("qsa-import", f"FlashAttentionImpl import failed: {exc!r}")
        return False
    orig = FlashAttentionImpl.forward
    FlashAttentionImpl.forward = _make_wrapped(orig, budget, min_ctx, span)
    _INSTALLED = True
    _log(
        f"hooked FlashAttentionImpl.forward "
        f"(budget={budget} span={span} min_ctx={min_ctx} "
        f"skip_mtp={_env_on('AFI_QSA_SKIP_MTP', '1')})"
    )
    return True
