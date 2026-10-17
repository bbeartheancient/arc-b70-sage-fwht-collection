"""vLLM general plugin: orthogonal FWHT on Q/K.

Gated by AFI_FWHT_QK=1. Does not touch GPTQ / XPUwNa16LinearKernel.

* Full attention (head_dim 256): after RMSNorm + RoPE. Invariant:
  H H^T = I with 1/sqrt(n), so QK^T is unchanged up to rounding.
* GDN (head_dim 128): optional, AFI_FWHT_GDN=1. Applied on the packed
  q/k slices *before* the fused conv+GDN kernel, so it is not
  attention-invariant (conv is per-channel along time). Used to
  measure 128-wide launch cost on the live path.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger("afi.vllm_plugin")


def _log(msg: str) -> None:
    logger.info(msg)
    print(f"[afi] {msg}", file=sys.stderr, flush=True)

_PATCHED = False


def _env_on(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _traceable_fwht_heads(impl: str):
    """FWHT-per-head that Dynamo can trace under fullgraph: shape math +
    views + the custom op (Meta kernel registered). No lazy load, no env
    lookup, no torch.compiler.disable — vLLM VLLM_COMPILE is fullgraph and
    rejects disable()d functions (torch._dynamo.exc.Unsupported gb0098).
    """
    import torch

    op = torch.ops.afi.fwht

    def fwht_heads(x, head_dim: int):
        last = x.shape[-1]
        if last % head_dim != 0:
            raise ValueError(
                f"fwht_heads: last dim {last} is not a multiple of {head_dim}"
            )
        heads = last // head_dim
        y = x.view(*x.shape[:-1], heads, head_dim)
        y = op(y, None, impl)
        return y.view(x.shape)

    return fwht_heads


def _fwht_gdn_qk(mod, qkvz, fwht_heads):
    import torch

    tp = getattr(mod, "tp_size", 1) or 1
    qk = mod.key_dim // tp
    rest = qkvz.shape[-1] - 2 * qk
    if rest < 0:
        logger.warning("afi GDN hook: packed last dim %s too small", qkvz.shape[-1])
        return qkvz
    q, k, tail = qkvz.split([qk, qk, rest], dim=-1)
    q = fwht_heads(q, mod.head_k_dim)
    k = fwht_heads(k, mod.head_k_dim)
    return torch_cat(q, k, tail)


def torch_cat(q, k, tail):
    import torch

    return torch.cat((q, k, tail), dim=-1)


def _patch_full_attn() -> None:
    from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

    impl = os.environ.get("AFI_FWHT_IMPL", "auto")
    fwht_heads = _traceable_fwht_heads(impl)

    orig = Qwen3NextAttention._project_qkv_gate

    def wrapped(self, qkv, positions):
        q, k, v, gate = orig(self, qkv, positions)
        q = fwht_heads(q, self.head_dim)
        k = fwht_heads(k, self.head_dim)
        return q, k, v, gate

    Qwen3NextAttention._project_qkv_gate = wrapped
    _log("hooked Qwen3NextAttention Q/K (head_dim after RoPE)")


def _patch_gdn() -> None:
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    orig = QwenGatedDeltaNetAttention.forward_xpu

    impl = os.environ.get("AFI_FWHT_IMPL", "auto")
    fwht_heads = _traceable_fwht_heads(impl)

    def wrapped(self, hidden_states):
        import torch

        num_tokens = hidden_states.size(0)
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)
        if not self.gqa_interleaved_layout:
            projected_states_qkvz = _fwht_gdn_qk(
                self, projected_states_qkvz, fwht_heads
            )

        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        z = torch.empty_like(core_attn_out)
        torch.ops.vllm.gdn_attention_core_xpu(
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
            self.prefix,
        )
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)
        out, _ = self.out_proj(core_attn_out)
        return out

    QwenGatedDeltaNetAttention.forward_xpu = wrapped
    _log("hooked GDN Q/K (pre-conv, not invariant)")


def _apply_compiler_shapes() -> None:
    """vLLM VLLM_COMPILE calls mark_dynamic on tokens. Our SYCL op
    specialized the 2048-row dummy (ConstraintViolation).

    torch.compiler.config.dynamic_shapes=False (docs 2.12) makes the
    stack specialize all dims. Dynamo's own note: that combo is not
    yet supported *with* mark_dynamic — we still try it when
    AFI_FWHT_DYNAMIC_SHAPES=static (default). Set =dynamic to leave
    the compiler flags alone.
    """
    mode = os.environ.get("AFI_FWHT_DYNAMIC_SHAPES", "static").strip().lower()
    if mode in ("dynamic", "1", "true", "on"):
        _log("torch.compiler.config: leaving dynamic_shapes as-is")
        return
    try:
        import torch.compiler.config as cc

        cc.dynamic_shapes = False
        cc.automatic_dynamic_shapes = False
        cc.assume_static_by_default = True
        _log(
            "torch.compiler.config: dynamic_shapes=False "
            "automatic_dynamic_shapes=False assume_static_by_default=True"
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"torch.compiler.config not applied: {exc}")


def register() -> None:
    """Entry point for vllm.general_plugins."""
    global _PATCHED
    if _PATCHED:
        return
    if _env_on("AFI_FWHT_QK"):
        from afi.xpu import load, xmx_ok

        load()
        _apply_compiler_shapes()
        _patch_full_attn()
        if _env_on("AFI_FWHT_GDN"):
            _patch_gdn()
        try:
            ok = xmx_ok()
        except Exception as exc:  # noqa: BLE001
            ok = f"err {exc}"
        _log(
            f"Q/K FWHT hook live (xmx_ok={ok} impl={os.environ.get('AFI_FWHT_IMPL', 'auto')})"
        )
    else:
        _log("AFI_FWHT_QK unset — torch op not hooked")
    if _env_on("AFI_QSA_DECODE"):
        from afi import qsa_decode

        qsa_decode.install()
    else:
        _log("AFI_QSA_DECODE unset — QSA decode patch not hooked")
    _PATCHED = True
