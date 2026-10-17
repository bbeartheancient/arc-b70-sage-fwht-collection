"""INT8/FP8 SaaS-attention variants for the XPU drop-in.

These entry points match the names Comfy/KJNodes list in their
sage_attention mode selector. The real INT8 QK^T / FP8 PV fused kernels
are NVIDIA binaries; on XPU every variant routes to the same SDPA-backed
core (numerically plain attention, fast oneDNN FA). The FWHT-rotate +
INT8-downcast path in b70-hadamard-kernel-opt.md is only spliced in
*if* it beats SDPA on a measured model.
"""

from __future__ import annotations

import warnings

from .core import sageattn

_WARNED = False


def _xvpu_fallback(name: str):
    def _fn(
        q,
        k,
        v,
        is_causal: bool = False,
        attn_mask=None,
        tensor_layout: str = "HND",
        pv_accum_dtype: str = "fp32",
        scale=None,
        sm_scale=None,
        **kwargs,
    ):
        global _WARNED
        if not _WARNED:
            warnings.warn(
                f"sageattention.{name}: INT8/FP8 fused kernels are not "
                "available on Intel XPU; routing to the SDPA (oneDNN FA) "
                "backend. Numerics are plain attention.",
                stacklevel=2,
            )
            _WARNED = True
        return sageattn(
            q,
            k,
            v,
            is_causal=is_causal,
            attn_mask=attn_mask,
            tensor_layout=tensor_layout,
            scale=scale,
            sm_scale=sm_scale,
        )

    _fn.__name__ = name
    return _fn


sageattn_qk_int8_pv_fp16 = _xvpu_fallback("sageattn_qk_int8_pv_fp16")
sageattn_qk_int8_pv_fp16_cuda = _xvpu_fallback("sageattn_qk_int8_pv_fp16_cuda")
sageattn_qk_int8_pv_fp16_triton = _xvpu_fallback("sageattn_qk_int8_pv_fp16_triton")
sageattn_qk_int8_pv_fp8 = _xvpu_fallback("sageattn_qk_int8_pv_fp8")
sageattn_qk_int8_pv_fp8_cuda = _xvpu_fallback("sageattn_qk_int8_pv_fp8_cuda")