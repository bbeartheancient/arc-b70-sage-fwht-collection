"""SageAttention API drop-in for Intel XPU (SDPA / oneDNN backend).

Same Python entry points Comfy / KJNodes / WanVideoWrapper import
(``sageattn``, ``sageattn_varlen``, ``sageattn_qk_int8_pv_*``), so a
CUDA Sage workflow keeps its node on an Arc card. Day-one dispatch is
``torch.nn.functional.scaled_dot_product_attention`` — the XPU FA
(oneDNN SDPA / SYCL-TLA) — which on a B70 is the fast attention path.
The FP16/FP8/INT8 fused Sage kernels are NVIDIA binaries and do not run
on XPU; the *_int8/*_fp8 entry points fall back to the same SDPA path
(see docs/FRAMEWORK.md, "If Sage Attention … comes back").
"""

from __future__ import annotations

from .core import sageattn, sageattn_varlen
from .int8 import (
    sageattn_qk_int8_pv_fp16,
    sageattn_qk_int8_pv_fp16_cuda,
    sageattn_qk_int8_pv_fp16_triton,
    sageattn_qk_int8_pv_fp8,
    sageattn_qk_int8_pv_fp8_cuda,
)

__version__ = "0.1.0"

__all__ = [
    "sageattn",
    "sageattn_varlen",
    "sageattn_qk_int8_pv_fp16",
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8",
    "sageattn_qk_int8_pv_fp8_cuda",
    "__version__",
]