"""XPU compatibility shim that shadows the official flash_attn package.

This package makes existing code that does ``import flash_attn`` work on Intel
XPU by forwarding the supported public API to ``xpu_flash_attn`` (which itself
uses ``torch.nn.functional.scaled_dot_product_attention``).

Unsupported functions/classes raise ``NotImplementedError`` with a clear
message rather than failing at import time.
"""
from xpu_flash_attn import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
)

# Aliases used by downstream code that expects versioned names.
flash_attn_func_v2 = flash_attn_func
flash_attn_func_v3 = flash_attn_func
flash_attn_func_v4 = flash_attn_func
flash_attn_varlen_func_v2 = flash_attn_varlen_func
flash_attn_varlen_func_v3 = flash_attn_varlen_func
flash_attn_varlen_func_v4 = flash_attn_varlen_func
flash_attn_varlen_qkvpacked_func = flash_attn_qkvpacked_func
flash_attn_kvpacked_func_v2 = flash_attn_kvpacked_func


def flash_attn_with_kvcache(*args, **kwargs):
    raise NotImplementedError(
        "flash_attn_with_kvcache is not supported by the XPU SDPA shim. "
        "Use the llama.cpp SYCL flash-attention path for KV-cache attention."
    )


def sparse_attn_with_sla(*args, **kwargs):
    raise NotImplementedError(
        "sparse_attn_with_sla is not supported by the XPU SDPA shim."
    )


__all__ = [
    "flash_attn_func",
    "flash_attn_kvpacked_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_func_v2",
    "flash_attn_func_v3",
    "flash_attn_func_v4",
    "flash_attn_varlen_func_v2",
    "flash_attn_varlen_func_v3",
    "flash_attn_varlen_func_v4",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_kvpacked_func_v2",
    "flash_attn_with_kvcache",
    "sparse_attn_with_sla",
]
