# SPDX-License-Identifier: Apache-2.0
"""xpu_flash_attn - FlashAttention-compatible API for Intel XPU via SDPA.

Import this package as a drop-in replacement for ``flash_attn`` in code that
runs on Intel XPU:

    import xpu_flash_attn as flash_attn

or add the package directory to ``PYTHONPATH`` and use:

    from xpu_flash_attn import flash_attn_func, flash_attn_varlen_func
"""
from .functional import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
)

__all__ = [
    "flash_attn_func",
    "flash_attn_kvpacked_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_varlen_func",
]

__version__ = "0.1.0"
