"""Compatibility re-export of xpu_flash_attn through flash_attn.flash_attn_interface."""
from xpu_flash_attn import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
)

# Aliases used by code that imports the interface module directly.
flash_attn_func_v2 = flash_attn_func
flash_attn_func_v3 = flash_attn_func
flash_attn_func_v4 = flash_attn_func
flash_attn_varlen_func_v2 = flash_attn_varlen_func
flash_attn_varlen_func_v3 = flash_attn_varlen_func
flash_attn_varlen_func_v4 = flash_attn_varlen_func


def _flash_attn_forward(*args, **kwargs):
    raise NotImplementedError(
        "_flash_attn_forward is an internal flash_attn symbol not supported by the XPU SDPA shim."
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
    "_flash_attn_forward",
]
