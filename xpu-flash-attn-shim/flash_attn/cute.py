"""Compatibility re-export used by some codebases (flash_attn.cute)."""
from xpu_flash_attn import (
    flash_attn_func,
    flash_attn_varlen_func,
)

flash_attn_func_v4 = flash_attn_func
flash_attn_varlen_func_v4 = flash_attn_varlen_func

__all__ = ["flash_attn_func", "flash_attn_varlen_func", "flash_attn_func_v4", "flash_attn_varlen_func_v4"]
