"""Load torch.ops.afi.fwht (SYCL extension).

The .so is built by scripts/build-torch-fwht.sh against the vLLM
venv + intel/llvm v7. Host-only installs have no .so; import stays
optional.
"""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False
_SO: Path | None = None


def _candidates() -> list[Path]:
    env = os.environ.get("AFI_XPU_SO")
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    out: list[Path] = []
    if env:
        out.append(Path(env))
    out.extend(sorted(here.glob("afi_xpu*.so")))
    out.append(root / "build-llvm7" / "afi_xpu.so")
    out.append(root / "build" / "afi_xpu.so")
    return out


def so_path() -> Path | None:
    for p in _candidates():
        if p.is_file():
            return p
    return None


def loaded() -> bool:
    return _LOADED


def load() -> None:
    """Idempotent. Raises FileNotFoundError if the extension is absent."""
    global _LOADED, _SO
    if _LOADED:
        return
    import torch

    path = so_path()
    if path is None:
        raise FileNotFoundError(
            "afi_xpu.so missing — run scripts/build-torch-fwht.sh "
            "against the vLLM venv (intel/llvm v7)"
        )
    torch.ops.load_library(str(path))
    # Dynamo/Inductor: torch's fallback fake for a C++-only Meta kernel
    # concretizes every symbolic size (ConstraintViolationError once vLLM
    # mark_dynamic hits the token dim). A Python fake impl is sym-safe.
    # torch < 2.13 refuses register_fake when a C++ Meta impl exists;
    # those versions just use the Meta kernel.
    try:
        torch.library.register_fake(
            "afi::fwht", lambda x, scale=None, impl="auto": torch.empty_like(x)
        )
    except RuntimeError:
        pass
    _SO = path
    _LOADED = True


def fwht(x, scale: float | None = None, impl: str | None = None):
    """Orthogonal FWHT along the last dim (n in {16,32,64,128,256})."""
    load()
    import torch

    if impl is None:
        impl = os.environ.get("AFI_FWHT_IMPL", "auto")
    return torch.ops.afi.fwht(x, scale, impl)


def fwht_heads(x, head_dim: int, scale: float | None = None, impl: str | None = None):
    """FWHT each head. Last dim is h*d; view to (..., h, d)."""
    last = x.shape[-1]
    if last % head_dim != 0:
        raise ValueError(
            f"fwht_heads: last dim {last} is not a multiple of {head_dim}"
        )
    heads = last // head_dim
    y = x.view(*x.shape[:-1], heads, head_dim)
    y = fwht(y, scale=scale, impl=impl)
    return y.view(x.shape)


def xmx_ok() -> bool:
    load()
    import torch

    return bool(torch.ops.afi.xmx_ok())
