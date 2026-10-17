"""arrayfire-intel host package — FWHT, flux-tile codec, model shapes.

Device kernels live under sycl/. This package is the correctness
oracle and the offline encoder. hoa64 is an optional sibling.
"""

from .fwht import fwht, ifwht, sylvester, fwht_matches_sylvester
from .codec import (
    h8_atoms,
    encode_blocks,
    decode_blocks,
    FluxBlob,
    groupwise_int4_error,
    reconstruct_report,
)
from .shapes import QWEN38_27B

try:
    from .xpu import fwht as fwht_xpu
    from .xpu import fwht_heads, load as load_xpu
except Exception:  # pragma: no cover - host-only
    fwht_xpu = None  # type: ignore[assignment]
    fwht_heads = None  # type: ignore[assignment]
    load_xpu = None  # type: ignore[assignment]

__version__ = "0.1.0"
__all__ = [
    "fwht",
    "ifwht",
    "sylvester",
    "fwht_matches_sylvester",
    "h8_atoms",
    "encode_blocks",
    "decode_blocks",
    "FluxBlob",
    "groupwise_int4_error",
    "reconstruct_report",
    "QWEN38_27B",
    "fwht_xpu",
    "fwht_heads",
    "load_xpu",
]
