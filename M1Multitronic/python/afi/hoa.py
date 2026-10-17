"""Optional hoa64 import. Sibling package at ~/hoa64, imported from ~."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HOA = None
_ERR: Exception | None = None


def hoa64_root() -> Path | None:
    env = os.environ.get("AFI_HOA64_ROOT")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path.home() / "hoa64")
    here = Path(__file__).resolve()
    candidates.append(here.parents[3] / "hoa64")
    for c in candidates:
        if (c / "hadamard.py").is_file():
            return c
    return None


def load():
    """Return the hoa64 package or raise ImportError."""
    global _HOA, _ERR
    if _HOA is not None:
        return _HOA
    if _ERR is not None:
        raise _ERR
    root = hoa64_root()
    if root is None:
        _ERR = ImportError(
            "hoa64 not found. Set AFI_HOA64_ROOT or keep ~/hoa64 as a sibling."
        )
        raise _ERR
    parent = str(root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    try:
        import hoa64  # type: ignore
    except Exception as e:  # pragma: no cover
        _ERR = e
        raise
    _HOA = hoa64
    return hoa64


def available() -> bool:
    try:
        load()
        return True
    except ImportError:
        return False
