"""Minimal safetensors reader. Header + mmap, no torch / safetensors pkg.

BF16 weights are expanded to float64 for the host codec. One tensor at
a time — 27B never has to be resident.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["TensorInfo", "list_tensors", "load_f64"]


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]  # relative to the data blob (after header)
    path: Path
    header_bytes: int

    @property
    def nbytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


def _header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        raw = f.read(n)
    return 8 + n, json.loads(raw)


def list_tensors(path: Path | str) -> list[TensorInfo]:
    path = Path(path)
    header_bytes, hdr = _header(path)
    out: list[TensorInfo] = []
    for name, spec in hdr.items():
        if name == "__metadata__":
            continue
        off = spec["data_offsets"]
        out.append(
            TensorInfo(
                name=name,
                dtype=spec["dtype"],
                shape=tuple(spec["shape"]),
                data_offsets=(int(off[0]), int(off[1])),
                path=path,
                header_bytes=header_bytes,
            )
        )
    return out


def _bf16_to_f64(u16: np.ndarray) -> np.ndarray:
    # IEEE bf16 → f32 is a 16-bit left shift; then promote.
    f32 = np.frombuffer((u16.astype(np.uint32) << 16).tobytes(), dtype=np.float32)
    return f32.astype(np.float64)


def load_f64(info: TensorInfo) -> np.ndarray:
    """mmap one tensor and return a float64 copy. Caller owns the memory."""
    with info.path.open("rb") as f:
        f.seek(info.header_bytes + info.data_offsets[0])
        blob = f.read(info.nbytes)
    if info.dtype == "BF16":
        u16 = np.frombuffer(blob, dtype="<u2")
        return _bf16_to_f64(u16).reshape(info.shape)
    if info.dtype == "F16":
        return np.frombuffer(blob, dtype="<f2").astype(np.float64).reshape(info.shape)
    if info.dtype == "F32":
        return np.frombuffer(blob, dtype="<f4").astype(np.float64).reshape(info.shape)
    if info.dtype == "F64":
        return np.frombuffer(blob, dtype="<f8").copy().reshape(info.shape)
    raise ValueError(f"unsupported dtype {info.dtype} for {info.name}")
