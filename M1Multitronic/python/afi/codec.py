"""Flux-tile codebook — host encode / decode.

Per 8×8 weight block we store which of the four H.8 seam-atoms
matched, plus a signed scale. Decode is a gather. This is not
micromag: SA is encode-only and lives in hoa64; the default
encoder here is nearest-atom (closed form).

The four atoms are Sylvester H_8 and its three Kronecker-seam
flips (right column, bottom row, both). That is the catalog the
SYCL kernel holds in SLM. Walsh row/col signs fold into the
signed scale for v1; the on-wire struct still has the fields.

Go/no-go (from the research note): reconstruction error on one
real layer must beat or match groupwise INT4. If it loses, keep
FWHT-as-rotate and drop the codebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fwht import sylvester

ATOM = 8
N_ATOMS = 4


def h8_atoms() -> np.ndarray:
    """(4, 8, 8) int8 catalog. Index 0 is Sylvester H_8."""
    H = sylvester(ATOM)
    atoms = np.stack([H, H.copy(), H.copy(), H.copy()])
    atoms[1][:, -1] *= -1
    atoms[2][-1, :] *= -1
    atoms[3][:, -1] *= -1
    atoms[3][-1, :] *= -1
    return atoms.astype(np.int8)


@dataclass
class FluxBlob:
    """Packed codebook for a (M, N) weight, M and N multiples of 8."""

    rows: int
    cols: int
    tile_id: np.ndarray  # (br, bc) uint8
    scale: np.ndarray  # (br, bc) float32, signed
    row_sign: np.ndarray  # (br, bc) int8, ±1
    col_sign: np.ndarray  # (br, bc) int8, ±1

    @property
    def n_blocks(self) -> int:
        return int(self.tile_id.size)


def encode_blocks(W: np.ndarray, atoms: np.ndarray | None = None) -> FluxBlob:
    """Nearest-atom encode. ``W`` is float, shape (M, N), 8 | M, 8 | N."""
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError("encode_blocks expects a 2-D weight")
    M, N = W.shape
    if M % ATOM or N % ATOM:
        raise ValueError(f"weight {M}x{N} is not 8-aligned")
    atoms = h8_atoms() if atoms is None else np.asarray(atoms, dtype=np.int8)
    br, bc = M // ATOM, N // ATOM
    tiles = W.reshape(br, ATOM, bc, ATOM).transpose(0, 2, 1, 3)
    # (br, bc, 4) inner products
    A = atoms.reshape(N_ATOMS, ATOM * ATOM).astype(np.float64)
    flat = tiles.reshape(br, bc, ATOM * ATOM)
    corr = flat @ A.T
    tid = np.argmax(np.abs(corr), axis=-1).astype(np.uint8)
    # gather the winning correlation → signed scale = <W, atom> / 64
    gathered = np.take_along_axis(corr, tid[..., None], axis=-1)[..., 0]
    scale = (gathered / (ATOM * ATOM)).astype(np.float32)
    ones = np.ones_like(tid, dtype=np.int8)
    return FluxBlob(
        rows=M,
        cols=N,
        tile_id=tid,
        scale=scale,
        row_sign=ones,
        col_sign=ones,
    )


def decode_blocks(blob: FluxBlob, atoms: np.ndarray | None = None) -> np.ndarray:
    atoms = h8_atoms() if atoms is None else np.asarray(atoms, dtype=np.int8)
    br, bc = blob.tile_id.shape
    out = np.empty((br, bc, ATOM, ATOM), dtype=np.float64)
    for i in range(N_ATOMS):
        mask = blob.tile_id == i
        if not np.any(mask):
            continue
        atom = atoms[i].astype(np.float64)
        # row_sign / col_sign are ±1 and broadcast over the 8×8
        rs = blob.row_sign[mask].astype(np.float64)[:, None, None]
        cs = blob.col_sign[mask].astype(np.float64)[:, None, None]
        sc = blob.scale[mask].astype(np.float64)[:, None, None]
        out[mask] = sc * rs * atom * cs
    W = out.transpose(0, 2, 1, 3).reshape(blob.rows, blob.cols)
    return W


def groupwise_int4_error(W: np.ndarray, group: int = 32) -> float:
    """Rel Frobenius error of symmetric INT4 (range ±8) with per-group scale."""
    W = np.asarray(W, dtype=np.float64)
    flat = W.ravel()
    pad = (-flat.size) % group
    if pad:
        flat = np.concatenate([flat, np.zeros(pad)])
    g = flat.reshape(-1, group)
    scale = np.max(np.abs(g), axis=1, keepdims=True)
    scale = np.where(scale == 0, 1.0, scale) / 8.0
    q = np.clip(np.round(g / scale), -8, 7)
    rec = (q * scale).ravel()[: W.size].reshape(W.shape)
    return float(np.linalg.norm(W - rec) / (np.linalg.norm(W) + 1e-12))


def reconstruct_report(W: np.ndarray) -> dict:
    """Codec go/no-go numbers on one matrix. No GPU."""
    W = np.asarray(W, dtype=np.float64)
    blob = encode_blocks(W)
    rec = decode_blocks(blob)
    rel = float(np.linalg.norm(W - rec) / (np.linalg.norm(W) + 1e-12))
    sign_rel = float(
        np.linalg.norm(W - np.sign(W) * np.mean(np.abs(W)))
        / (np.linalg.norm(W) + 1e-12)
    )
    int4 = groupwise_int4_error(W)
    return {
        "shape": list(W.shape),
        "n_blocks": blob.n_blocks,
        "flux_rel": rel,
        "sign_rel": sign_rel,
        "int4_g32_rel": int4,
        "pass_vs_int4": rel <= int4,
        "tile_hist": np.bincount(blob.tile_id.ravel(), minlength=N_ATOMS).tolist(),
    }
