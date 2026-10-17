"""Fast Walsh–Hadamard transform.

Butterflies only. The Sylvester matrix is never stored on the
hot path. ``fwht(x)`` equals ``sylvester(n) @ x`` for a vector of
length ``n = 2^k``. The inverse is the same network divided by n
(unnormalized) or the same network (orthogonal / ``normalize=True``).

This is the host oracle for the SYCL kernel. Keep it boring.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "is_pow2",
    "sylvester",
    "fwht",
    "ifwht",
    "fwht_matches_sylvester",
]


def is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def sylvester(n: int) -> np.ndarray:
    """Build Sylvester H(n) as int8. Exists so tests can form H @ x.

    Production code must not multiply by this matrix.
    """
    if not is_pow2(n):
        raise ValueError(f"sylvester requires n = 2^k, got {n}")
    H = np.array([[1]], dtype=np.int8)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]]).astype(np.int8)
    return H


def fwht(x: np.ndarray, axis: int = -1, normalize: bool = False) -> np.ndarray:
    """In-place-style FWHT along ``axis``. Returns a new float64 array.

    Unnormalized: y = H x with H H^T = n I.
    ``normalize=True``: multiply by n^{-1/2} so the map is orthogonal.
    """
    y = np.asarray(x, dtype=np.float64).copy()
    n = y.shape[axis]
    if not is_pow2(n):
        raise ValueError(f"fwht requires axis length 2^k, got {n}")
    y = np.moveaxis(y, axis, -1)
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        a = y[..., 0, :].copy()
        b = y[..., 1, :]
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
        y = y.reshape(*y.shape[:-3], n)
        h *= 2
    if normalize:
        y *= n ** -0.5
    return np.moveaxis(y, -1, axis)


def ifwht(y: np.ndarray, axis: int = -1, normalize: bool = False) -> np.ndarray:
    """Inverse. Same butterflies; unnormalized path divides by n."""
    x = fwht(y, axis=axis, normalize=False)
    n = x.shape[axis]
    if normalize:
        x *= n ** -0.5
    else:
        x /= n
    return x


def fwht_matches_sylvester(n: int, rows: int = 4, rng: np.random.Generator | None = None) -> float:
    """Max abs error of fwht(X) vs (H @ X.T).T. Used by tests and C++ dumps."""
    rng = rng or np.random.default_rng(0)
    H = sylvester(n).astype(np.float64)
    X = rng.standard_normal((rows, n))
    ref = X @ H.T
    got = fwht(X, normalize=False)
    return float(np.max(np.abs(got - ref)))
