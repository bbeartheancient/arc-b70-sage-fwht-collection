#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afi.fwht import fwht, ifwht, sylvester, fwht_matches_sylvester  # noqa: E402


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {msg}")


def main() -> int:
    rng = np.random.default_rng(1)
    for n in (2, 4, 8, 16, 32, 64, 128, 256):
        err = fwht_matches_sylvester(n, rows=5, rng=rng)
        check(err < 1e-10, f"n={n} vs Sylvester {err}")
        x = rng.standard_normal((3, n))
        check(np.max(np.abs(ifwht(fwht(x)) - x)) < 1e-10, f"n={n} invert")
        y = fwht(x, normalize=True)
        check(abs(np.linalg.norm(y) - np.linalg.norm(x)) < 1e-8, f"n={n} ortho")
    # involution up to n: H (H x) = n x
    n = 32
    H = sylvester(n).astype(np.float64)
    x = rng.standard_normal(n)
    check(np.allclose(fwht(fwht(x)), n * x), "H^2 = n I")
    check(np.allclose(H @ H.T, n * np.eye(n)), "H H^T = n I")
    print("test_fwht ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
