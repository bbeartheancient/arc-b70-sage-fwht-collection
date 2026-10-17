"""python -m afi  — host self-test + optional codec go/no-go."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import QWEN38_27B, fwht_matches_sylvester, reconstruct_report
from .fwht import fwht, ifwht
from .shapes import walsh_pad


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="afi")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("selftest", help="host FWHT + codec checks")
    t.add_argument("--seed", type=int, default=0)

    c = sub.add_parser("codec", help="reconstruction report on a random or given slice")
    c.add_argument("--rows", type=int, default=256)
    c.add_argument("--cols", type=int, default=256)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--rotate", action="store_true", help="FWHT-rotate columns first (pad)")

    s = sub.add_parser("shapes", help="print Qwen 3.8 27B widths")

    args = p.parse_args(argv)
    if args.cmd == "selftest":
        return _selftest(args.seed)
    if args.cmd == "codec":
        return _codec(args)
    if args.cmd == "shapes":
        m = QWEN38_27B
        print(
            json.dumps(
                {
                    "name": m.name,
                    "hidden": m.hidden_size,
                    "hidden_fwht": m.hidden_fwht,
                    "ffn": m.intermediate_size,
                    "ffn_fwht": m.ffn_fwht,
                    "attn_head": m.attn_fwht,
                    "linear_head": m.linear_fwht,
                    "layers": m.num_layers,
                },
                indent=2,
            )
        )
        return 0
    return 2


def _selftest(seed: int) -> int:
    rng = np.random.default_rng(seed)
    worst = 0.0
    for n in (16, 32, 64, 128, 256):
        err = fwht_matches_sylvester(n, rows=8, rng=rng)
        worst = max(worst, err)
        x = rng.standard_normal((4, n))
        rec = ifwht(fwht(x))
        inv = float(np.max(np.abs(rec - x)))
        print(f"fwht n={n:3d}  vs H@x {err:.3e}  invert {inv:.3e}")
        if err > 1e-8 or inv > 1e-8:
            print("FAIL", file=sys.stderr)
            return 1
    W = rng.standard_normal((64, 64))
    rep = reconstruct_report(W)
    print("codec", json.dumps(rep))
    print(f"worst fwht err {worst:.3e}  ok")
    return 0


def _codec(args) -> int:
    rng = np.random.default_rng(args.seed)
    W = rng.standard_normal((args.rows, args.cols))
    if args.rotate:
        pad = walsh_pad(W.shape[1])
        if pad != W.shape[1]:
            W = np.pad(W, ((0, 0), (0, pad - W.shape[1])))
        from .fwht import fwht

        W = fwht(W, axis=1, normalize=True)
    print(json.dumps(reconstruct_report(W), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
