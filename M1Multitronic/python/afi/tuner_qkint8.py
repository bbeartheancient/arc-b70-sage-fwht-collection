"""Phase 1 QkInt8RotKernel launch-config tuner.

Same ILS skeleton as afi.tuner (Layer 2) with the Phase 1 config dimension
set (docs/fused-dpas-int8-plan.md): tm, sg, wg_k (subgroups = 16-key tiles
per WG), wg_q (query-block width in tm tiles), pad, kt (16-key tiles per WG
along K; 0 = whole row). Energy is measured by
sycl/apps/bench_qkint8_tune.cpp (build-llvm7/afi-bench-qkint8-tune); every
measured config is correctness-gated device-side against the host oracle.

The afi.tuner Tuner class is reused by patching its module globals (DIMS,
TEMPLATE, Config, domains, admissible) — the methods read them at call
time. No code is vendored.

CLI: python -m afi.tuner_qkint8 --d 128 --grain decode --budget 300
     python -m afi.tuner_qkint8 --d 128 --rq 2048 --rk 2048 --budget 300
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import afi.tuner as _t

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = REPO_ROOT / "build-llvm7" / "afi-bench-qkint8-tune"

# Known-good template: Phase 0 / first-check default.
TEMPLATE = {"tm": 8, "sg": 32, "wg_k": 4, "wg_q": 1, "pad": 0, "kt": 0}

DIMS = ("tm", "sg", "wg_k", "wg_q", "pad", "kt")

GRAINS = {"decode": (28, 2048), "prefill": (2048, 2048)}


@dataclass(frozen=True)
class Config:
    tm: int = 8    # DPAS m, {4,8}
    sg: int = 32   # sub-group size, {16,32}
    wg_k: int = 4  # subgroups per WG = 16-key tiles in flight, {1,2,4,8}
    wg_q: int = 1  # Q row-tiles (tm rows) per WG, {1,2,4}
    pad: int = 0   # SLM leading-dim pad, {0,8} (pad%8!=0 → DPAS garbage)
    kt: int = 0    # 16-key tiles per WG along K, {0,1,2,4,8,16}

    def cli(self):
        return (f"tm={self.tm},sg={self.sg},wg_k={self.wg_k},"
                f"wg_q={self.wg_q},pad={self.pad},kt={self.kt}")


def domains(d):
    """Admissible values per dim — must match scan_configs in
    sycl/apps/bench_qkint8_tune.cpp."""
    return {
        "tm": (4, 8),
        "sg": (16, 32),
        "wg_k": (1, 2, 4, 8),
        "wg_q": (1, 2, 4),
        "pad": (0, 8),
        "kt": (0, 1, 2, 4, 8, 16),
    }


def admissible(cfg, d, limits=_t.Limits()):
    """Host mirror of qkint8_config_ok (sycl/src/qkint8.cpp)."""
    if d not in (128, 256):
        return False
    if cfg.tm not in (4, 8):
        return False
    if cfg.sg not in limits.subgroup_sizes:
        return False
    if not 1 <= cfg.wg_k <= 8 or not 1 <= cfg.wg_q <= 4:
        return False
    if cfg.pad not in (0, 8):
        return False
    if cfg.kt < 0:
        return False
    if cfg.wg_k * cfg.sg > limits.max_work_group_size:
        return False
    qb = cfg.tm * cfg.wg_q
    ld = d + cfg.pad
    slm = (qb * ld * 6 + qb * d + cfg.wg_k * 16 * ld * 6
           + cfg.wg_k * 16 * d + cfg.wg_k * qb * 16 * 4
           + (qb + cfg.wg_k * 16) * 4 + 512)
    return slm <= limits.local_mem_size


class BenchRunner:
    """Energy oracle: subprocess to afi-bench-qkint8-tune, batched."""

    def __init__(self, binary=DEFAULT_BINARY, device=0, rq=28, rk=2048,
                 iters=50, elem="fp16"):
        self.binary = str(binary)
        self.device = device
        self.rq = rq
        self.rk = rk
        self.iters = iters
        self.elem = elem

    def measure(self, cfgs, d):
        """-> {Config: ms or None} (None = illegal/wrong/error)."""
        cmd = [self.binary, "-d", str(self.device),
               "--rq", str(self.rq), "--rk", str(self.rk),
               "--iters", str(self.iters), "--elem", self.elem,
               "--d", str(d)]
        for c in cfgs:
            cmd += ["--config", c.cli()]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        res = {}
        for line in out.stdout.splitlines():
            if line.startswith(("#", "d,")) or not line.strip():
                continue
            parts = line.split(",")
            if len(parts) < 13:
                continue
            cfg = Config(tm=int(parts[4]), sg=int(parts[5]),
                         wg_k=int(parts[6]), wg_q=int(parts[7]),
                         pad=int(parts[8]), kt=int(parts[9]))
            status = parts[10]
            res[cfg] = float(parts[12]) if status == "ok" else None
        return res


# Patch the Layer 2 Tuner's globals; its methods read them at call time.
_t.DIMS = DIMS
_t.TEMPLATE = TEMPLATE
_t.Config = Config
_t.domains = domains
_t.admissible = admissible
load_limits = _t.load_limits


class Tuner(_t.Tuner):
    """_t.Tuner with the dim-name-hardcoded methods made generic."""

    def _full_lattice(self):
        import itertools
        dom = domains(self.n)
        for vals in itertools.product(*(dom[k] for k in DIMS)):
            c = Config(**dict(zip(DIMS, vals)))
            if admissible(c, self.n, self.limits):
                yield c

    def backtrack(self, node_limit=64):
        order = ("sg", "tm", "wg_k", "wg_q", "pad", "kt")
        leaves = []

        def rec(k, assignment, budget):
            if budget[0] <= 0:
                return
            if k == len(order):
                cfg = Config(**assignment)
                if admissible(cfg, self.n, self.limits):
                    leaves.append(cfg)
                    budget[0] -= 1
                return
            dim = order[k]
            dom = self.csp(assignment)[dim]
            for v in dom:
                rec(k + 1, {**assignment, dim: v}, budget)

        rec(0, {}, [node_limit])
        self.measure_many(leaves)
        return self.best


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="QkInt8RotKernel launch-config tuner")
    p.add_argument("--d", type=int, default=128, choices=[128, 256])
    p.add_argument("--grain", choices=sorted(GRAINS), default=None)
    p.add_argument("--rq", type=int, default=None)
    p.add_argument("--rk", type=int, default=None)
    p.add_argument("--elem", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--budget", type=float, default=300.0,
                   help="ILS time budget, seconds")
    p.add_argument("--binary", default=str(DEFAULT_BINARY))
    p.add_argument("--out", default=None, help="cache JSON path")
    args = p.parse_args(argv)
    rq, rk = GRAINS[args.grain] if args.grain else (
        args.rq or 28, args.rk or 2048)

    runner = BenchRunner(binary=args.binary, device=args.device, rq=rq, rk=rk,
                         iters=args.iters, elem=args.elem)
    limits = load_limits(device=args.device)
    tuner = Tuner(args.d, elem=args.elem, limits=limits, runner=runner)
    best_cfg, best_ms, log = tuner.tune_ils(
        budget_s=args.budget,
        progress_callback=lambda m, b: print(
            f"[d={args.d} rq={rq} rk={rk}] {m}: best {b[0].cli()} = "
            f"{b[1]:.4f} ms" if b else f"[d={args.d}] {m}: no valid config"))
    print(f"d={args.d} rq={rq} rk={rk} best: "
          f"{best_cfg.cli() if best_cfg else '-'} "
          f"{best_ms if best_ms is not None else float('nan'):.4f} ms "
          f"({len(tuner.cache)} measured)")
    out = args.out or (REPO_ROOT / "logs" /
                       f"tune-qkint8-d{args.d}-rq{rq}-rk{rk}-{args.elem}.json")
    tuner.save_cache(out)
    print(f"cache -> {out}")


if __name__ == "__main__":
    main()
