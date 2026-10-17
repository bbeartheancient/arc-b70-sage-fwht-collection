"""Layer 2 XMX launch-config autotuner.

sudoku.py's method skeleton remapped from Hadamard rows to FWHT XMX launch
configs: digits -> dim values, pairwise orthogonality -> launch legality,
residual energy -> measured ms. Same METHODS cycle (backtrack / overlay /
csp / exact / residual / stochastic) driven by tune_ils.

Nothing here imports hoa64: the search domain is launch configs, not ±1
rows, so the constraint model is reimplemented (AGENTS.md: do not vendor
sudoku.py). Energy is measured by sycl/apps/bench_xmx_tune.cpp
(build-llvm7/afi-bench-xmx-tune); host tests inject a stub runner instead.

CLI: python -m afi.tuner --n 128 256 --elem fp16 --device 0 --budget 600
"""

import json
import math
import random
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path

METHODS = ("overlay", "exact", "csp", "backtrack", "residual", "stochastic")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = REPO_ROOT / "build-llvm7" / "afi-bench-xmx-tune"

# Known-good DPAS template: today's hardcoded kernel config (8x16x16 atom,
# one WG per 8 rows, subgroups spanning the row).
TEMPLATE = {"tm": 8, "sg": 16, "wg_n": 0, "wg_m": 1, "pad": 0, "pf": 0}

DIMS = ("tm", "sg", "wg_n", "wg_m", "pad", "pf")


@dataclass(frozen=True)
class Config:
    tm: int = 8        # DPAS m, {1,2,4,8}
    sg: int = 16       # sub-group size, {16,32}
    wg_n: int = 0      # subgroups per WG along the row; 0 = n/16 (legacy)
    wg_m: int = 1      # row-tiles per WG
    pad: int = 0       # SLM leading-dim padding
    pf: int = 0        # prefetch double-buffer

    def cli(self):
        return (f"tm={self.tm},sg={self.sg},wg_n={self.wg_n},"
                f"wg_m={self.wg_m},pad={self.pad},pf={self.pf}")


@dataclass(frozen=True)
class Limits:
    """Device limits for the legality predicate (mirrors fwht_xmx_config_ok).
    Defaults match inventory/b70.json (BMG-G31)."""
    subgroup_sizes: tuple = (16, 32)
    max_work_group_size: int = 1024
    local_mem_size: int = 131072


def domains(n):
    """Admissible values per dim — must match scan_configs in
    sycl/apps/bench_xmx_tune.cpp."""
    return {
        "tm": (1, 2, 4, 8),
        "sg": (16, 32),
        "wg_n": tuple(w for w in (1, 2, 4, 8, 16) if w * 16 <= n),
        "wg_m": (1, 2, 4),
        "pad": (0, 8),  # DPAS ldm must stay 16B-aligned; pad=1 computes garbage
        "pf": (0, 1),
    }


def admissible(cfg, n, limits=Limits()):
    """Host mirror of fwht_xmx_config_ok (sycl/src/fwht.cpp)."""
    if cfg.tm not in (1, 2, 4, 8):
        return False
    if cfg.sg not in limits.subgroup_sizes:
        return False
    wg_n = cfg.wg_n if cfg.wg_n > 0 else n // 16
    if wg_n < 1 or wg_n * 16 > n:
        return False
    if cfg.wg_m < 1 or cfg.pf not in (0, 1):
        return False
    # Measured on BMG-G31: tm=1/sg=32/wg_n>1 launches but computes garbage.
    if cfg.tm == 1 and cfg.sg == 32 and (cfg.wg_n or n // 16) > 1:
        return False
    if cfg.pad < 0 or cfg.pad > 16 or cfg.pad % 8 != 0:
        return False  # DPAS ldm 16B alignment (see fwht_xmx_config_ok)
    local = wg_n * cfg.sg
    if local > limits.max_work_group_size:
        return False
    if cfg.pf and (cfg.tm * n + local - 1) // local > 8:
        return False
    slm = cfg.tm * (n + cfg.pad) * 6 + 512  # fp32 tile + 2B leaf tile + H16
    return slm <= limits.local_mem_size


def load_limits(path=None, device=0):
    """Best-effort limits from an inventory dump; falls back to defaults."""
    path = Path(path) if path else REPO_ROOT / "inventory" / "b70.json"
    try:
        data = json.loads(path.read_text())
        dev = data["devices"][device]
        return Limits(
            subgroup_sizes=tuple(dev.get("subgroup_sizes", (16, 32))),
            max_work_group_size=int(dev.get("max_work_group_size", 1024)),
            local_mem_size=int(dev.get("local_mem_bytes", 131072)),
        )
    except Exception:
        return Limits()


class BenchRunner:
    """Energy oracle: subprocess to afi-bench-xmx-tune, batched per call."""

    def __init__(self, binary=DEFAULT_BINARY, device=0, rows=4096, iters=20,
                 elem="fp16"):
        self.binary = str(binary)
        self.device = device
        self.rows = rows
        self.iters = iters
        self.elem = elem

    def measure(self, cfgs, n):
        """-> {Config: ms or None} (None = illegal/wrong/error)."""
        cmd = [self.binary, "-d", str(self.device), "-r", str(self.rows),
               "--iters", str(self.iters), "--elem", self.elem, "--n", str(n)]
        for c in cfgs:
            cmd += ["--config", c.cli()]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        res = {}
        for line in out.stdout.splitlines():
            if line.startswith(("#", "n,")) or not line.strip():
                continue
            parts = line.split(",")
            if len(parts) < 13:
                continue
            cfg = Config(tm=int(parts[3]), sg=int(parts[4]),
                         wg_n=int(parts[5]), wg_m=int(parts[6]),
                         pad=int(parts[7]), pf=int(parts[8]))
            status = parts[9]
            res[cfg] = float(parts[11]) if status == "ok" else None
        return res


class Tuner:
    """ILS over launch configs. runner is anything with .measure(cfgs, n)."""

    def __init__(self, n, elem="fp16", limits=None, runner=None, seed=0):
        self.n = n
        self.elem = elem
        self.limits = limits or Limits()
        self.runner = runner
        self.rng = random.Random(seed)
        self.cache = {}   # Config -> ms or None (the "accepted rows" analog)
        self.best = None  # (Config, ms)
        self.log = {"n": n, "elem": elem, "methods": {}, "history": []}

    # -- energy -------------------------------------------------------------
    def measure(self, cfg):
        if cfg not in self.cache:
            self.measure_many([cfg])
        return self.cache[cfg]

    def measure_many(self, cfgs):
        todo = [c for c in cfgs if c not in self.cache]
        if todo and self.runner is not None:
            self.cache.update(self.runner.measure(todo, self.n))
        for c in cfgs:
            ms = self.cache.get(c)
            if ms is not None and (self.best is None or ms < self.best[1]):
                self.best = (c, ms)
                self.log["history"].append({"cfg": c.cli(), "ms": ms})
        return {c: self.cache.get(c) for c in cfgs}

    def _ranked(self):
        ok = [(c, ms) for c, ms in self.cache.items() if ms is not None]
        return sorted(ok, key=lambda kv: kv[1])

    # -- methods (sudoku.py skeleton) ----------------------------------------
    def overlay(self):
        """Place the known-good DPAS template (today's hardcoded config)."""
        self.measure(Config(**TEMPLATE))
        return self.best

    def exact(self):
        """Exhaustive pass over the fully-discrete core dims (tm x sg) with
        the runtime dims at their template values."""
        cfgs = [replace(Config(**TEMPLATE), tm=tm, sg=sg)
                for tm in domains(self.n)["tm"] for sg in domains(self.n)["sg"]
                if admissible(replace(Config(**TEMPLATE), tm=tm, sg=sg),
                              self.n, self.limits)]
        self.measure_many(cfgs)
        return self.best

    def csp(self, assignment=None):
        """AC-3 over the dim domains: wipe values that no admissible config
        can use given the partial assignment. Returns reduced domains."""
        dom = {d: list(v) for d, v in domains(self.n).items()}
        assignment = dict(assignment or {})
        for d, v in assignment.items():
            dom[d] = [v]
        changed = True
        while changed:
            changed = False
            for dim in DIMS:
                if dim in assignment:
                    continue
                keep = []
                for v in dom[dim]:
                    probe = dict(assignment)
                    probe[dim] = v
                    # value survives if some completion of the remaining dims
                    # is admissible (arc consistency against the full predicate)
                    if any(admissible(Config(**{**TEMPLATE, **probe, **rest}),
                                      self.n, self.limits)
                           for rest in self._completions(probe)):
                        keep.append(v)
                if len(keep) < len(dom[dim]):
                    dom[dim] = keep
                    changed = True
        return dom

    def _completions(self, assignment, cap=64):
        """Sample completions of the unassigned dims (bounded)."""
        free = [d for d in DIMS if d not in assignment]
        out = [{}]
        for d in free:
            out = [{**o, d: v} for o in out for v in domains(self.n)[d]]
            if len(out) > cap:
                self.rng.shuffle(out)
                out = out[:cap]
        return out

    def backtrack(self, node_limit=64):
        """DFS with MRV: assign most-constrained dims first (sg, tm), prune
        branches whose partial assignment has no admissible completion."""
        order = ("sg", "tm", "wg_n", "wg_m", "pad", "pf")
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

    def residual(self):
        """Repair: flip each dim of the best config toward the runner-up's
        value (the 'worst pair' analog), remeasure."""
        ranked = self._ranked()
        if not ranked:
            return None
        best = ranked[0][0]
        cfgs = []
        for dim in DIMS:
            for v in domains(self.n)[dim]:
                if v != getattr(best, dim):
                    c = replace(best, **{dim: v})
                    if admissible(c, self.n, self.limits):
                        cfgs.append(c)
        self.measure_many(cfgs)
        return self.best

    def stochastic(self, t_start=1.0, t_end=0.01, cooling=0.95, max_steps=32):
        """SA over the leftover dims around the best skeleton."""
        if self.best is None:
            return None
        cur, e_cur = self.best
        temp = t_start
        for _ in range(max_steps):
            dim = self.rng.choice(DIMS)
            vals = [v for v in domains(self.n)[dim] if v != getattr(cur, dim)]
            if not vals:
                continue
            nxt = replace(cur, **{dim: self.rng.choice(vals)})
            if not admissible(nxt, self.n, self.limits):
                continue
            e_nxt = self.measure(nxt)
            if e_nxt is None:
                continue
            if e_nxt < e_cur or self.rng.random() < math.exp(
                    -(e_nxt - e_cur) / max(temp * e_cur, 1e-12)):
                cur, e_cur = nxt, e_nxt
            temp = max(temp * cooling, t_end)
        return self.best

    # -- ILS driver -----------------------------------------------------------
    def tune_ils(self, budget_s=None, node_limit=64, progress_callback=None):
        """Cycle METHODS under a time budget; constructive first, then repair
        + SA bursts. Returns (best_config, best_ms, log)."""
        deadline = time.monotonic() + budget_s if budget_s else None
        for method in METHODS:
            if deadline and time.monotonic() > deadline:
                break
            fn = getattr(self, method)
            if method == "backtrack":
                fn(node_limit=node_limit)
            else:
                fn()
            self.log["methods"][method] = (
                {"cfg": self.best[0].cli(), "ms": self.best[1]}
                if self.best else None)
            if progress_callback:
                progress_callback(method, self.best)
            # stop early once the lattice is fully measured
            total = sum(1 for _ in self._full_lattice())
            if len(self.cache) >= total:
                break
        return self.best[0] if self.best else None, \
            self.best[1] if self.best else None, self.log

    def _full_lattice(self):
        dom = domains(self.n)
        for tm in dom["tm"]:
            for sg in dom["sg"]:
                for wg_n in dom["wg_n"]:
                    for wg_m in dom["wg_m"]:
                        for pad in dom["pad"]:
                            for pf in dom["pf"]:
                                c = Config(tm=tm, sg=sg, wg_n=wg_n, wg_m=wg_m,
                                           pad=pad, pf=pf)
                                if admissible(c, self.n, self.limits):
                                    yield c

    def save_cache(self, path):
        data = {c.cli(): ms for c, ms in self.cache.items()}
        Path(path).write_text(json.dumps(
            {"n": self.n, "elem": self.elem, "cache": data,
             "best": {"cfg": self.best[0].cli(), "ms": self.best[1]}
             if self.best else None}, indent=2))


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="XMX FWHT launch-config tuner")
    p.add_argument("--n", type=int, nargs="+", default=[128, 256])
    p.add_argument("--elem", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--rows", type=int, default=4096)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--budget", type=float, default=600.0,
                   help="ILS time budget per n, seconds")
    p.add_argument("--binary", default=str(DEFAULT_BINARY))
    p.add_argument("--out", default=None, help="cache JSON path")
    args = p.parse_args(argv)

    runner = BenchRunner(binary=args.binary, device=args.device,
                         rows=args.rows, iters=args.iters, elem=args.elem)
    limits = load_limits(device=args.device)
    for n in args.n:
        tuner = Tuner(n, elem=args.elem, limits=limits, runner=runner)
        best_cfg, best_ms, log = tuner.tune_ils(
            budget_s=args.budget,
            progress_callback=lambda m, b: print(
                f"[n={n}] {m}: best {b[0].cli()} = {b[1]:.4f} ms"
                if b else f"[n={n}] {m}: no valid config"))
        print(f"n={n} best: {best_cfg.cli() if best_cfg else '-'} "
              f"{best_ms if best_ms is not None else float('nan'):.4f} ms "
              f"({len(tuner.cache)} measured)")
        out = args.out or (REPO_ROOT / "logs" /
                           f"tune-xmx-n{n}-{args.elem}.json")
        tuner.save_cache(out)
        print(f"cache -> {out}")


if __name__ == "__main__":
    main()
