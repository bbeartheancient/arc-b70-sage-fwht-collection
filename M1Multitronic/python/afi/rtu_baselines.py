#!/usr/bin/env python
"""M2 baseline arms for the RTU terrain race (docs/hoa64-rtu-backend.md).

Arms:
  1. numpy vectorized heightfield ray-march (step-limited approximation)
  2. hoa64 site_survey Deygout link analysis on the same profile lines
     (LOS verdict parity; diffraction loss is the documented RTU delta)

Input: /tmp/opencode/rtu_terrain_G256.bin written by
`afi-rtu-terrain 1 256 8192 360 --dump` (heights, rays, RTU verdicts).

Usage: python rtu_baselines.py [dump.bin]
"""

import struct
import sys

import numpy as np

sys.path.insert(0, "/home/bbear")
import hoa64.site_survey as ss  # noqa: E402


def load_dump(path):
    with open(path, "rb") as f:
        (G,) = struct.unpack("<I", f.read(4))
        (span,) = struct.unpack("<f", f.read(4))
        h = np.frombuffer(f.read(4 * (G + 1) * (G + 1)), dtype="<f4").copy()
        n_prof = 8192  # matches the dump invocation
        prof_rays = np.frombuffer(f.read(4 * 6 * n_prof), dtype="<f4").copy()
        prof_tmax = np.frombuffer(f.read(4 * n_prof), dtype="<f4").copy()
        prof_rtu = np.frombuffer(
            f.read(12 * n_prof), dtype="<f4"
        ).reshape(n_prof, 3).copy()
        fan_rays = np.frombuffer(f.read(4 * 6 * 360), dtype="<f4").copy()
        fan_tmax = np.frombuffer(f.read(4 * 360), dtype="<f4").copy()
        fan_rtu = np.frombuffer(f.read(12 * 360), dtype="<f4").reshape(360, 3).copy()
    h = h.reshape(G + 1, G + 1)
    return G, float(span), h, prof_rays.reshape(n_prof, 6), prof_tmax, prof_rtu, fan_rays


def bilinear(h, span, x, z):
    G = h.shape[0] - 1
    s = G / span
    fi = np.clip(x * s, 0, G - 1e-4)
    fj = np.clip(z * s, 0, G - 1e-4)
    i = fi.astype(np.int64)
    j = fj.astype(np.int64)
    u = fi - i
    v = fj - j
    return (
        h[j, i] * (1 - u) * (1 - v)
        + h[j, i + 1] * u * (1 - v)
        + h[j + 1, i] * (1 - u) * v
        + h[j + 1, i + 1] * u * v
    )


def raymarch_occlusion(h, span, rays, tmax, step_m=4.0, chunk=4096):
    """Vectorized march: sample ground under the ray, occluded if ground
    rises above the ray line before the segment end (per-ray tmax)."""
    org = rays[:, 0:3]
    d = rays[:, 3:6]
    n = len(rays)
    out = np.zeros(n, dtype=bool)
    for c0 in range(0, n, chunk):
        c1 = min(c0 + chunk, n)
        steps = np.maximum((tmax[c0:c1] / step_m).astype(int), 2)
        smax = steps.max()
        t = np.arange(1, smax + 1)[None, :].astype(np.float64) * step_m
        t = np.minimum(t, tmax[c0:c1][:, None] - 1e-3)
        px = org[c0:c1, 0][:, None] + d[c0:c1, 0][:, None] * t
        py = org[c0:c1, 1][:, None] + d[c0:c1, 1][:, None] * t
        pz = org[c0:c1, 2][:, None] + d[c0:c1, 2][:, None] * t
        valid = t < tmax[c0:c1][:, None] - 1e-2
        g = bilinear(h, span, px, pz)
        blocked = (py < g) & valid
        out[c0:c1] = blocked.any(axis=1)
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/rtu_terrain_G256.bin"
    G, span, h, prof_rays, prof_tmax, prof_rtu, _ = load_dump(path)
    print(f"terrain G={G} span={span:.0f} m")

    # --- arm 1: numpy ray-march ------------------------------------------
    import time

    t0 = time.perf_counter()
    march = raymarch_occlusion(h, span, prof_rays, prof_tmax, step_m=1.0)
    dt_march = time.perf_counter() - t0
    rtu_occ = prof_rtu[:, 2] > 0
    agree = (march == rtu_occ).sum()
    print(f"numpy ray-march: {dt_march * 1e3:.1f} ms for {len(prof_rays)} rays"
          f" ({len(prof_rays) / dt_march / 1e6:.2f} Mray/s)")
    print(f"  vs RTU: {agree}/{len(prof_rtu)} agreement "
          f"({100.0 * agree / len(prof_rtu):.2f}%), "
          f"rtu-occluded {rtu_occ.sum()}, march-occluded {march.sum()}")

    # --- arm 2: hoa64 Deygout on the same profile lines -------------------
    # sample the straight TX->RX line, feed _analyze, compare LOS verdict.
    # tx/rx antenna heights are reconstructed exactly from the dumped rays.
    n_dey = 256
    idx = np.linspace(0, len(prof_rays) - 1, 64).astype(int)
    los_agree = 0
    losses = []
    t0 = time.perf_counter()
    for k in idx:
        org = prof_rays[k, 0:3].astype(np.float64)
        d = prof_rays[k, 3:6].astype(np.float64)
        L = float(prof_tmax[k])
        t = np.linspace(0.0, L, n_dey)
        px = org[0] + d[0] * t
        py = org[1] + d[1] * t
        pz = org[2] + d[2] * t
        inside = (px >= 0) & (px <= span) & (pz >= 0) & (pz <= span)
        if inside.sum() < 10 or not inside[-1]:
            continue
        elev = bilinear(h, span, px[inside], pz[inside])
        dist = t[inside]
        tx_h = org[1] - bilinear(h, span, np.array([org[0]]), np.array([org[2]]))[0]
        rx_abs = org[1] + d[1] * L
        rx_h = rx_abs - bilinear(h, span, np.array([org[0] + d[0] * L]),
                                 np.array([org[2] + d[2] * L]))[0]
        try:
            res = ss._analyze(dist, elev, tx_h=float(tx_h), rx_h=float(rx_h),
                              f_hz=5.8e9)
        except Exception as e:  # noqa: BLE001
            print("  _analyze failed:", e)
            continue
        los = bool(res.get("los", True))
        losses.append(res.get("diffraction_loss_db", 0.0))
        los_agree += int(los == (not rtu_occ[k]))
    dt_dey = time.perf_counter() - t0
    n_done = len(losses)
    print(f"hoa64 Deygout: {n_done} profiles in {dt_dey * 1e3:.1f} ms "
          f"({dt_dey * 1e3 / max(n_done, 1):.2f} ms/profile)")
    print(f"  LOS verdict agreement vs RTU: {los_agree}/{n_done}")
    if losses:
        print(f"  diffraction loss (RTU delta): median "
              f"{np.median(losses):.2f} dB, max {np.max(losses):.2f} dB")


if __name__ == "__main__":
    main()
