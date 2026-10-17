"""Room impulse response for HOA decode — specular acoustic rays vs a room BVH.

Traversal-shaped (docs/hoa64-rtu-backend.md): bounce rays against a triangle
mesh, collect arrivals at the listener, encode each as a delayed SN3D plane
wave. Arithmetic (SH encode/decode) stays on the host vector path (hoa64).

The RTU is optional. A Möller–Trumbore host mesh is the correctness oracle
and the no-libafi_rtu fallback. Gate (shoebox): first-arrival delay matches
||src−lst||/c and first-order image-source delays match the bounce trace.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

C_SOUND = 343.0  # m/s, dry air ~20 °C


def shoebox(lx: float, ly: float, lz: float):
    """Axis-aligned room [0,lx]×[0,ly]×[0,lz], Y up. Inward-facing tris.

    8 vertices, 12 triangles (2 per wall). Winding is CCW when viewed along
    the inward normal so rays from inside hit the front face.
    """
    lx, ly, lz = float(lx), float(ly), float(lz)
    v = np.array([
        [0.0, 0.0, 0.0], [lx, 0.0, 0.0], [lx, ly, 0.0], [0.0, ly, 0.0],
        [0.0, 0.0, lz], [lx, 0.0, lz], [lx, ly, lz], [0.0, ly, lz],
    ], dtype=np.float32)
    # quads as (a,b,c)+(a,c,d), CCW along inward normal
    f = np.array([
        [0, 1, 5], [0, 5, 4],  # floor  y=0,  +Y
        [3, 7, 6], [3, 6, 2],  # ceil   y=ly, -Y
        [0, 3, 2], [0, 2, 1],  # back   z=0,  +Z
        [4, 5, 6], [4, 6, 7],  # front  z=lz, -Z
        [0, 4, 7], [0, 7, 3],  # left   x=0,  +X
        [1, 2, 6], [1, 6, 5],  # right  x=lx, -X
    ], dtype=np.uint32)
    return v, f


def aabb(x0, y0, z0, x1, y1, z1):
    """Axis-aligned box as 12 triangles (furniture occluder)."""
    x0, y0, z0, x1, y1, z1 = map(float, (x0, y0, z0, x1, y1, z1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if z1 < z0:
        z0, z1 = z1, z0
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)
    f = np.array([
        [0, 2, 1], [0, 3, 2],  # z=z0
        [4, 5, 6], [4, 6, 7],  # z=z1
        [0, 1, 5], [0, 5, 4],  # y=y0
        [3, 7, 6], [3, 6, 2],  # y=y1
        [0, 4, 7], [0, 7, 3],  # x=x0
        [1, 2, 6], [1, 6, 5],  # x=x1
    ], dtype=np.uint32)
    return v, f


def furniture_mesh(pieces):
    """Concatenate AABB pieces → (verts, faces). Each piece is 6 floats."""
    vs, fs, off = [], [], 0
    for p in pieces:
        if len(p) < 6:
            continue
        v, f = aabb(*p[:6])
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    if not vs:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32)
    return np.concatenate(vs), np.concatenate(fs)


def compose_room(box, furniture=None):
    v, f = shoebox(*box)
    if furniture:
        fv, ff = furniture_mesh(furniture)
        if len(fv):
            f = np.concatenate([f, ff + len(v)])
            v = np.concatenate([v, fv])
    return v, f


def sabine_rt60(box, absorption: float, furniture=None) -> dict:
    """Sabine and Eyring T60 for a shoebox, optional furniture surface area.

    T60_sabine = 0.161 V / A, A = Σ S_i α_i (uniform α). Eyring uses
    −0.161 V / (S ln(1−α)). Specular MC is not a diffuse field — these are
    the closed-form labels; measured T20 is the IR number.
    """
    lx, ly, lz = (float(box[0]), float(box[1]), float(box[2]))
    V = lx * ly * lz
    S = 2.0 * (lx * ly + lx * lz + ly * lz)
    a = float(max(1e-6, min(0.99, absorption)))
    n_furn = 0
    if furniture:
        for p in furniture:
            if len(p) < 6:
                continue
            dx = abs(float(p[3]) - float(p[0]))
            dy = abs(float(p[4]) - float(p[1]))
            dz = abs(float(p[5]) - float(p[2]))
            S += 2.0 * (dx * dy + dx * dz + dy * dz)
            n_furn += 1
    A = S * a
    sabine = 0.161 * V / A
    eyring = -0.161 * V / (S * math.log(1.0 - a))
    return {
        "V_m3": round(V, 3),
        "S_m2": round(S, 3),
        "absorption": a,
        "n_furniture": n_furn,
        "t60_sabine_s": round(sabine, 4),
        "t60_eyring_s": round(eyring, 4),
    }


def schroeder_times(arrivals) -> dict:
    """T20/T60 from reverse-integrated arrival energy (Schroeder)."""
    if not arrivals:
        return {"t20_s": None, "t60_s": None, "n": 0}
    t = np.array([float(a["delay_s"]) for a in arrivals], dtype=np.float64)
    e = np.array([float(a["gain"]) ** 2 for a in arrivals], dtype=np.float64)
    order = np.argsort(t)
    t, e = t[order], e[order]
    t = t - t[0]
    cume = np.cumsum(e[::-1])[::-1]
    if cume[0] <= 0:
        return {"t20_s": None, "t60_s": None, "n": int(len(t))}
    db = 10.0 * np.log10(np.maximum(cume / cume[0], 1e-12))

    def _cross(level):
        i = np.where(db <= level)[0]
        if not len(i):
            return None
        k = int(i[0])
        if k == 0:
            return float(t[0])
        # linear interp in dB vs time
        d0, d1 = float(db[k - 1]), float(db[k])
        if d1 == d0:
            return float(t[k])
        w = (level - d0) / (d1 - d0)
        return float(t[k - 1] + w * (t[k] - t[k - 1]))

    t20 = _cross(-20.0)
    t60 = _cross(-60.0)
    return {
        "t20_s": None if t20 is None else round(t20, 4),
        "t60_s": None if t60 is None else round(t60, 4),
        "t60_from_t20_s": None if t20 is None else round(3.0 * t20, 4),
        "n": int(len(t)),
    }


def image_sources(src, lst, box, max_order: int = 1):
    """First-order (and optional higher) image sources of a shoebox.

    Returns a list of dicts: {kind, bounces, image, dist_m, delay_s,
    arrival_dir} where arrival_dir is the unit vector at the listener
    (from listener toward the image — the direction the sound arrives
    FROM is the opposite of the outgoing look, i.e. -arrival? No:
    sound arrives *from* the image direction, so HOA encode uses the
    unit vector (image - listener) as the source direction on the sphere.
    """
    src = np.asarray(src, dtype=np.float64)
    lst = np.asarray(lst, dtype=np.float64)
    lx, ly, lz = (float(box[0]), float(box[1]), float(box[2]))
    out = []
    d0 = float(np.linalg.norm(src - lst))
    if d0 < 1e-6:
        return out
    look = (src - lst) / d0
    out.append({
        "kind": "direct", "bounces": 0, "image": src.tolist(),
        "dist_m": d0, "delay_s": d0 / C_SOUND, "arrival_dir": look.tolist(),
        "gain": 1.0 / d0,
    })
    if max_order < 1:
        return out
    reflections = [
        (np.array([-src[0], src[1], src[2]]), 1),
        (np.array([2 * lx - src[0], src[1], src[2]]), 1),
        (np.array([src[0], -src[1], src[2]]), 1),
        (np.array([src[0], 2 * ly - src[1], src[2]]), 1),
        (np.array([src[0], src[1], -src[2]]), 1),
        (np.array([src[0], src[1], 2 * lz - src[2]]), 1),
    ]
    for img, b in reflections:
        d = float(np.linalg.norm(img - lst))
        if d < 1e-6:
            continue
        look = (img - lst) / d
        out.append({
            "kind": "image", "bounces": b, "image": img.tolist(),
            "dist_m": d, "delay_s": d / C_SOUND, "arrival_dir": look.tolist(),
            "gain": 1.0 / d,
        })
    return out


def _mt_batch(origins, dirs, v0, v1, v2, eps=1e-7):
    """Vectorized Möller–Trumbore: t, hit-mask for one triangle."""
    e1 = v1 - v0
    e2 = v2 - v0
    pvec = np.cross(dirs, e2)
    det = pvec @ e1
    mask = np.abs(det) > eps
    inv = np.zeros_like(det)
    inv[mask] = 1.0 / det[mask]
    tvec = origins - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv
    qvec = np.cross(tvec, e1)
    v = np.einsum("ij,ij->i", dirs, qvec) * inv
    t = (qvec @ e2) * inv
    hit = mask & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-4)
    return t.astype(np.float32), hit


class HostMesh:
    """CPU closest-hit / occlusion over a triangle list (room-scale)."""

    def __init__(self, verts, faces):
        self.verts = np.ascontiguousarray(verts, dtype=np.float32)
        self.faces = np.ascontiguousarray(faces, dtype=np.uint32)
        self.ntri = int(self.faces.shape[0])
        nrm = []
        for i0, i1, i2 in self.faces:
            e1 = self.verts[i1] - self.verts[i0]
            e2 = self.verts[i2] - self.verts[i0]
            n = np.cross(e1, e2)
            nrm.append(n / max(float(np.linalg.norm(n)), 1e-12))
        self.normals = np.asarray(nrm, dtype=np.float32)

    def closest(self, origins, dirs, tmax=None):
        o = np.ascontiguousarray(origins, dtype=np.float32)
        d = np.ascontiguousarray(dirs, dtype=np.float32)
        if o.ndim == 1:
            o = o[None, :]
            d = d[None, :]
        n = o.shape[0]
        if tmax is None:
            cap = np.full(n, 1e30, dtype=np.float32)
        else:
            cap = np.broadcast_to(
                np.ascontiguousarray(tmax, dtype=np.float32), (n,)).copy()
        best_t = cap.copy()
        best_p = np.full(n, -1, dtype=np.int32)
        for pi, (i0, i1, i2) in enumerate(self.faces):
            t, hit = _mt_batch(o, d, self.verts[i0], self.verts[i1],
                               self.verts[i2])
            better = hit & (t < best_t)
            best_t = np.where(better, t, best_t)
            best_p = np.where(better, pi, best_p)
        miss = best_p < 0
        best_t = np.where(miss, np.float32(-1.0), best_t)
        return best_t, best_p

    def occlusion(self, origins, dirs, tmax):
        t, p = self.closest(origins, dirs, tmax)
        return (p >= 0) & (t > 0) & (t < np.asarray(tmax) - 1e-4)

    def prim_normal(self, prim_id: int) -> np.ndarray:
        return self.normals[int(prim_id)]


def fibonacci_dirs(n: int) -> np.ndarray:
    i = np.arange(int(n), dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.cos(phi)
    z = np.sin(phi) * np.sin(theta)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _rtu_mesh(verts, faces):
    try:
        repo_python = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if repo_python not in sys.path:
            sys.path.insert(0, repo_python)
        from afi import rtu  # noqa: WPS433
        rtu.init(int(os.environ.get("AFI_RTU_GPU", "1")))
        return rtu.MeshBLAS(verts, faces)
    except Exception:
        return None


def _engine_closest(engine, origins, dirs, tmax):
    if hasattr(engine, "horizon") and hasattr(engine, "_handle"):
        rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
        return engine.horizon(rays, tmax)
    return engine.closest(origins, dirs, tmax)


def _engine_occluded(engine, origins, dirs, tmax):
    if hasattr(engine, "occlusion") and hasattr(engine, "_handle"):
        rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
        return engine.occlusion(rays, tmax)
    return engine.occlusion(origins, dirs, tmax)


def trace_ir(src, lst, box=(8.0, 3.0, 6.0), n_rays: int = 2048,
             max_bounce: int = 4, absorption: float = 0.2,
             src_radius: float = 0.15, engine=None, furniture=None):
    """Specular bounce trace from the listener; record hits on a source sphere.

    Reciprocity: directions at the listener *are* the initial ray dirs, which
    is the HOA arrival direction. Energy = (1−α)^bounces / path_length.

    Returns dict with arrivals + engine name + wall time.
    """
    src = np.asarray(src, dtype=np.float32)
    lst = np.asarray(lst, dtype=np.float32)
    lx, ly, lz = (float(box[0]), float(box[1]), float(box[2]))
    verts, faces = compose_room((lx, ly, lz), furniture)
    host = HostMesh(verts, faces)
    if engine is None:
        engine = _rtu_mesh(verts, faces) or host
        engine_name = "rtu" if engine is not host else "host-mt"
    else:
        engine_name = "injected"
    R = 1.0 - float(absorption)
    dirs0 = fibonacci_dirs(int(n_rays))
    org = np.tile(lst, (len(dirs0), 1)).astype(np.float32)
    dirs = dirs0.copy()
    path = np.zeros(len(dirs0), dtype=np.float32)
    alive = np.ones(len(dirs0), dtype=bool)
    energy = np.ones(len(dirs0), dtype=np.float64)
    arrivals = []
    t0 = time.perf_counter()
    # Direct path (analytic, gated by occlusion)
    delta = src - lst
    d0 = float(np.linalg.norm(delta))
    if d0 > 1e-4:
        u = (delta / d0).astype(np.float32)
        blocked = bool(_engine_occluded(
            engine, lst[None, :], u[None, :],
            np.array([d0], dtype=np.float32))[0])
        if not blocked:
            arrivals.append({
                "kind": "direct", "bounces": 0, "dist_m": d0,
                "delay_s": d0 / C_SOUND, "gain": 1.0 / d0,
                "arrival_dir": u.tolist(),
            })
    # max_bounce reflections ⇒ max_bounce+1 closest-hit steps (the last
    # step is the post-reflect segment that can still catch the source).
    for bounce in range(int(max_bounce) + 1):
        if not alive.any():
            break
        idx = np.nonzero(alive)[0]
        tmax = np.full(len(idx), 1e5, dtype=np.float32)
        t, prim = _engine_closest(engine, org[idx], dirs[idx], tmax)
        hit = (prim >= 0) & (t > 0)
        # source-sphere intercept along this segment, before the wall
        rel = org[idx] - src
        bq = 2.0 * np.einsum("ij,ij->i", rel, dirs[idx])
        cq = np.einsum("ij,ij->i", rel, rel) - src_radius ** 2
        disc = bq * bq - 4.0 * cq
        sphere = disc >= 0
        sqrt_d = np.sqrt(np.maximum(disc, 0.0))
        s1 = (-bq - sqrt_d) * 0.5
        s2 = (-bq + sqrt_d) * 0.5
        s_hit = np.where((s1 > 1e-4) & sphere, s1,
                         np.where((s2 > 1e-4) & sphere, s2, -1.0))
        cap = np.where(hit, t, 1e5)
        caught = (s_hit > 1e-4) & (s_hit < cap)
        for j in np.nonzero(caught)[0]:
            gi = idx[j]
            dist = float(path[gi] + s_hit[j])
            if dist < 1e-4:
                continue
            arrivals.append({
                "kind": "bounce", "bounces": bounce,
                "dist_m": dist, "delay_s": dist / C_SOUND,
                "gain": float(energy[gi] / dist),
                "arrival_dir": dirs0[gi].tolist(),
            })
            alive[gi] = False
        still = hit & ~caught
        died = idx[~hit & ~caught]
        alive[died] = False
        if bounce == int(max_bounce) or not still.any():
            alive[idx[still]] = False
            continue
        keep = idx[still]
        tt = t[still]
        pp = prim[still]
        org[keep] = org[keep] + dirs[keep] * tt[:, None]
        path[keep] = path[keep] + tt
        energy[keep] *= R
        nrm = np.stack([
            (engine.prim_normal(int(p)) if hasattr(engine, "prim_normal")
             else host.prim_normal(int(p)))
            for p in pp
        ]).astype(np.float32)
        dn = np.einsum("ij,ij->i", dirs[keep], nrm)
        nrm = np.where(dn[:, None] > 0, -nrm, nrm)
        dn = np.einsum("ij,ij->i", dirs[keep], nrm)
        dirs[keep] = dirs[keep] - 2.0 * dn[:, None] * nrm
        nn = np.linalg.norm(dirs[keep], axis=1, keepdims=True)
        dirs[keep] = dirs[keep] / np.maximum(nn, 1e-12)
        org[keep] = org[keep] + nrm * 1e-3
        alive[keep] = energy[keep] > 1e-4
    wall_ms = (time.perf_counter() - t0) * 1e3
    if hasattr(engine, "close") and engine is not host:
        try:
            engine.close()
        except Exception:
            pass
    return {
        "engine": engine_name,
        "n_rays": int(n_rays),
        "max_bounce": int(max_bounce),
        "n_arrivals": len(arrivals),
        "arrivals": arrivals,
        "trace_ms": round(wall_ms, 3),
        "box_m": [lx, ly, lz],
        "src_m": [round(float(v), 3) for v in src],
        "lst_m": [round(float(v), 3) for v in lst],
        "absorption": float(absorption),
        "c_m_s": C_SOUND,
        "n_tris": int(len(faces)),
        "n_furniture": 0 if not furniture else len(furniture),
    }


def encode_hoa(arrivals, fs: float = 48000.0, duration_s: float = 0.25,
               order: int = 3):
    """Bin arrivals into an SN3D HOA IR (C, T) via hoa64.encode.encode_xyz."""
    n_ch = (int(order) + 1) ** 2
    n_t = max(8, int(round(float(duration_s) * float(fs))))
    ir = np.zeros((n_ch, n_t), dtype=np.float64)
    if not arrivals:
        return ir
    dirs_room = np.array([a["arrival_dir"] for a in arrivals], dtype=np.float64)
    # Room is Y-up (x east, y up, z south). hoa64 SH is z-up
    # (x=cos az cos el, y=sin az cos el, z=sin el). Swap y↔z.
    dirs = np.stack([dirs_room[:, 0], dirs_room[:, 2], dirs_room[:, 1]], axis=1)
    gains = np.array([a["gain"] for a in arrivals], dtype=np.float64)
    delays = np.array([a["delay_s"] for a in arrivals], dtype=np.float64)
    bins = np.clip(np.round(delays * fs).astype(int), 0, n_t - 1)
    try:
        root = os.environ.get("HOA64_ROOT", "/home/bbear")
        if os.path.isdir(os.path.join(root, "hoa64")) and root not in sys.path:
            sys.path.insert(0, root)
        from hoa64.basis import sh_sn3d_batch  # noqa: WPS433
        Y = sh_sn3d_batch(dirs, max_order=int(order))  # (N, C)
    except Exception:
        # omni-only fallback: Y00 = 1/sqrt(4π)
        Y = np.zeros((len(arrivals), n_ch), dtype=np.float64)
        Y[:, 0] = 1.0 / math.sqrt(4.0 * math.pi)
    for i, b in enumerate(bins):
        nuse = min(n_ch, Y.shape[1])
        ir[:nuse, b] += Y[i, :nuse] * gains[i]
    return ir


def analyze_ir(ir, fs: float, order: int):
    """Peak delay, RT-ish decay, HOA peak direction of the first frame."""
    n_ch, n_t = ir.shape
    w = ir[0]
    energy = np.sum(ir * ir, axis=0)
    peak_bin = int(np.argmax(np.abs(w))) if n_t else 0
    peak_delay_s = peak_bin / float(fs)
    # T20-ish from energy envelope (coarse; shoebox gate uses delay, not this)
    cume = np.cumsum(energy[::-1])[::-1]
    if cume[0] > 0:
        db = 10.0 * np.log10(np.maximum(cume / cume[0], 1e-12))
        t20 = np.where(db <= -20.0)[0]
        t20_s = float(t20[0] / fs) if len(t20) else None
    else:
        t20_s = None
    peak_az = peak_el = None
    try:
        root = os.environ.get("HOA64_ROOT", "/home/bbear")
        if os.path.isdir(os.path.join(root, "hoa64")) and root not in sys.path:
            sys.path.insert(0, root)
        from hoa64.analysis import peak_direction  # noqa: WPS433
        az, el, _val = peak_direction(ir[:, peak_bin], max_order=int(order))
        peak_az, peak_el = float(az), float(el)
    except Exception:
        pass
    return {
        "peak_delay_s": round(peak_delay_s, 6),
        "peak_bin": peak_bin,
        "t20_s": None if t20_s is None else round(t20_s, 4),
        "peak_az_deg": None if peak_az is None else round(peak_az, 2),
        "peak_el_deg": None if peak_el is None else round(peak_el, 2),
        "n_channels": n_ch,
        "n_samples": n_t,
        "fs": float(fs),
        "w_peak": round(float(w[peak_bin]), 6),
        "energy": [round(float(v), 8) for v in energy[:: max(1, n_t // 64)]],
    }


def room_impulse(src, lst, box=(8.0, 3.0, 6.0), n_rays: int = 1024,
                 max_bounce: int = 3, absorption: float = 0.2,
                 fs: float = 48000.0, duration_s: float = 0.2,
                 order: int = 3, engine=None, furniture=None) -> dict:
    """Calculator entry: trace + HOA encode + analysis, JSON-safe."""
    src = [float(v) for v in src]
    lst = [float(v) for v in lst]
    box = [float(v) for v in box]
    if any(src[i] <= 0.05 or src[i] >= box[i] - 0.05 for i in range(3)):
        return {"error": "source must sit inside the box (5 cm inset)"}
    if any(lst[i] <= 0.05 or lst[i] >= box[i] - 0.05 for i in range(3)):
        return {"error": "listener must sit inside the box (5 cm inset)"}
    pieces = []
    if furniture:
        for p in furniture:
            if p is None or len(p) < 6:
                continue
            q = [float(v) for v in p[:6]]
            pieces.append(q)
            def _inside(pt, a):
                return (min(a[0], a[3]) < pt[0] < max(a[0], a[3])
                        and min(a[1], a[4]) < pt[1] < max(a[1], a[4])
                        and min(a[2], a[5]) < pt[2] < max(a[2], a[5]))
            if _inside(src, q) or _inside(lst, q):
                return {"error": "src/lst must not sit inside furniture"}
    n_rays = int(max(16, min(int(n_rays), 8192)))
    max_bounce = int(max(0, min(int(max_bounce), 8)))
    order = int(max(1, min(int(order), 7)))
    tr = trace_ir(src, lst, box=box, n_rays=n_rays, max_bounce=max_bounce,
                  absorption=absorption, engine=engine,
                  furniture=pieces or None)
    ir = encode_hoa(tr["arrivals"], fs=fs, duration_s=duration_s, order=order)
    ana = analyze_ir(ir, fs, order)
    decay = schroeder_times(tr["arrivals"])
    ana.update({f"schroeder_{k}": v for k, v in decay.items()})
    sabine = sabine_rt60(box, absorption, pieces or None)
    images = image_sources(src, lst, box, max_order=1)
    direct = next((a for a in tr["arrivals"] if a["kind"] == "direct"), None)
    d_err = None
    if direct and images and not pieces:
        d_err = abs(direct["delay_s"] - images[0]["delay_s"])
    w = ir[0]
    step = max(1, w.size // 256)
    furn_note = f" · {len(pieces)} boxes" if pieces else ""
    figure = {
        "type": "xy",
        "x": [round(i / fs * 1000.0, 3) for i in range(0, w.size, step)],
        "y": [round(float(v), 6) for v in w[::step]],
        "caption": (
            f"room IR W-channel {box[0]:.1f}×{box[1]:.1f}×{box[2]:.1f} m · "
            f"{tr['engine']} · {tr['n_arrivals']} arrivals{furn_note}"
        ),
    }
    out = {
        **tr,
        "arrivals": [
            {k: (round(v, 5) if isinstance(v, float) else v)
             for k, v in a.items() if k != "image"}
            for a in tr["arrivals"][:48]
        ],
        "order": order,
        "fs": float(fs),
        "duration_s": float(duration_s),
        "analysis": ana,
        "image_direct_delay_s": round(images[0]["delay_s"], 6) if images else None,
        "direct_delay_error_s": None if d_err is None else round(d_err, 7),
        "sabine": sabine,
        "furniture": pieces,
        "direct_occluded": direct is None,
        "figure": figure,
    }
    return out
