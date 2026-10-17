"""rtu — B70 ray-tracing-unit terrain primitives (python/afi individual
functions). ctypes binding over libafi_rtu.so (build with
`bash scripts/with-intel-llvm.sh cmake --build --preset llvm7 --target afi_rtu`).

Individual functions (docs/hoa64-rtu-backend.md step 4 + room-IR):
  rtu.init(gpu_index)                 -> device ready
  rtu.TerrainBLAS(heights, span)      -> opaque BLAS handle (context manager)
  rtu.MeshBLAS(verts, faces)          -> generic triangle BLAS (room-IR)
  blas.occlusion(rays6, tmax)         -> bool[n]   LOS occlusion
  blas.horizon(rays6, tmax)           -> (t, prim)  closest-hit distances
  rtu.los(blas, tx, rx)               -> single-segment convenience
  rtu.horizon_fan(blas, c, h_agl, n)  -> azimuth fan convenience

All coordinates are world meters; heights arrays are (G+1)x(G+1) row-major
float32. Mesh verts are (V,3) float32, faces (T,3) uint32. Rays are
[origin xyz | unit dir xyz] float32, shape (n, 6).
Import fails loudly if the library is missing — callers that treat RTU as
optional should catch ImportError (hoa64 hook does exactly that).
"""

import ctypes as C
import os

import numpy as np

_LIB_PATH = os.environ.get(
    "AFI_RTU_LIB",
    os.path.join(os.path.dirname(__file__), "..", "..", "build-llvm7",
                 "libafi_rtu.so"),
)


def _load_lib():
    try:
        return C.CDLL(os.path.abspath(_LIB_PATH))
    except OSError:
        # libafi_rtu needs the DPC++ runtime (repo convention: AFI_DPCPP_ROOT,
        # default ~/intel-llvm-7.0.0). None of the runtime objects carry
        # usable RUNPATHs for in-process discovery, so preload the chain by
        # absolute path: umf -> ur loader -> L0 adapter -> libsycl. Once a
        # soname is on the loaded-objects list, later NEEDED/dlopen resolves
        # against it.
        root = os.path.expanduser(
            os.environ.get("AFI_DPCPP_ROOT", "~/intel-llvm-7.0.0"))
        libd = os.path.join(root, "lib")
        chain = ["libumf.so.1", "libur_loader.so.0"]
        for a in ("libur_adapter_level_zero_v2.so.0",
                  "libur_adapter_level_zero.so.0"):
            if os.path.exists(os.path.join(libd, a)):
                chain.append(a)
                break
        chain.append("libsycl.so.9")
        for name in chain:
            C.CDLL(os.path.join(libd, name), mode=C.RTLD_GLOBAL)
        return C.CDLL(os.path.abspath(_LIB_PATH))


_lib = _load_lib()

_lib.rtu_init.argtypes = [C.c_int]
_lib.rtu_init.restype = C.c_int
_lib.rtu_device_name.argtypes = [C.c_char_p, C.c_int]
_lib.rtu_device_name.restype = C.c_int
_lib.rtu_blas_build.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    C.c_int, C.c_float, C.POINTER(C.c_double),
]
_lib.rtu_blas_build.restype = C.c_void_p
_lib.rtu_blas_free.argtypes = [C.c_void_p]
_lib.rtu_blas_free.restype = None
_lib.rtu_blas_bytes.argtypes = [C.c_void_p]
_lib.rtu_blas_bytes.restype = C.c_size_t
_lib.rtu_occlusion.argtypes = [
    C.c_void_p,
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    C.c_int,
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
]
_lib.rtu_occlusion.restype = C.c_int
_lib.rtu_horizon.argtypes = [
    C.c_void_p,
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    C.c_int,
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
]
_lib.rtu_horizon.restype = C.c_int
if hasattr(_lib, "rtu_mesh_build"):
    _lib.rtu_mesh_build.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        C.c_int,
        np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
        C.c_int,
        C.POINTER(C.c_double),
    ]
    _lib.rtu_mesh_build.restype = C.c_void_p

_DEFAULT_GPU = int(os.environ.get("AFI_RTU_GPU", "1"))
_init_done = False


def init(gpu_index: int = _DEFAULT_GPU) -> str:
    """Bring up the RTU context on L0 GPU `gpu_index`. Returns device name."""
    global _init_done
    rc = _lib.rtu_init(gpu_index)
    if rc != 0:
        raise RuntimeError(f"rtu_init({gpu_index}) failed: {rc}")
    _init_done = True
    buf = C.create_string_buffer(128)
    _lib.rtu_device_name(buf, 128)
    return buf.value.decode()


def _ensure_init() -> None:
    if not _init_done:
        init()


class _Blas:
    """Shared occlusion / closest-hit over an opaque RTAS handle."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        if getattr(self, "_handle", None):
            _lib.rtu_blas_free(self._handle)
            self._handle = None

    def _check_rays(self, rays, tmax=None):
        r = np.ascontiguousarray(rays, dtype=np.float32)
        if r.ndim == 2 and r.shape[1] == 6:
            r = r.reshape(-1)
        if r.ndim != 1 or r.size % 6:
            raise ValueError("rays must be (n, 6) [origin xyz | unit dir xyz]")
        n = r.size // 6
        if tmax is None:
            t = np.full(n, 1e30, dtype=np.float32)
        else:
            t = np.ascontiguousarray(tmax, dtype=np.float32)
            if t.size != n:
                raise ValueError("tmax must have n entries")
        return r, t, n

    def occlusion(self, rays, tmax=None) -> np.ndarray:
        """Boolean LOS occlusion per ray (segment capped at tmax)."""
        r, t, n = self._check_rays(rays, tmax)
        out = np.zeros(n, dtype=np.int32)
        rc = _lib.rtu_occlusion(self._handle, r, t, n, out)
        if rc != 0:
            raise RuntimeError(f"rtu_occlusion failed: {rc}")
        return out.astype(bool)

    def horizon(self, rays, tmax=None):
        """Closest-hit distance (m) and primitive id per ray; -1 on miss."""
        r, t, n = self._check_rays(rays, tmax)
        out_t = np.zeros(n, dtype=np.float32)
        out_p = np.zeros(n, dtype=np.int32)
        rc = _lib.rtu_horizon(self._handle, r, t, n, out_t, out_p)
        if rc != 0:
            raise RuntimeError(f"rtu_horizon failed: {rc}")
        return out_t, out_p


class TerrainBLAS(_Blas):
    """Opaque acceleration structure over a (G+1)x(G+1) heightfield."""

    def __init__(self, heights: np.ndarray, span: float):
        _ensure_init()
        h = np.ascontiguousarray(heights, dtype=np.float32)
        if h.ndim != 2 or h.shape[0] != h.shape[1]:
            raise ValueError("heights must be a square (G+1, G+1) array")
        build_ms = C.c_double(0.0)
        self._handle = _lib.rtu_blas_build(
            h.reshape(-1), int(h.shape[0] - 1), float(span), C.byref(build_ms))
        if not self._handle:
            raise RuntimeError("rtu_blas_build failed")
        self.build_ms = build_ms.value
        self.bytes = _lib.rtu_blas_bytes(self._handle)
        self.G = int(h.shape[0] - 1)
        self.span = float(span)


class MeshBLAS(_Blas):
    """Opaque acceleration structure over a triangle mesh (room-IR)."""

    def __init__(self, verts: np.ndarray, faces: np.ndarray):
        _ensure_init()
        if not hasattr(_lib, "rtu_mesh_build"):
            raise RuntimeError("libafi_rtu.so has no rtu_mesh_build — rebuild")
        v = np.ascontiguousarray(verts, dtype=np.float32)
        f = np.ascontiguousarray(faces, dtype=np.uint32)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError("verts must be (V, 3)")
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError("faces must be (T, 3)")
        if int(f.max(initial=0)) >= v.shape[0]:
            raise ValueError("face index out of range")
        build_ms = C.c_double(0.0)
        self._handle = _lib.rtu_mesh_build(
            v.reshape(-1), int(v.shape[0]), f.reshape(-1), int(f.shape[0]),
            C.byref(build_ms))
        if not self._handle:
            raise RuntimeError("rtu_mesh_build failed")
        self.build_ms = build_ms.value
        self.bytes = _lib.rtu_blas_bytes(self._handle)
        self.verts = v
        self.faces = f
        self.ntri = int(f.shape[0])

    def prim_normal(self, prim_id: int) -> np.ndarray:
        """Unit inward-ish geometric normal of triangle `prim_id`."""
        i0, i1, i2 = self.faces[int(prim_id)]
        e1 = self.verts[i1] - self.verts[i0]
        e2 = self.verts[i2] - self.verts[i0]
        n = np.cross(e1, e2)
        return n / max(float(np.linalg.norm(n)), 1e-12)


def _unit(dx, dy, dz):
    n = np.sqrt(dx * dx + dy * dy + dz * dz)
    return np.array([dx / n, dy / n, dz / n], dtype=np.float32)


def los(blas: TerrainBLAS, tx, rx) -> bool:
    """Single-segment LOS: tx/rx are (x, y, z) world meters. True = blocked."""
    tx = np.asarray(tx, dtype=np.float32)
    rx = np.asarray(rx, dtype=np.float32)
    d = _unit(*(rx - tx))
    rays = np.concatenate([tx, d])[None, :]
    dist = float(np.linalg.norm(rx - tx))
    return bool(blas.occlusion(rays, np.array([dist], dtype=np.float32))[0])


def horizon_fan(blas: TerrainBLAS, center, h_agl: float, n_az: int = 360,
                elev_deg: float = -2.0, tmax: float = 1e30):
    """Azimuth horizon sweep: first-hit distance per azimuth from a sensor at
    `center` (x, z) + h_agl above ground. Returns (azimuths_deg, t_m)."""
    cx, cz = float(center[0]), float(center[1])
    org = np.array([cx, h_agl, cz], dtype=np.float32)
    az = np.linspace(0.0, 360.0, n_az, endpoint=False)
    e = np.deg2rad(elev_deg)
    dirs = np.stack(
        [np.cos(np.deg2rad(az)), np.full(n_az, np.sin(e)),
         np.sin(np.deg2rad(az))], axis=1).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rays = np.concatenate(
        [np.tile(org, (n_az, 1)), dirs.astype(np.float32)], axis=1)
    t, _ = blas.horizon(rays, np.full(n_az, tmax, dtype=np.float32))
    return az, t
