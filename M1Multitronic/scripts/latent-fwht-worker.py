#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""JSON-line FWHT service for the looping transformer (fabric/latent.py).

The fabric venv is py3.14 without torch; spectra for /v1/hidden_states
captures are computed here under the vLLM venv (torch 2.13+xpu +
afi_xpu.so). Protocol, one request per stdin line:

    {"b64": "<base64 fp16 LE [L,H]>", "shape": [L, H]}

one response line per request:

    {"ok": true,  "backend": "xpu", "fp": {...}}   # afi/spectral.py shape
    {"ok": false, "error": "..."}

Hidden rows are viewed as per-token blocks of 256 (Qwen-FN sparse-
attn head_dim; override with FABRIC_FWHT_BLOCK=128 for Qwen 3.8 27B)
and rotated with torch.ops.afi.fwht (impl=auto -> ESIMD/XMX on B70).
The transform is ORTHONORMAL: Parseval holds, so fingerprint energies
equal token-domain squared norms.

Ops:
  {"op": "spectra", "b64": ..., "shape": [L, H]}          (default)
      -> {"ok": true, "backend": ..., "fp": {...}}
  {"op": "compact", "b64": ..., "shape": [L, H], "rate": 0.5}
      dct_topk compaction along the token axis + torch.save payload
      -> {"ok": true, "b64": "<torch.save'd fp16 [L,H] CPU tensor>",
          "fp": {...}, "effective_rate": ...}
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

# Qwen-FN sparse/full-attention head_dim = 256; hidden rows are
# viewed as per-token blocks of this width for the per-block H4
# fingerprint.  Qwen 3.8 27B used 128 — that's the OLD default; if
# the worker is launched against the 27B checkpoint, set
# FABRIC_FWHT_BLOCK=128 in the calling environment.
import os
BLOCK = int(os.environ.get("FABRIC_FWHT_BLOCK", "256"))


def _compact(arr, rate: float, block: int):
    """dct_topk-compact [L, H] along axis 0; return (recon fp16 numpy,
    effective rate, mean per-row cosine vs input). Uses the
    whole-sequence orthonormal DCT-II — the Phase 3 gate winner
    (mdct tracks just under at every rate)."""
    import numpy as np

    from afi.mdct import dct2, idct2, keep_topk

    coeffs = dct2(arr.astype(np.float64), axis=0)
    kept = keep_topk(coeffs.T, rate).T
    recon = idct2(kept, axis=0)
    eff = float(np.count_nonzero(kept) / max(kept.size, 1))
    a = arr.astype(np.float64)
    r = recon.astype(np.float64)
    num = (a * r).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(r, axis=1)
    ok = den > 1e-30
    cos = float(np.mean(num[ok] / den[ok])) if ok.any() else 0.0
    return recon.astype(np.float16), eff, round(cos, 4)


def main() -> int:
    import numpy as np

    from afi.fwht import fwht as host_fwht

    import torch

    from afi import xpu
    from afi.spectral import fingerprint

    use_xpu = torch.xpu.is_available()
    backend = "xpu" if use_xpu else "cpu"

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op", "spectra")
            L, H = (int(v) for v in req["shape"])
            raw = base64.b64decode(req["b64"])
            if len(raw) != L * H * 2:
                raise ValueError(
                    f"payload {len(raw)}B != L*H*2 = {L * H * 2}")
            arr = np.frombuffer(raw, dtype="<f2").reshape(L, H)

            resp_extra: dict = {}
            if op == "compact":
                recon, eff, rcos = _compact(arr,
                                            float(req.get("rate", 0.5)),
                                            BLOCK)
                buf = io.BytesIO()
                torch.save(torch.from_numpy(
                    np.ascontiguousarray(recon)), buf)
                resp_extra = {
                    "b64": base64.b64encode(buf.getvalue()).decode("ascii"),
                    "effective_rate": round(eff, 4),
                    "row_cos": rcos,
                    "rows": int(recon.shape[0]),
                }
                stats_arr = recon
            else:
                stats_arr = arr

            t = torch.from_numpy(
                np.ascontiguousarray(stats_arr.astype(np.float16)))
            nb = -(-H // BLOCK)
            if nb * BLOCK != H:
                t = torch.nn.functional.pad(t, (0, nb * BLOCK - H))
            rows = t.view(L * nb, BLOCK)
            if use_xpu:
                coeffs = rows.to("xpu")
                coeffs = xpu.fwht(coeffs).float().cpu().numpy()
            else:
                coeffs = host_fwht(rows.numpy(), normalize=True)
                coeffs = coeffs.astype(np.float32)
            resp = {"ok": True, "backend": backend,
                    "fp": fingerprint(coeffs), **resp_extra}
        except Exception as e:  # noqa: BLE001
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
