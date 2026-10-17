# Arc B70 — SageAttention + FWHT kernel collection

Collected 2026-10-17 from `192.168.1.107` (user `bbear`, machine "generator", CachyOS)
via ssh. Paths below are relative to `/home/bbear` on that machine; the local tree
preserves the same layout.

## Custom (hand-written for XPU / Arc B70) — verified NOT stock

Stock llama.cpp contains **no** `fwht.*` files at all; every FWHT kernel here is a
local addition, confirmed via `git ls-files` + commit history:

| File group | Source | Evidence of custom |
|---|---|---|
| `llama.cpp/ggml/src/ggml-cuda/fwht.{cu,cuh}` | ~/llama.cpp | commits `cuda: FWHT kernels for block widths above 2048`, `cuda: block-per-row FWHT for widths 512 and up` |
| `llama.cpp/ggml/src/ggml-sycl/fwht.{cpp,hpp}` | ~/llama.cpp | SYCL sub-group butterfly kernels; commit `sycl: serve block_size 1024 in the Hadamard fast path` |
| `llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/fwht.comp` | ~/llama.cpp | subgroup-shuffle Vulkan compute, custom |
| `llama.cpp-flashnext/ggml/src/ggml-sycl/fwht_blocked.{cpp,hpp}` | ~/llama.cpp-flashnext | commit `Add blockwise signed Hadamard transform for SYCL (ternary weights)` — register-footprint-bounded blocked variant |
| `llama.cpp-flashnext/tests/test-fwht-blocked/test-fwht-blocked.cpp` | ~/llama.cpp-flashnext | host test for the blocked kernel |
| `M1Multitronic/sycl/**` | ~/M1Multitronic | `namespace afi`, `sycl::ext::intel::esimd` GRF-resident kernels (`fwht_esimd`), qkint8 (INT8 QK·V), dequant, torch op bridge (`sycl/torch/fwht_op.cpp`) |
| `M1Multitronic/python/sageattention/{__init__,core,int8}.py` | ~/M1Multitronic | "XPU SageAttention drop-in" — SDPA-backed, no CUDA; API-compatible with upstream sageattention (HND/NHD layouts) |
| `M1Multitronic/python/afi/**` (incl. `afi_xpu.so`) | ~/M1Multitronic | AFI support package: `fwht.py` host oracle, `xpu.py`, `vllm_plugin.py`, qkint8 tuner; `afi_xpu.so` is the compiled XPU ext |
| `M1Multitronic/python/tests/test_{fwht,fwht_xpu,sageattention_xpu}.py` | ~/M1Multitronic | XPU validation tests |
| `M1Multitronic/scripts/{build-torch-fwht.sh,bench-comfy-fwht.py,validate_qk_fwht.py,latent-fwht-worker.py}` | ~/M1Multitronic | build + validation glue |
| `M1Multitronic/CMakeLists.txt` | ~/M1Multitronic | SYCL/esimd build glue for the sycl tree |
| `M1Multitronic/docs/fwht-bridge-architecture.md` | ~/M1Multitronic | design doc for the FWHT bridge |
| `.config/systemd/user/vllm.service.d/afi-fwht.conf` | ~/vllm service | `AFI_FWHT_QK=1`, `AFI_FWHT_IMPL=auto`, etc. — how the vLLM service enables FWHT on B70 |
| `intel-arc-pro-b70-inference-cookbook/patches/vllm-xpu-kernels/*` | cookbook | vLLM XPU kernel compat patches (xe2 grouped-GEMM atomics, muse paged decode) |
| `_integration-patches/*.patch` | git format-patch | the 5 kernel commits incl. their wiring into `ggml-cuda.cu` / `ggml-sycl.cpp` / `vulkan-shaders-gen.cpp` |

## Excluded — stock / reference / false positives

- `comfy-kitchen-xpu/comfy_kitchen/sage_attention.py` — **stock upstream NVIDIA**
  (Apache-2.0, NVIDIA copyright header); the XPU wheel keeps the API importable but
  disables the CUDA backend. Not custom XPU code.
- `IEMPluginSuite/resources/WalshHadamard/{fwht.h,mx_fwht.cpp,mx_ifwht.cpp}` —
  third-party academic reference (Jonathan Hadida, dtc.ox.ac.uk). Reference only.
- `flashnext-harness/fabric/sage.py` — SageMath hook, NOT SageAttention.
- `flashnext-harness/kernels/m1multitronic-sycl/sycl/**` — byte-identical mirror of
  `M1Multitronic/sycl` (`diff -rq` clean); duplicate, skipped.
- `~/vllm-cross-model-kv`, `~/vllm-lora`, `~/runtime/.venv-vllm` — searched for
  sage/fwht: no relevant hits (only "usage"/"sagemaker" substring noise). No
  sageattention package is installed in the vLLM venv.
- `~/llama8016` — no fwht/sage files.
- Metal-only FWHT commits not collected as files (the metal-vulkan commit patch is
  included since it also carries the vulkan kernel).

## Notes

- llama.cpp FWHT = Hadamard rotation for quantization paths (Q4/Q5/K tensors);
  the SageAttention INT8-QK path lives in M1Multitronic (`qkint8` + `fwht_esimd`
  + the `sageattention` drop-in), enabled in vLLM via the systemd `afi-fwht.conf`.
- `M1Multitronic/python/afi/` includes some modules unrelated to FWHT/SAGE
  (rtu, room_ir, mdct, bicameral, ...); kept whole so package imports stay intact.
