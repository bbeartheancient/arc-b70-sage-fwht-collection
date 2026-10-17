#!/usr/bin/env bash
# Build torch.ops.afi.fwht against the vLLM venv + intel/llvm v7.
# Do not source this into Comfy. Do not put llvm7 on Comfy's LD_LIBRARY_PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLVM="${AFI_DPCPP_ROOT:-$HOME/intel-llvm-7.0.0}"
VENV="${VLLM_VENV:-$HOME/Documents/OlympusServer/.venv-vllm}"
PY="${AFI_PYTHON:-$VENV/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "missing vLLM python at $PY" >&2
    exit 1
fi
if [[ ! -x "$LLVM/bin/clang++" ]]; then
    echo "intel/llvm missing at $LLVM" >&2
    exit 1
fi
SYCL_RPATH="$VENV/lib"
if [[ ! -e "$SYCL_RPATH/libsycl.so.9" ]]; then
    SYCL_RPATH="$(dirname "$PY")/../lib"
fi
export AFI_DPCPP_ROOT="$LLVM"
WITH="$ROOT/scripts/with-intel-llvm.sh"
cd "$ROOT"
bash "$WITH" cmake --preset llvm7 \
    -DAFI_TORCH=ON \
    -DAFI_PYTHON="$PY" \
    -DAFI_SYCL_RPATH="$(cd "$SYCL_RPATH" && pwd)"
bash "$WITH" cmake --build "$ROOT/build-llvm7" --target afi_xpu afi-fwht-check
cp -f "$ROOT/build-llvm7/afi_xpu.so" "$ROOT/python/afi/afi_xpu.so"
echo "wrote $ROOT/python/afi/afi_xpu.so"
"$PY" -c "import afi.xpu as x; x.load(); print('loaded', x.so_path())"
