# xpu-flash-attn

A pure-Python FlashAttention-compatible interface for Intel XPU GPUs.

It is **not** a new kernel implementation. Instead it maps the public
`flash_attn` API to PyTorch's native `scaled_dot_product_attention`, which has
an XPU backend in `torch==2.12.1+xpu`.  This avoids building the CUDA-only
`flash-attn` wheel and works on Intel Arc A770, PVC, BMG, etc.

## Install

```bash
source /home/bbear/Documents/OlympusServer/activate-env.sh
cd /home/bbear/Documents/xpu-flash-attn-shim
pip install --no-build-isolation .
```

## Use

The package installs both the explicit `xpu_flash_attn` module and a
`flash_attn` compatibility shim that shadows the official (CUDA-only) package,
so existing code keeps working unchanged:

```python
# Either import style works
import flash_attn
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.flash_attn_interface import flash_attn_func
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.modules.mha import FlashSelfAttention, FlashCrossAttention
```

## Supported APIs

- `flash_attn_func` (and `flash_attn_func_v2/v3/v4` aliases)
- `flash_attn_varlen_func` (and `flash_attn_varlen_func_v2/v3/v4` aliases)
- `flash_attn_qkvpacked_func`
- `flash_attn_kvpacked_func`
- `flash_attn.bert_padding.pad_input` / `unpad_input`
- `flash_attn.modules.mha.FlashSelfAttention` / `FlashCrossAttention`
- `flash_attn.cute` re-export

Unsupported symbols (e.g. `flash_attn_with_kvcache`, `sparse_attn_with_sla`)
raise `NotImplementedError` at call time rather than at import time.

## Caveats

- GQA/MQA is handled by repeating K/V heads.
- Alibi, softcap, sliding-window masks, paged KV block tables, and
  `return_attn_probs` fall back to an explicit PyTorch reference.
- On Intel Arc A770, PyTorch XPU SDPA does **not** use the flash-attention
  kernel (only PVC/BMG/LNL do); it uses an optimized oneDNN/math path.  It is
  still ~10x faster than naive attention for long sequences.

## Test

```bash
python tests/test_xpu_flash_attn.py
```
