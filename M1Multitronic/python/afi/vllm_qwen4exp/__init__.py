# SPDX-License-Identifier: Apache-2.0
"""vLLM plugin for Qwen4-Exp (Qwen3.8-Flash-Next).

This package provides the model_executor entry for Qwen3.8-Flash-Next:

  - Registers ``Qwen4ExpForCausalLM`` under the ``Qwen4ExpForConditionalGeneration``
    architecture name (the top-level config.json key) so vLLM's ModelRegistry
    can find it when loading ~/QwenFlash.
  - Registers ``Qwen4ExpConfig`` as the text-sub-config handler for
    ``qwen4_exp_text`` (the ``text_config.model_type`` key).

Wired via the ``afi.vllm_qwen4exp:register`` general plugin entry point in
pyproject.toml.  Idempotent: safe to call multiple times.

Architecture notes (BICAMERAL_FRAMEWORK.md §3 + tech report Eq. 30–34):
  - Linear attention layers (3 per block) → GatedDeltaNet (XPU kernel).
  - Full-attention layers → GQA + mRoPE; QSA indexer skipped (fabric
    ``qsa_indexer.py`` over captured hidden states, BICAMERAL §6.4).
  - MoE: EVERY decoder layer, 512 experts, top-10 + shared.  Routed
    tiles stay on NVMe; shared expert + gate load on GPU0.
  - N-gram PLE at config ple_layer_ids=[2] (weights on layers.1).
    128 shards skipped; host igram store is BICAMERAL §6.3.
  - Gated Residual Mix/Combine on a 4-wide residual; Hres dropped.
  - MTP head: stubbed (skip_prefixes mtp.).
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

__all__ = ["register"]


def register() -> None:
    """vllm.general_plugins entry point. Idempotent."""
    try:
        from vllm.model_executor.models.registry import ModelRegistry
    except ImportError:
        print(
            "[afi-qwen4exp] vllm not installed — skipping registration",
            file=sys.stderr,
        )
        return

    from .config import Qwen4ExpConfig
    from .top_config import Qwen4ExpTopConfig

    try:
        from transformers import AutoConfig
        AutoConfig.register("qwen4_exp", Qwen4ExpTopConfig, exist_ok=True)
        AutoConfig.register("qwen4_exp_text", Qwen4ExpConfig, exist_ok=True)
    except Exception as exc:
        logger.warning("[afi-qwen4exp] AutoConfig.register failed: %s", exc)

    # Register the model class under the top-level config architecture name.
    try:
        ModelRegistry.register_model(
            "Qwen4ExpForConditionalGeneration",
            "afi.vllm_qwen4exp.model:Qwen4ExpForCausalLM",
        )
        logger.info(
            "[afi-qwen4exp] registered Qwen4ExpForConditionalGeneration "
            "→ Qwen4ExpForCausalLM"
        )
    except Exception as exc:
        logger.warning(
            "[afi-qwen4exp] failed to register Qwen4ExpForConditionalGeneration: %s",
            exc,
        )

    # Register both config types so transformers can load ~/QwenFlash/config.json.
    #   top-level  model_type = "qwen4_exp"           -> Qwen4ExpTopConfig
    #   text sub-config model_type = "qwen4_exp_text" -> Qwen4ExpConfig
    # LazyConfigDict resolves string keys via getattr(configs_module, name).
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        import vllm.transformers_utils.configs as _cfgs_pkg
        # Inject so LazyConfigDict getattr resolves correctly.
        setattr(_cfgs_pkg, "Qwen4ExpConfig", Qwen4ExpConfig)
        setattr(_cfgs_pkg, "Qwen4ExpTopConfig", Qwen4ExpTopConfig)
        _CONFIG_REGISTRY["qwen4_exp"] = Qwen4ExpTopConfig
        _CONFIG_REGISTRY["qwen4_exp_text"] = Qwen4ExpConfig
        # Also with AutoConfig so transformers.from_pretrained works.
        try:
            from transformers import AutoConfig
            AutoConfig.register("qwen4_exp", Qwen4ExpTopConfig, exist_ok=True)
            AutoConfig.register("qwen4_exp_text", Qwen4ExpConfig, exist_ok=True)
        except Exception:
            pass
        logger.info(
            "[afi-qwen4exp] registered Qwen4ExpTopConfig (qwen4_exp) "
            "and Qwen4ExpConfig (qwen4_exp_text)"
        )
    except Exception as exc:
        logger.warning("[afi-qwen4exp] failed to register config: %s", exc)

    _patch_sparse_safetensors()
    print(
        "[afi-qwen4exp] Qwen3.8-Flash-Next model registered "
        "(GR Mix/Combine, MoE-every-layer, PLE@layer2, QSA→fabric, MTP stub)",
        file=sys.stderr,
    )


_SPARSE_PATCHED = False


def _patch_sparse_safetensors() -> None:
    """Skip n-gram / expert / visual / MTP keys *before* get_tensor.

    vLLM's iterator loads each tensor, then our model.load_weights
    drops it.  Expert tiles are 3.4 GB; n-gram shards 800 MB.  Filter
    at keys() so FIRSTRUN swap does not spike host RAM.

    LANDMINE (2026-08-27): this monkeypatch is process-global, so it
    also sees the duty 27B GPTQ checkpoint — whose MTP draft weights
    are named ``mtp.*`` exactly like Qwen-FN's stubbed ones.  Filtering
    them hid all 15 draft tensors and MTP4 acceptance collapsed to
    0.0% (decode 48-52 -> ~14 tok/s).  The filter must only apply to
    Qwen-FN checkpoints.  Per-file detection is NOT enough (Qwen-FN's
    own mtp.* shards carry no signature key), so verdicts are made
    per *model directory* via the safetensors index: any
    ``.ngram_embedding.shard_`` key in the index => Qwen-FN => filter;
    anything else (27B duty included) passes through unfiltered.
    """
    global _SPARSE_PATCHED
    if _SPARSE_PATCHED:
        return
    try:
        import vllm.model_executor.model_loader.weight_utils as wu
    except ImportError:
        return

    import json
    from pathlib import Path

    from .layout import should_skip_weight as _skip

    orig_open = wu.safe_open
    _dir_verdict: dict[str, bool] = {}

    def _is_qwenfn_dir(filename) -> bool:
        d = str(Path(str(filename)).parent)
        if d not in _dir_verdict:
            verdict = False
            idx = Path(d) / "model.safetensors.index.json"
            try:
                if idx.exists():
                    with open(idx) as f:
                        wm = json.load(f).get("weight_map", {})
                    verdict = any(".ngram_embedding.shard_" in k for k in wm)
            except Exception:  # noqa: BLE001
                verdict = False
            _dir_verdict[d] = verdict
        return _dir_verdict[d]

    class _FilteredOpen:
        def __init__(self, inner_cm, filename):
            self._cm = inner_cm
            self._filter = _is_qwenfn_dir(filename)

        def __enter__(self):
            inner = self._cm.__enter__()
            do_filter = self._filter

            class _Handle:
                def keys(handle_self):
                    ks = inner.keys()
                    if not do_filter:
                        return ks
                    return [k for k in ks if not _skip(k)]

                def get_tensor(handle_self, name):
                    return inner.get_tensor(name)

                def metadata(handle_self):
                    return inner.metadata()

                def __getattr__(handle_self, item):
                    return getattr(inner, item)

            return _Handle()

        def __exit__(self, *exc):
            return self._cm.__exit__(*exc)

    def filtered_open(*args, **kwargs):
        filename = args[0] if args else kwargs.get("filename")
        return _FilteredOpen(orig_open(*args, **kwargs), filename)

    wu.safe_open = filtered_open
    _SPARSE_PATCHED = True
    logger.info("[afi-qwen4exp] safetensors iterator filters cold shards "
                "(Qwen-FN checkpoint dirs only)")
