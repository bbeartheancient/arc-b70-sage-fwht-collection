# SPDX-License-Identifier: Apache-2.0
"""Top-level Qwen4-Exp config (Qwen3.8-Flash-Next, multimodal wrapper).

Qwen3.8-Flash-Next's config.json has ``model_type = "qwen4_exp"`` at the
top level and ``text_config.model_type = "qwen4_exp_text"``. The
top-level config owns the vision tower (out_hidden_size=2560, depth=27,
patch_size=16) and the text sub-config, just like Qwen3VLForConditionalGeneration
wraps Qwen3_5TextConfig.

We only need the top-level class to (a) satisfy transformers' auto-mapping
so ``AutoConfig.from_pretrained`` returns a config object, and (b) hand
the text sub-config to vLLM via ``hf_text_config``. The vision path is
not exercised on the B70 yet; the duty path is text-only.
"""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .config import Qwen4ExpConfig

logger = logging.get_logger(__name__)


class Qwen4ExpTopConfig(PretrainedConfig):
    """Top-level wrapper for Qwen3.8-Flash-Next.

    Holds a vision_config block (informational, not used by vLLM) and a
    text_config block (the active path).
    """

    model_type = "qwen4_exp"
    sub_configs = {"text_config": Qwen4ExpConfig, "vision_config": PretrainedConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id: int = 248056,
        video_token_id: int = 248057,
        vision_start_token_id: int = 248053,
        vision_end_token_id: int = 248054,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if text_config is None:
            text_config = Qwen4ExpConfig()
        elif isinstance(text_config, dict):
            text_config = Qwen4ExpConfig(**text_config)
        if vision_config is None:
            vision_config = PretrainedConfig()
        elif isinstance(vision_config, dict):
            vision_config = PretrainedConfig(**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_text_config(self, decoder=None) -> PretrainedConfig:
        """HF hook: return the active text sub-config."""
        return self.text_config

    def __getattr__(self, key: str):
        """Proxy unknown attributes to the text sub-config.

        Qwen3NextDecoderLayer / Qwen3NextAttention / QwenGatedDeltaNetAttention
        all access `vllm_config.model_config.hf_config.<field>` and expect
        num_experts / decoder_sparse_step / linear_conv_kernel_dim / etc. to
        live on the same object that holds hidden_size.  We put those on the
        text sub-config (Qwen4ExpConfig), so proxy through transparently.
        """
        # Avoid recursion on internals (PretrainedConfig stores a lot of
        # sub-objects in __dict__; only proxy once _init_ has run).
        if key.startswith("_") or key == "text_config" or key == "vision_config":
            raise AttributeError(key)
        text_config = self.__dict__.get("text_config")
        if text_config is not None and hasattr(text_config, key):
            return getattr(text_config, key)
        raise AttributeError(
            f"'Qwen4ExpTopConfig' object (and its text_config) has no attribute {key!r}"
        )


__all__ = ["Qwen4ExpTopConfig"]
