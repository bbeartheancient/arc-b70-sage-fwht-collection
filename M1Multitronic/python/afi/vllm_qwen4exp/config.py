# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp config (Qwen3.8-Flash-Next).

Modeled on Qwen3NextConfig, adds the N-gram embedding fields and Gated
Residual knobs that Qwen3-Next does not have. Loaded by the auto-config
machinery when the config.json ``model_type`` is ``qwen4_exp`` (top) or
``qwen4_exp_text`` (text sub-config).

Field source: ~/QwenFlash/config.json + the Qwen3.8-Flash-Next model
card.  All defaults match Qwen3.8-Flash-Next (hidden 2560, 48 layers,
512 experts, ngram layer 2, etc.).
"""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .layout import default_layer_types

logger = logging.get_logger(__name__)


class Qwen4ExpConfig(PretrainedConfig):
    """Qwen3.8-Flash-Next (Qwen4-Exp) text config."""

    model_type = "qwen4_exp_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.*.gate_proj": "colwise",
        "layers.*.mlp.experts.*.up_proj": "colwise",
        "layers.*.mlp.experts.*.down_proj": "rowwise",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=2560,
        intermediate_size=0,
        num_hidden_layers=48,
        num_attention_heads=24,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        # Linear attention (Gated DeltaNet) — used by Qwen3-Next analog.
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        mamba_ssm_dtype="float32",
        # MoE.
        # Qwen3-Next convention: decoder_sparse_step=4 means layer N is MoE iff
        # (N+1) % 4 == 0.  Qwen-FN's config.json sets this to 4 and its
        # layer_types[3,7,11,...,47] are all 'full_attention' (the MoE layers).
        # Qwen-FN ships MoE experts on EVERY decoder layer (and MTP).
        # decoder_sparse_step=1 makes (layer_idx+1)%step==0 always true.
        decoder_sparse_step=1,
        moe_intermediate_size=640,
        # expert_intermediate_size is the alias Qwen3NextSparseMoeBlock looks
        # up; vLLM's `Qwen3NextSparseMoeBlock.__init__` reads
        # `config.moe_intermediate_size` for the experts.  We keep the alias
        # so any code path that reads either field is happy.
        expert_intermediate_size=640,
        shared_expert_intermediate_size=640,
        num_experts_per_tok=10,
        num_experts=512,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        output_gate_type="sigmoid",
        mlp_only_layers=None,
        layer_types=None,
        full_attention_interval=4,
        # QSA indexer (BICAMERAL §6.4).
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        # N-gram embedding (BICAMERAL §6.3). 20M vocab trigrams at layer 2.
        ngram_vocab_size_base=20000000,
        ngram_size=3,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=128,
        ple_layer_ids=None,
        ple_embed_dim=2560,
        ple_conv_kernel_size=4,
        heads_per_ngram=8,
        # Gated Residual (4 branches, rank-320 bottleneck).
        hc_count=4,
        hc_lowrank=320,
        # MTP head (1 layer, hybrid full-attn).
        mtp=None,
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        **kwargs,
    ):
        if mlp_only_layers is None:
            mlp_only_layers = []
        # config.json is 1-indexed: ple_layer_ids=[2] → weights at layers.1.ple.
        if ple_layer_ids is None:
            ple_layer_ids = [2]

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        # RoPE — Qwen3.8-Flash-Next ships YaRN / mrope / partial-roPE.
        rope_scaling = kwargs.pop("rope_scaling", None)
        rope_parameters = rope_scaling or rope_parameters or {
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        }
        rope_theta = kwargs.pop("rope_theta", 10000000.0)
        if "rope_theta" not in rope_parameters:
            rope_parameters["rope_theta"] = rope_theta
        partial_rotary_factor = kwargs.pop("partial_rotary_factor", 0.25)
        if "partial_rotary_factor" not in rope_parameters:
            rope_parameters["partial_rotary_factor"] = partial_rotary_factor
        self.rope_parameters = rope_parameters
        self.partial_rotary_factor = partial_rotary_factor

        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        # q_proj is 2× (query + sigmoid output gate); matches checkpoint
        # self_attn.q_proj [12288, 2560] = 24 heads × 256 × 2.
        self.attn_output_gate = kwargs.pop("attn_output_gate", True)

        # Layer types: 3 linear_attention + 1 full_attention, repeating
        # (full at 0-indexed 3/7/11/…/47).  Do NOT put full on i%4==0.
        self.full_attention_interval = full_attention_interval
        if layer_types is None:
            self.layer_types = default_layer_types(
                num_hidden_layers, full_attention_interval
            )
        else:
            self.layer_types = list(layer_types)
        self._validate_layer_types()

        # Linear attention (GatedDeltaNet).
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.mamba_ssm_dtype = mamba_ssm_dtype

        # MoE.
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.expert_intermediate_size = expert_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.output_gate_type = output_gate_type
        self.mlp_only_layers = mlp_only_layers

        # QSA indexer.
        self.indexer_n_heads = indexer_n_heads
        self.indexer_kv_heads = indexer_kv_heads
        self.indexer_head_dim = indexer_head_dim
        self.indexer_budget = indexer_budget
        self.indexer_compress_ratio = indexer_compress_ratio

        # N-gram embedding.
        self.ngram_vocab_size_base = ngram_vocab_size_base
        self.ngram_size = ngram_size
        self.make_ngram_vocab_size_divisible_by = make_ngram_vocab_size_divisible_by
        self.split_ngram_parts = split_ngram_parts
        self.ple_layer_ids = list(ple_layer_ids)
        self.ple_embed_dim = ple_embed_dim
        self.ple_conv_kernel_size = ple_conv_kernel_size
        self.heads_per_ngram = heads_per_ngram

        # Gated Residual.
        self.hc_count = hc_count
        self.hc_lowrank = hc_lowrank

        # MTP.
        self.mtp = mtp or {
            "hybrid": True,
            "layer_types": ["full_attention"],
            "mtp_use_hidden_state_from_layer": None,
            "num_hidden_layers": 1,
            "rope_theta": 10000000,
        }
        self.mtp_num_hidden_layers = mtp_num_hidden_layers
        self.mtp_use_dedicated_embeddings = mtp_use_dedicated_embeddings

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def _validate_layer_types(self) -> None:
        valid = {"linear_attention", "full_attention"}
        for i, t in enumerate(self.layer_types):
            if t not in valid:
                raise ValueError(
                    f"layer_types[{i}]={t!r} is not one of {valid}"
                )


__all__ = ["Qwen4ExpConfig"]
