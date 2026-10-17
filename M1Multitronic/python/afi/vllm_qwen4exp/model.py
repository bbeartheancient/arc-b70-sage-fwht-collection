# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp model for vLLM (Qwen3.8-Flash-Next).

This module is the vLLM model_executor entry for the Qwen3.8-Flash-Next
architecture (6B active params + 51B n-gram store + 4B MTP, 180B total).

Architecture decisions (per BICAMERAL_FRAMEWORK.md §3 + Qwen-FN model card):

  - 48 layers × mixed types: 3×linear_attention + full_attention per
    block (full_attention_interval=4).  linear_attention layers use the
    GatedDeltaNet (QwenGatedDeltaNetAttention, XPU kernel via
    vllm-xpu-kernels libgdn_attn_kernels_xe_2.so).  full_attention
    layers use standard GQA + mRoPE (partial_rotary_factor=0.25,
    mrope_section=[11,11,10]).
  - MoE: every decoder layer is SparseMoeBlock (512 experts, top-10 +
    shared).  Routed tiles stay on NVMe; BicameralShardedMoE runs the
    gate + shared expert on GPU0.
  - Gated Residual: Mix/Combine on a 4-wide residual (tech report
    Eq. 30–34).  Hres is dropped.  Final hyper_connection_mixer Mixes
    4-wide → d; there is no model.norm.
  - N-gram PLE at config ple_layer_ids=[2] (weights on layers.1).
    Shards skipped; small PLE tensors load; forward identity until
    BICAMERAL_NGRAM_MODEL=1.
  - QSA indexer (budget 2048, 4Q/1KV, head_dim 128, r=4): stubbed to
    identity.  The QSA compression runs in fabric/latent.py over the
    captured hidden states (python/afi/qsa_indexer.py).
  - MTP head: 1 layer, hybrid full-attention, 4B params.  Stubbed.
  - mRoPE / YaRN: handled by Qwen3NextAttention's rope_parameters path
    (partial_rotary_factor=0.25, mrope_section=[11,11,10]).

Dependency on Qwen3Next (vllm/model_executor/models/qwen3_next.py):
  - Qwen3NextDecoderLayer: layer types + attention (linear / full)
  - Qwen3NextAttention: full_attention with mRoPE
  - Qwen3NextRMSNorm / Qwen3NextMLP: norm + dense MLP
  - (Qwen3NextSparseMoeBlock: REPLACED by BicameralShardedMoE)

BICAMERAL_FRAMEWORK.md §6.3 landmine: N-gram is a *store*, not a
compute kernel.  The framework's igram.py (host Python) owns the I64
dequant path; do NOT write a SYCL I64 dequant kernel.
"""

from __future__ import annotations

import logging
import os
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsMRoPE,
    SupportsPP,
)
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from afi.bicameral import BicameralShardedMoE, BicameralNgramEmbedding

from .config import Qwen4ExpConfig
from .layout import classify_weight, should_skip_weight, strip_language_model_prefix
from .norm import Qwen4ExpGroupedRMSNorm

logger = init_logger(__name__)


def _support_torch_compile_safe(cls):
    """Decorator that applies @support_torch_compile if available, else no-op."""
    try:
        from vllm.compilation.decorators import support_torch_compile
        return support_torch_compile(cls)
    except Exception:
        return cls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_on(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


_NGRAM_LOADER: object | None = None
_NGRAM_LOADER_LOCK_PATH: str | None = None


def _get_ngram_loader():
    """Singleton factory for the LazyNgramLoader.

    Reads BICAMERAL_NGRAM_MODEL_DIR (default: the vLLM model dir
    resolved from VLLM_MODEL_NAME/QWENFN_MODEL_DIR).  If unset or the
    directory has no safetensors index, returns None and the PLE forward
    stays identity.
    """
    global _NGRAM_LOADER, _NGRAM_LOADER_LOCK_PATH
    if _NGRAM_LOADER is not None:
        return _NGRAM_LOADER
    from afi.bicameral.lazy_ngram import LazyNgramLoader
    model_dir = os.environ.get(
        "BICAMERAL_NGRAM_MODEL_DIR"
    ) or os.environ.get("QWENFN_MODEL_DIR") or ""
    if not model_dir or not os.path.isdir(model_dir):
        return None
    if _NGRAM_LOADER_LOCK_PATH == model_dir:
        return _NGRAM_LOADER
    max_shards = int(os.environ.get("BICAMERAL_NGRAM_RESIDENT_SHARDS", "10"))
    try:
        _NGRAM_LOADER = LazyNgramLoader(
            model_dir, max_resident_shards=max_shards
        )
        _NGRAM_LOADER_LOCK_PATH = model_dir
    except Exception as exc:  # noqa: BLE001
        logger.warning("LazyNgramLoader init failed: %s", exc)
        _NGRAM_LOADER = None
    return _NGRAM_LOADER


# ---------------------------------------------------------------------------
# Gated Residual — tech report Eq. 30–34
# ---------------------------------------------------------------------------
# R ∈ R^{nr × d}, nr=4, d=2560, rank r=d/8=320.
#   (30) R̃_i = RMSNorm(R_i; γ_i)          group RMS over each branch
#   (31) G   = unvec σ(Wu SiLU((1/nr) Wd vec(R̃)))   ∈ R^{nr×d}
#   (32) x   = (1/nr) Σ_i G_i ⊙ R̃_i
#   (33) s   = 2 σ((1/nr) Ww vec(R̃))                 ∈ R^{nr}
#   (34) R'_i = R_i + s_i y                            Hres dropped
# Wd [320, 10240], Wu [10240, 320], Ww [4, 10240], γ [10240].


class Qwen4ExpGatedResidual(nn.Module):
    """Gated Residual Mix / Combine (tech report Eq. 30–34).

    Residual state is 4-wide ``[T, nr*d]``.  ``Hres`` is dropped.
    The final ``hyper_connection_mixer`` uses ``use_combine=False``
    (Mix only; no ``block_inject_weight`` on that checkpoint module).
    """

    def __init__(
        self,
        config: Qwen4ExpConfig,
        prefix: str = "",
        use_combine: bool = True,
    ) -> None:
        super().__init__()
        self.nr = config.hc_count
        self.d = config.hidden_size
        hc = self.nr * self.d
        self.hc_norm = Qwen4ExpGroupedRMSNorm(
            hc, group_size=self.d, eps=config.rms_norm_eps
        )
        self.input_mix_weight_down = ReplicatedLinear(
            hc, config.hc_lowrank, bias=False,
            quant_config=None, prefix=f"{prefix}.input_mix_weight_down",
        )
        self.input_mix_weight_up = ReplicatedLinear(
            config.hc_lowrank, hc, bias=False,
            quant_config=None, prefix=f"{prefix}.input_mix_weight_up",
        )
        if use_combine:
            self.block_inject_weight = ReplicatedLinear(
                hc, self.nr, bias=False,
                quant_config=None, prefix=f"{prefix}.block_inject_weight",
            )
        else:
            self.block_inject_weight = None

    def mix(
        self, hyper_input: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hyper_normed = self.hc_norm(hyper_input)
        z, _ = self.input_mix_weight_down(hyper_normed)
        z = F.silu(z / self.nr)
        g, _ = self.input_mix_weight_up(z)
        g = torch.sigmoid(g).unflatten(-1, (self.nr, self.d))
        mixed = (g * hyper_normed.unflatten(-1, (self.nr, self.d))).mean(dim=-2)
        if self.block_inject_weight is None:
            return mixed
        inj, _ = self.block_inject_weight(hyper_normed)
        inj = 2 * torch.sigmoid(inj / self.nr)
        return mixed, hyper_input, inj

    @staticmethod
    def combine(
        hyper_input: torch.Tensor,
        y: torch.Tensor,
        injection_weights: torch.Tensor,
    ) -> torch.Tensor:
        injection = y.unsqueeze(-2) * injection_weights.unsqueeze(-1)
        return hyper_input + injection.flatten(-2)



# ---------------------------------------------------------------------------
# MTP head (stub — future: speculative decoding)
# ---------------------------------------------------------------------------

class Qwen4ExpMTPHead(nn.Module):
    """MTP head: 1 layer, hybrid full-attention, 4B params.

    Stub: no-op.  MTP speculative decoding will be wired when the
    vLLM speculative decoding path for hybrid models stabilizes.
    """

    def __init__(self, config: Qwen4ExpConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return hidden_states


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class Qwen4ExpDecoderLayer(nn.Module):
    """One decoder block: Mix → (GDN | GQA) → Combine → Mix → MoE → Combine.

    Every layer is a SparseMoeBlock (checkpoint has experts on all 48).
    Routed experts stay on NVMe; BicameralShardedMoE runs gate + shared
    expert.  PLE (n-gram) attaches only on the 1-indexed layer in
    ``ple_layer_ids`` and injects into the 4-wide residual *before*
    the attention Mix (tech report §2.3, SGLang PLE diagram).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config: Qwen4ExpConfig = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        self.hc_width = config.hc_count * config.hidden_size

        self.use_attn_reduce_scatter_for_moe = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
        )

        if self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        # Every Qwen-FN decoder layer is MoE (512 experts + shared).
        self.mlp = BicameralShardedMoE(
            vllm_config=vllm_config,
            prefix=f"{prefix}.mlp",
        )
        self.mlp.layer_idx = self.layer_idx

        self.attn_hyper_connection = Qwen4ExpGatedResidual(
            config, prefix=f"{prefix}.attn_hyper_connection"
        )
        self.mlp_hyper_connection = Qwen4ExpGatedResidual(
            config, prefix=f"{prefix}.mlp_hyper_connection"
        )

        self.ple: BicameralNgramEmbedding | None = None
        if (self.layer_idx + 1) in (config.ple_layer_ids or []):
            self.ple = BicameralNgramEmbedding(
                config=config,
                prefix=f"{prefix}.ple",
                shard_loader=_get_ngram_loader(),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        token_ids: torch.Tensor | None = None,
        **kwargs: object,
    ):
        # hidden_states is the 4-wide residual [T, nr*d].
        R = hidden_states
        if R.shape[-1] != self.hc_width:
            raise ValueError(
                f"decoder layer expected 4-wide residual "
                f"{self.hc_width}, got {tuple(R.shape)}"
            )

        if self.ple is not None and _env_on("BICAMERAL_NGRAM_MODEL"):
            R = self.ple(R, token_ids)

        full_num_tokens = positions.shape[-1]
        input_is_sequence_parallel = (
            self.use_attn_reduce_scatter_for_moe
            and R.shape[0] != full_num_tokens
        )

        x, R, inj = self.attn_hyper_connection.mix(R)

        if input_is_sequence_parallel:
            x = tensor_model_parallel_all_gather(x, 0)
            x = x[:full_num_tokens]

        if self.layer_type == "linear_attention":
            x = self.linear_attn(hidden_states=x)
        elif self.layer_type == "full_attention":
            x = self.self_attn(hidden_states=x, positions=positions)
        else:
            raise ValueError("Invalid layer_type")

        if self.use_attn_reduce_scatter_for_moe:
            tp_world_size = get_tensor_model_parallel_world_size()
            sp_pad = (-x.shape[0]) % tp_world_size
            x = torch.nn.functional.pad(x, (0, 0, 0, sp_pad))
            x = tensor_model_parallel_reduce_scatter(x, 0)

        R = Qwen4ExpGatedResidual.combine(R, x, inj)

        x, R, inj = self.mlp_hyper_connection.mix(R)
        if self.use_attn_reduce_scatter_for_moe:
            x = self.mlp(x, already_sequence_parallel=True)
        else:
            x = self.mlp(x)
        R = Qwen4ExpGatedResidual.combine(R, x, inj)
        return R, None


# ---------------------------------------------------------------------------
# Transformer body
# ---------------------------------------------------------------------------

@_support_torch_compile_safe
class Qwen4ExpModel(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Qwen-FN ships a `model.language_model.layers.X.…` prefix (because
            # the top-level Qwen4ExpTopConfig wraps a text sub-config).  vLLM
            # constructs the model with no wrapper, so strip both layers.
            "model.language_model.": "",
        },
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_expert.gate_proj": (".shared_expert.gate_up_proj", 0),
            ".shared_expert.up_proj": (".shared_expert.gate_up_proj", 1),
            # Qwen3.5 / Qwen-FN split GDN projections.
            ".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2)),
            ".in_proj_z": (".in_proj_qkvz", 3),
            ".in_proj_b": (".in_proj_ba", 0),
            ".in_proj_a": (".in_proj_ba", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Qwen4ExpConfig = vllm_config.model_config.hf_text_config

        self.config = config
        self.vocab_size = config.vocab_size

        # Token embeddings.
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        # Transformer layers.  vLLM's make_layers() calls layer_fn(prefix=...)
        # (no positional ix) and binds ix via the prefix string suffix.
        def get_layer(prefix: str):
            # Extract the integer layer index from the prefix suffix.
            ix = int(prefix.rsplit(".", 1)[-1])
            layer_type = config.layer_types[ix]
            return Qwen4ExpDecoderLayer(
                vllm_config,
                layer_type=layer_type,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        hc_width = config.hc_count * config.hidden_size
        self.hc_width = hc_width
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], hc_width
        )

        # No model.norm — the final Mix collapses 4-wide → d (llama.cpp /
        # transformers: mixer.hc_norm is the last norm).
        if get_pp_group().is_last_rank:
            self.hyper_connection_mixer = Qwen4ExpGatedResidual(
                config,
                prefix=f"{prefix}.hyper_connection_mixer",
                use_combine=False,
            )
        else:
            self.hyper_connection_mixer = PPMissingLayer()

        self.ple_layer_ids = list(config.ple_layer_ids or [])
        self._cached_token_ids: torch.Tensor | None = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _expand_hc(self, hidden: torch.Tensor) -> torch.Tensor:
        """Tile token embed across all ``hc_count`` residual branches.

        Official ``Qwen4ExpTextModel`` does
        ``hidden_states.repeat(1, 1, hc_count)``.  Zero-filling the extra
        branches (AltUp-style) starves Mix and collapses logits onto
        special tokens.
        """
        return hidden.repeat(1, self.config.hc_count)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            self._cached_token_ids = input_ids
            if hidden_states.shape[-1] != self.hc_width:
                hidden_states = self._expand_hc(hidden_states)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        full_num_tokens = positions.shape[-1]
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if (
                hidden_states.shape[0] != full_num_tokens
                and not layer.use_attn_reduce_scatter_for_moe
            ):
                hidden_states, _ = _all_gather(
                    hidden_states, None, full_num_tokens, self.hc_width
                )

            hidden_states, _ = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=None,
                token_ids=self._cached_token_ids,
            )

            if (layer_idx + 1) in self.aux_hidden_state_layers and hidden_states.shape[0] != full_num_tokens:
                hidden_states, _ = _all_gather(
                    hidden_states, None, full_num_tokens, self.hc_width
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        if hidden_states.shape[0] != full_num_tokens:
            hidden_states, _ = _all_gather(
                hidden_states, None, full_num_tokens, self.hc_width
            )

        return self.hyper_connection_mixer.mix(hidden_states)

    def load_weights(self, weights) -> set[str]:
        def _skip_visual(ws):
            for name, tensor in ws:
                if name.startswith("visual.") or name.startswith("model.visual."):
                    continue
                yield name, tensor

        # Mapper is applied once in Qwen4ExpForCausalLM.load_weights.
        # Applying hf_to_vllm_mapper twice turns in_proj_ba → in_proj_baa
        # because `.in_proj_b` is a prefix of `.in_proj_ba`.
        loader = AutoWeightsLoader(self)
        return loader.load_weights(_skip_visual(weights))

    @property
    def aux_hidden_state_layers(self) -> tuple[int, ...]:
        return getattr(self, "_aux_hidden_state_layers", ())


# ---------------------------------------------------------------------------
# All-gather helper
# ---------------------------------------------------------------------------

def _all_gather(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    full_num_tokens: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if residual is None:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        hidden_states = hidden_states[:full_num_tokens]
        return hidden_states, None
    combined = torch.cat([hidden_states, residual], dim=-1)
    combined = tensor_model_parallel_all_gather(combined, 0)
    combined = combined[:full_num_tokens]
    hidden_states, residual = combined.split([hidden_size, hidden_size], dim=-1)
    return hidden_states, residual


# ---------------------------------------------------------------------------
# Top-level LM
# ---------------------------------------------------------------------------

class Qwen4ExpForCausalLM(
    nn.Module, HasInnerState, IsHybrid, SupportsPP, SupportsMRoPE
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Qwen4ExpConfig = vllm_config.model_config.hf_text_config

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen4Exp does not support 'all' prefix caching; "
                "use --mamba-cache-mode=align"
            )
        self.quant_config = vllm_config.quant_config
        self.config = config

        self.model = Qwen4ExpModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.mtp_head = Qwen4ExpMTPHead(config)

        self._sparse_active: set[str] = set()
        self._sparse_ngram: set[str] = set()
        self._sparse_moe: set[str] = set()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Text-only M-RoPE: T/H/W rows are identical 0..L-1 (Qwen3.5 layout).

        Vision is not served on this duty path; mm_features is ignored.
        """
        seq_len = len(input_tokens)
        pos = torch.arange(seq_len, dtype=torch.int64)
        llm_positions = pos.unsqueeze(0).expand(3, -1).contiguous()
        return llm_positions, 0

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def load_weights(self, weights) -> set[str]:
        """Sparse load: n-gram shards + routed experts stay off GPU0.

        Active partition: GDN, GQA (incl. q/k/v), GR Mix/Combine, shared
        expert, router, embeddings, PLE projections.  QSA indexer is
        fabric-side (BICAMERAL §6.4) so those tensors are skipped.
        q_proj / k_proj are NOT skipped — they fuse into qkv_proj.
        """
        active_names = self._sparse_table_active_names
        ngram_names = self._sparse_table_ngram_names
        moe_names = self._sparse_table_moe_names

        def filtered():
            for name, tensor in weights:
                bucket = classify_weight(name)
                if bucket == "ngram":
                    ngram_names.add(name)
                    continue
                if bucket == "moe_expert":
                    moe_names.add(name)
                    continue
                if should_skip_weight(name, skip_indexer=True):
                    continue
                active_names.add(name)
                yield strip_language_model_prefix(name), tensor

        loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
        result = loader.load_weights(filtered(), mapper=self.model.hf_to_vllm_mapper)
        logger.info(
            "Sparse load: %d active, %d n-gram skipped (igram cache), "
            "%d MoE expert skipped (expert cache)",
            len(active_names), len(ngram_names), len(moe_names),
        )
        return result

    @property
    def _sparse_table_active_names(self) -> set[str]:
        return self._sparse_active

    @property
    def _sparse_table_ngram_names(self) -> set[str]:
        return self._sparse_ngram

    @property
    def _sparse_table_moe_names(self) -> set[str]:
        return self._sparse_moe

    def post_init(self) -> None:
        """Hook called after the model is constructed.

        Initializes the sparse weight classifier sets and (when
        available) attaches the bicameral caches to each MoE / PLE
        module.  vLLM calls this automatically if defined.
        """
        if not hasattr(self, "_sparse_active"):
            self._sparse_active = set()
            self._sparse_ngram = set()
            self._sparse_moe = set()
