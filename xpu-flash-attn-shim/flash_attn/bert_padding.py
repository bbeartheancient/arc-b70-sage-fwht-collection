"""Minimal XPU-compatible replacement for flash_attn.bert_padding."""
from typing import Tuple

import torch


def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    """Scatter ragged ``hidden_states`` into a padded tensor.

    Args:
        hidden_states: [total_tokens, ...]
        indices: [total_tokens] positions in the flattened padded batch.
        batch: batch size.
        seqlen: padded sequence length.
    Returns:
        [batch, seqlen, ...]
    """
    output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    output[indices] = hidden_states
    return output.view(batch, seqlen, *hidden_states.shape[1:])


def unpad_input(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Strip padding from a padded tensor.

    Args:
        hidden_states: [batch, seqlen, ...]
        attention_mask: [batch, seqlen] bool/byte tensor (1 = real token).
    Returns:
        hidden_states: [total_tokens, ...]
        indices: [total_tokens] positions in the flattened padded batch.
        cu_seqlens: [batch+1] cumulative token counts.
        max_seqlen: maximum sequence length in the batch.
    """
    batch, seqlen = hidden_states.shape[:2]
    valid_mask = attention_mask.bool().view(-1)
    indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
    cu_seqlens = torch.cat([
        attention_mask.new_zeros(1),
        attention_mask.sum(dim=1).cumsum(dim=0),
    ], dim=0).int()
    return (
        hidden_states.view(batch * seqlen, *hidden_states.shape[2:])[indices],
        indices,
        cu_seqlens,
        int(attention_mask.sum(dim=1).max().item()),
    )
