from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def _attention_bias(padding_mask: torch.Tensor, value_mask: torch.Tensor) -> torch.Tensor:
    """
    Constructs a bias tensor that masks padded tokens and disallows masked queries
    from attending to masked keys.
    """
    bsz, seq_len = padding_mask.shape
    bias = torch.zeros(bsz, 1, seq_len, seq_len, device=padding_mask.device)

    pad_bias = padding_mask[:, None, None, :]
    bias = bias + pad_bias

    vm_q = value_mask[:, None, :, None]
    vm_k = value_mask[:, None, None, :]
    bias = bias + vm_q * vm_k

    return bias


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        bsz, seq_len, dim = x.size()

        def shape_proj(proj):
            return (
                proj(x)
                .view(bsz, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

        q = shape_proj(self.q_proj)
        k = shape_proj(self.k_proj)
        v = shape_proj(self.v_proj)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask.bool(), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v).transpose(1, 2).contiguous()
        context = context.view(bsz, seq_len, dim)
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x


@dataclass
class ValueMaskedConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.1


class ValueMaskedTransformer(nn.Module):
    def __init__(self, config: ValueMaskedConfig, pad_token_id: int) -> None:
        super().__init__()
        self.config = config
        self.pad_token_id = pad_token_id

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.value_projection = nn.Linear(1, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model, config.num_heads, config.d_ff, config.dropout
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.value_head = nn.Linear(config.d_model, 1)

    def forward(
        self,
        *,
        token_ids: torch.Tensor,
        values: torch.Tensor,
        value_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)

        masked_values = torch.where(
            value_mask.bool(), torch.zeros_like(values), values
        ).unsqueeze(-1)

        x = (
            self.token_embedding(token_ids)
            + self.value_projection(masked_values)
            + self.position_embedding(positions)
        )

        bias = _attention_bias(padding_mask, value_mask)
        for layer in self.layers:
            x = layer(x, bias)

        x = self.norm(x)
        value_pred = self.value_head(x).squeeze(-1)
        return value_pred
