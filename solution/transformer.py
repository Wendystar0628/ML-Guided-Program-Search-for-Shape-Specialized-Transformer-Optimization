"""Current Transformer implementation optimized by the performance mainline."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class _SelfAttention(nn.Module):
    """Reference-compatible multi-head self-attention."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = value.shape
        return (
            value.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        value: torch.Tensor,
        valid_token_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = value.shape
        query = self._split_heads(self.q_proj(value))
        key = self._split_heads(self.k_proj(value))
        projected_value = self._split_heads(self.v_proj(value))
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if causal:
            causal_mask = torch.ones(
                (sequence_length, sequence_length),
                device=value.device,
                dtype=torch.bool,
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))
        if valid_token_mask is not None:
            invalid_keys = ~valid_token_mask[:, None, None, :]
            scores = scores.masked_fill(invalid_keys, float("-inf"))

        probabilities = torch.softmax(scores.float(), dim=-1).to(dtype=value.dtype)
        context = torch.matmul(probabilities, projected_value)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.d_model)
        )
        output = self.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class _TransformerBlock(nn.Module):
    """Reference-compatible pre-normalized Transformer block."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = _SelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        value: torch.Tensor,
        valid_token_mask: torch.Tensor | None,
        causal: bool,
    ) -> torch.Tensor:
        value = value + self.attention(
            self.norm1(value),
            valid_token_mask,
            causal,
        )
        value = value + self.ffn_out(
            F.gelu(self.ffn_in(self.norm2(value)), approximate="none")
        )
        if valid_token_mask is not None:
            value = value.masked_fill(~valid_token_mask[..., None], 0)
        return value


class UserOptimizedTransformer(nn.Module):
    """Solution entry with the exact official constructor and forward API."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                _TransformerBlock(
                    config.d_model,
                    config.num_heads,
                    config.ffn_dim,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, valid_token_mask, self.config.causal)
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x
