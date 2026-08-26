"""Transformer implementation optimized for the project performance mainline."""

from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class _SelfAttention(nn.Module):
    """Reference-compatible attention with a one-time packed QKV projection."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
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

    def _explicit_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        projected_value: torch.Tensor,
        invalid_key_mask: torch.Tensor | None,
        causal_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Preserve the official low-precision operation and accumulation order."""

        sequence_length = query.shape[-2]
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores.mul_(self.scale)

        if causal_mask is not None:
            scores.masked_fill_(
                causal_mask[:sequence_length, :sequence_length],
                float("-inf"),
            )
        if invalid_key_mask is not None:
            scores.masked_fill_(invalid_key_mask, float("-inf"))

        probabilities = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
        return torch.matmul(probabilities, projected_value)

    def forward(
        self,
        value: torch.Tensor,
        valid_token_mask: torch.Tensor | None = None,
        invalid_key_mask: torch.Tensor | None = None,
        invalid_query_mask: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = value.shape
        query, key, projected_value = self.qkv_proj(value).chunk(3, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        projected_value = self._split_heads(projected_value)

        # The validated short, non-causal CUDA float32 region uses fused SDPA.
        # Other regions retain the reference order as the numerical fallback.
        use_fp32_sdpa = (
            value.is_cuda
            and value.dtype == torch.float32
            and causal_mask is None
            and sequence_length <= 128
        )
        if use_fp32_sdpa:
            attention_mask = (
                None if valid_token_mask is None else valid_token_mask[:, None, None, :]
            )
            context = F.scaled_dot_product_attention(
                query,
                key,
                projected_value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            context = self._explicit_attention(
                query,
                key,
                projected_value,
                invalid_key_mask,
                causal_mask,
            )

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.d_model)
        )
        output = self.out_proj(context)
        if invalid_query_mask is not None:
            output.masked_fill_(invalid_query_mask, 0)
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
        invalid_key_mask: torch.Tensor | None,
        invalid_query_mask: torch.Tensor | None,
        causal_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        value = value + self.attention(
            self.norm1(value),
            valid_token_mask,
            invalid_key_mask,
            invalid_query_mask,
            causal_mask,
        )
        value = value + self.ffn_out(
            F.gelu(self.ffn_in(self.norm2(value)), approximate="none")
        )
        if invalid_query_mask is not None:
            value.masked_fill_(invalid_query_mask, 0)
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

        causal_mask = None
        if config.causal:
            causal_mask = torch.ones(
                config.seq_len,
                config.seq_len,
                dtype=torch.bool,
            ).triu(diagonal=1)
        self.register_buffer("_causal_mask", causal_mask, persistent=False)

    def describe_execution_path(self) -> dict[str, Any]:
        """Describe the intended path without claiming an observed SDPA backend."""

        parameter = next(self.parameters())
        use_fp32_sdpa = (
            parameter.is_cuda
            and parameter.dtype == torch.float32
            and not self.config.causal
            and self.config.seq_len <= 128
        )
        return {
            "qkv_projection": "packed",
            "attention_policy": (
                "auto_sdpa" if use_fp32_sdpa else "explicit_reference_order"
            ),
            "selected_attention_backend": "auto" if use_fp32_sdpa else "explicit",
            "causal_mask": "shared_buffer" if self.config.causal else "none",
            "token_mask_preprocessing": "shared_broadcast_views",
        }

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        invalid_key_mask = None
        invalid_query_mask = None
        if valid_token_mask is not None:
            invalid_token_mask = ~valid_token_mask
            invalid_key_mask = invalid_token_mask[:, None, None, :]
            invalid_query_mask = invalid_token_mask[..., None]

        for layer in self.layers:
            x = layer(
                x,
                valid_token_mask,
                invalid_key_mask,
                invalid_query_mask,
                self._causal_mask,
            )
        x = self.final_norm(x)
        if invalid_query_mask is not None:
            x.masked_fill_(invalid_query_mask, 0)
        return x


def _packed_source_keys(target_key: str) -> tuple[str, str, str] | None:
    marker = ".attention.qkv_proj."
    if marker not in target_key:
        return None
    prefix, suffix = target_key.split(marker, maxsplit=1)
    attention_prefix = f"{prefix}.attention."
    return tuple(
        f"{attention_prefix}{projection}.{suffix}"
        for projection in ("q_proj", "k_proj", "v_proj")
    )


def copy_model_weights(
    baseline: nn.Module,
    optimized: nn.Module,
    strict: bool = True,
) -> None:
    """Copy official weights and derive packed QKV tensors exactly once."""

    source_state = baseline.state_dict()
    target_state = optimized.state_dict()
    mapped_state: dict[str, torch.Tensor] = {}
    consumed_source_keys: set[str] = set()

    for target_key, target_tensor in target_state.items():
        packed_keys = _packed_source_keys(target_key)
        if packed_keys is None:
            if target_key in source_state:
                mapped_state[target_key] = source_state[target_key]
                consumed_source_keys.add(target_key)
            continue

        if not all(source_key in source_state for source_key in packed_keys):
            continue
        packed_tensor = torch.cat(
            [source_state[source_key] for source_key in packed_keys],
            dim=0,
        )
        if packed_tensor.shape != target_tensor.shape:
            raise RuntimeError(
                f"packed tensor shape mismatch for {target_key}: "
                f"source={tuple(packed_tensor.shape)}, "
                f"target={tuple(target_tensor.shape)}"
            )
        mapped_state[target_key] = packed_tensor
        consumed_source_keys.update(packed_keys)

    missing_target_keys = sorted(set(target_state) - set(mapped_state))
    unused_source_keys = sorted(set(source_state) - consumed_source_keys)
    if strict and (missing_target_keys or unused_source_keys):
        raise RuntimeError(
            "strict weight mapping failed; "
            f"missing target keys={missing_target_keys}, "
            f"unused source keys={unused_source_keys}"
        )

    incompatible = optimized.load_state_dict(mapped_state, strict=strict)
    if not strict:
        warning_parts = []
        if missing_target_keys or incompatible.missing_keys:
            warning_parts.append(
                "missing target keys="
                f"{sorted(set(missing_target_keys + incompatible.missing_keys))}"
            )
        if unused_source_keys:
            warning_parts.append(f"unused source keys={unused_source_keys}")
        if incompatible.unexpected_keys:
            warning_parts.append(
                f"unexpected mapped keys={sorted(incompatible.unexpected_keys)}"
            )
        if warning_parts:
            warnings.warn(
                "non-strict weight mapping: " + ", ".join(warning_parts),
                RuntimeWarning,
                stacklevel=2,
            )
