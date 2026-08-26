"""Transformer implementation optimized for the project performance mainline."""

from __future__ import annotations

import os
import warnings
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .kernels import (
    TRITON_ATTENTION_SOFTMAX_AVAILABLE,
    TRITON_QKV_LAYOUT_AVAILABLE,
    TRITON_RESIDUAL_AVAILABLE,
    can_use_triton_attention_softmax,
    can_use_triton_qkv_layout,
    can_use_triton_residual,
    triton_qkv_to_bhsd,
    triton_residual_add_padding,
    triton_scale_mask_softmax,
)


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
        self.qkv_layout_policy = "auto"
        self.attention_policy = "auto"

    def configure_runtime_policy(self, qkv_layout: str, attention: str) -> None:
        """Configure the small set of real execution candidates."""

        self.qkv_layout_policy = qkv_layout
        self.attention_policy = attention

    def _split_heads_view(self, value: torch.Tensor) -> torch.Tensor:
        """Expose the head dimension without materializing a layout copy."""

        batch_size, sequence_length, _ = value.shape
        return value.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _split_heads_contiguous(self, value: torch.Tensor) -> torch.Tensor:
        """Materialize the official head layout as a conservative candidate."""

        return self._split_heads_view(value).contiguous()

    def _can_use_fp32_sdpa(
        self,
        value: torch.Tensor,
        sequence_length: int,
        causal_mask: torch.Tensor | None,
    ) -> bool:
        return (
            value.is_cuda
            and value.dtype == torch.float32
            and causal_mask is None
            and sequence_length <= 128
        )

    def _project_heads(
        self,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project QKV once and select the safe layout implementation."""

        packed_qkv = self.qkv_proj(value)
        if self.qkv_layout_policy == "triton" and can_use_triton_qkv_layout(
            packed_qkv,
            self.num_heads,
        ):
            packed_heads = triton_qkv_to_bhsd(packed_qkv, self.num_heads)
            query, key, projected_value = packed_heads.unbind(dim=0)
            return query, key, projected_value

        query, key, projected_value = packed_qkv.chunk(3, dim=-1)
        split_heads = (
            self._split_heads_contiguous
            if self.qkv_layout_policy == "torch_contiguous"
            else self._split_heads_view
        )
        return (
            split_heads(query),
            split_heads(key),
            split_heads(projected_value),
        )

    def _explicit_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        projected_value: torch.Tensor,
        valid_token_mask: torch.Tensor | None,
        invalid_key_mask: torch.Tensor | None,
        causal_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Preserve the official low-precision operation and accumulation order."""

        sequence_length = query.shape[-2]
        scores = torch.matmul(query, key.transpose(-2, -1))
        use_triton_softmax = (
            self.attention_policy == "triton_softmax"
            and can_use_triton_attention_softmax(
                scores,
                valid_token_mask,
                self.head_dim,
            )
        )
        if use_triton_softmax:
            probabilities = triton_scale_mask_softmax(
                scores,
                valid_token_mask,
                head_dim=self.head_dim,
                scale=self.scale,
                causal=causal_mask is not None,
            )
        else:
            scores.mul_(self.scale)

            if causal_mask is not None:
                scores.masked_fill_(
                    causal_mask[:sequence_length, :sequence_length],
                    float("-inf"),
                )
            if invalid_key_mask is not None:
                scores.masked_fill_(invalid_key_mask, float("-inf"))

            probabilities = torch.softmax(scores.float(), dim=-1).to(
                dtype=query.dtype
            )
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
        query, key, projected_value = self._project_heads(value)

        # Low-precision and unsupported shapes retain the reference order.
        use_fp32_sdpa = (
            self.attention_policy in ("auto", "fp32_sdpa")
            and self._can_use_fp32_sdpa(
                value,
                sequence_length,
                causal_mask,
            )
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
                valid_token_mask,
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
        self.use_packed_ffn = False
        self.use_triton_residual = False

    def configure_runtime_policy(
        self,
        qkv_layout: str,
        attention: str,
        use_packed_ffn: bool,
        use_triton_residual: bool,
    ) -> None:
        """Apply one resolved policy to the block and its attention module."""

        self.attention.configure_runtime_policy(qkv_layout, attention)
        self.use_packed_ffn = use_packed_ffn
        self.use_triton_residual = use_triton_residual

    def _packed_token_ffn(
        self,
        value: torch.Tensor,
        valid_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run token-wise normalization and FFN only for valid token rows."""

        flat_value = value.reshape(-1, value.shape[-1])
        valid_indices = torch.nonzero(
            valid_token_mask.reshape(-1),
            as_tuple=False,
        ).flatten()
        packed_value = flat_value.index_select(0, valid_indices)
        packed_update = self.ffn_out(
            F.gelu(self.ffn_in(self.norm2(packed_value)), approximate="none")
        )
        if self.use_triton_residual:
            flat_update = torch.zeros_like(flat_value)
            flat_update.index_copy_(0, valid_indices, packed_update)
            update = flat_update.view_as(value)
            if can_use_triton_residual(value, update, valid_token_mask):
                return triton_residual_add_padding(
                    value,
                    update,
                    valid_token_mask,
                )

        packed_value = packed_value + packed_update
        output = torch.zeros_like(flat_value)
        output.index_copy_(0, valid_indices, packed_value)
        return output.view_as(value)

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
        if self.use_packed_ffn and valid_token_mask is not None:
            return self._packed_token_ffn(value, valid_token_mask)

        ffn_update = self.ffn_out(F.gelu(
            self.ffn_in(self.norm2(value)),
            approximate="none",
        ))
        if (
            self.use_triton_residual
            and valid_token_mask is not None
            and can_use_triton_residual(value, ffn_update, valid_token_mask)
        ):
            return triton_residual_add_padding(
                value,
                ffn_update,
                valid_token_mask,
            )

        value = value + ffn_update
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

        requested_policy = os.environ.get("TRANSFORMER_OPT_POLICY", "auto")
        self._configure_named_policy(requested_policy)

    def _apply_runtime_policy(
        self,
        *,
        requested_policy: str,
        qkv_layout: str,
        attention: str,
        use_packed_ffn: bool,
        use_triton_residual: bool,
    ) -> None:
        self.requested_policy = requested_policy
        self.requested_qkv_layout = qkv_layout
        self.requested_attention = attention
        self.use_packed_ffn = use_packed_ffn
        self.use_triton_residual = use_triton_residual
        for layer in self.layers:
            layer.configure_runtime_policy(
                qkv_layout,
                attention,
                use_packed_ffn,
                use_triton_residual,
            )

    def _configure_named_policy(self, policy: str) -> None:
        """Resolve the project-level policy selected by the benchmark process."""

        normalized = policy.strip().lower()
        named_policies = {
            "auto": ("auto", "auto", False, False),
            "torch": ("torch_contiguous", "auto", False, False),
            "triton": ("triton", "triton_softmax", False, False),
            "padding": ("view", "auto", False, True),
            "packed": ("view", "auto", True, False),
        }
        if normalized not in named_policies:
            choices = ", ".join(sorted(named_policies))
            raise ValueError(
                f"unknown TRANSFORMER_OPT_POLICY={policy!r}; expected one of {choices}"
            )
        qkv_layout, attention, use_packed_ffn, use_triton_residual = named_policies[
            normalized
        ]
        self._apply_runtime_policy(
            requested_policy=normalized,
            qkv_layout=qkv_layout,
            attention=attention,
            use_packed_ffn=use_packed_ffn,
            use_triton_residual=use_triton_residual,
        )

    def configure_runtime_policy(
        self,
        *,
        qkv_layout: str = "auto",
        attention: str = "auto",
    ) -> None:
        """Select concrete candidates for controlled benchmark comparisons."""

        qkv_layout_choices = {"auto", "view", "triton", "torch_contiguous"}
        attention_choices = {"auto", "explicit", "fp32_sdpa", "triton_softmax"}
        if qkv_layout not in qkv_layout_choices:
            choices = ", ".join(sorted(qkv_layout_choices))
            raise ValueError(f"unknown qkv_layout={qkv_layout!r}; expected {choices}")
        if attention not in attention_choices:
            choices = ", ".join(sorted(attention_choices))
            raise ValueError(f"unknown attention={attention!r}; expected {choices}")
        self._apply_runtime_policy(
            requested_policy="custom",
            qkv_layout=qkv_layout,
            attention=attention,
            use_packed_ffn=False,
            use_triton_residual=False,
        )

    def describe_execution_path(self) -> dict[str, Any]:
        """Describe the intended path without claiming an observed SDPA backend."""

        parameter = next(self.parameters())
        fp32_sdpa_eligible = (
            parameter.is_cuda
            and parameter.dtype == torch.float32
            and not self.config.causal
            and self.config.seq_len <= 128
        )
        use_fp32_sdpa = (
            self.requested_attention in ("auto", "fp32_sdpa")
            and fp32_sdpa_eligible
        )
        head_dim = self.config.d_model // self.config.num_heads
        triton_layout_eligible = (
            TRITON_QKV_LAYOUT_AVAILABLE
            and parameter.is_cuda
            and parameter.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and 16 <= head_dim <= 128
            and head_dim & (head_dim - 1) == 0
        )
        triton_softmax_eligible = (
            TRITON_ATTENTION_SOFTMAX_AVAILABLE
            and parameter.is_cuda
            and parameter.dtype == torch.float16
            and self.config.seq_len in (512, 2048)
            and head_dim == 64
        )
        triton_residual_eligible = (
            TRITON_RESIDUAL_AVAILABLE
            and parameter.is_cuda
            and parameter.dtype in (torch.float16, torch.bfloat16, torch.float32)
        )
        fallback_reasons = []
        if self.requested_qkv_layout == "triton":
            if triton_layout_eligible:
                resolved_qkv_layout = "triton_single_pass"
            else:
                resolved_qkv_layout = "view_fallback"
                fallback_reasons.append("triton_qkv_layout_not_available_or_compatible")
        elif self.requested_qkv_layout == "torch_contiguous":
            resolved_qkv_layout = "torch_three_contiguous_copies"
        else:
            resolved_qkv_layout = "torch_zero_copy_view"

        if (
            self.requested_attention == "triton_softmax"
            and triton_softmax_eligible
        ):
            resolved_attention = "explicit_qk_triton_softmax_pv"
        elif use_fp32_sdpa:
            resolved_attention = "fp32_sdpa"
        else:
            resolved_attention = "explicit_reference_order"
            if self.requested_attention == "fp32_sdpa":
                fallback_reasons.append("fp32_sdpa_not_eligible")
            elif self.requested_attention == "triton_softmax":
                fallback_reasons.append("triton_attention_softmax_not_eligible")
        if self.use_triton_residual and not triton_residual_eligible:
            fallback_reasons.append("triton_residual_fusion_not_eligible")

        if self.requested_policy == "triton" and not (
            triton_layout_eligible and triton_softmax_eligible
        ):
            selected_policy = (
                "triton_partial"
                if triton_layout_eligible or triton_softmax_eligible
                else "torch_fallback"
            )
        else:
            selected_policy = self.requested_policy
        if resolved_attention == "explicit_qk_triton_softmax_pv":
            shape_route = "triton_long_or_masked_attention"
        elif use_fp32_sdpa:
            shape_route = "short_fp32_sdpa"
        elif self.config.seq_len >= 512:
            shape_route = "long_or_masked_reference_attention"
        elif parameter.dtype == torch.bfloat16 and self.config.d_model >= 1024:
            shape_route = "wide_bf16_reference_attention"
        else:
            shape_route = "general_reference_attention"
        return {
            "requested_policy": self.requested_policy,
            "selected_policy": selected_policy,
            "qkv_projection": "packed",
            "requested_qkv_layout": self.requested_qkv_layout,
            "resolved_qkv_layout": resolved_qkv_layout,
            "qkv_head_layout": resolved_qkv_layout,
            "requested_attention": self.requested_attention,
            "resolved_attention": resolved_attention,
            "attention_policy": resolved_attention,
            "selected_attention_backend": (
                "triton_softmax"
                if resolved_attention == "explicit_qk_triton_softmax_pv"
                else "auto" if use_fp32_sdpa else "explicit"
            ),
            "attention_candidate_status": (
                "experimental_requires_correctness_gate"
                if self.requested_attention == "triton_softmax"
                else "validated_route"
            ),
            "shape_route": shape_route,
            "block_fusion": (
                "triton_residual_add_padding_when_masked"
                if self.use_triton_residual and triton_residual_eligible
                else "torch_residual_fallback"
                if self.use_triton_residual
                else "none"
            ),
            "padding_route": (
                "packed_valid_token_ffn"
                if self.use_packed_ffn
                else "full_ffn_with_fused_padding_residual"
                if self.use_triton_residual and triton_residual_eligible
                else "shared_mask_only"
            ),
            "fallback_reason": fallback_reasons or None,
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
