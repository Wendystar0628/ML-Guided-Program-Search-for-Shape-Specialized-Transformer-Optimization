"""Transformer implementation optimized for the project performance mainline."""

from __future__ import annotations

import os
import warnings
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .cuda_graph import CudaGraphReplay
from .dispatch import OfflineDispatcher
from .kernels import (
    TRITON_ATTENTION_PREPROCESS_AVAILABLE,
    TRITON_ATTENTION_PV_AVAILABLE,
    TRITON_ATTENTION_SOFTMAX_AVAILABLE,
    TRITON_QKV_LAYOUT_AVAILABLE,
    TRITON_RESIDUAL_AVAILABLE,
    can_use_triton_attention_preprocess,
    can_use_triton_attention_softmax,
    can_use_triton_fp32_probability_value,
    can_use_triton_qkv_layout,
    can_use_triton_residual,
    triton_fp32_probability_value,
    triton_qkv_to_bhsd,
    triton_residual_add_padding,
    triton_scale_mask_softmax,
    triton_scale_mask_to_fp32,
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
        causal: bool,
    ) -> bool:
        return (
            value.is_cuda
            and value.dtype == torch.float32
            and not causal
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
        score_mask: torch.Tensor | None,
        causal: bool,
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
                causal=causal,
            )
            return torch.matmul(probabilities, projected_value)
        else:
            use_triton_preprocess = (
                (self.attention_policy == "auto" and sequence_length == 2048)
                or self.attention_policy in {"triton_preprocess", "triton_pv"}
            ) and can_use_triton_attention_preprocess(
                scores,
                valid_token_mask,
                self.head_dim,
            )
            if use_triton_preprocess:
                softmax_input = triton_scale_mask_to_fp32(
                    scores,
                    valid_token_mask,
                    head_dim=self.head_dim,
                    scale=self.scale,
                    causal=causal,
                )
            else:
                scores.mul_(self.scale)

                if score_mask is not None:
                    scores.masked_fill_(score_mask, float("-inf"))
                softmax_input = scores

            if (
                self.attention_policy != "reference"
                and scores.dtype in (torch.float16, torch.bfloat16)
                and sequence_length <= 512
            ):
                probabilities_fp32 = torch.softmax(
                    softmax_input,
                    dim=-1,
                    dtype=torch.float32,
                )
            else:
                probabilities_fp32 = torch.softmax(
                    softmax_input.float(),
                    dim=-1,
                )

            if (
                self.attention_policy == "triton_pv"
                and use_triton_preprocess
                and can_use_triton_fp32_probability_value(
                    probabilities_fp32,
                    projected_value,
                )
            ):
                return triton_fp32_probability_value(
                    probabilities_fp32,
                    projected_value,
                )

            probabilities = probabilities_fp32.to(dtype=query.dtype)
            return torch.matmul(probabilities, projected_value)

    def forward(
        self,
        value: torch.Tensor,
        valid_token_mask: torch.Tensor | None = None,
        score_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = value.shape
        query, key, projected_value = self._project_heads(value)

        # Low-precision and unsupported shapes retain the reference order.
        use_fp32_sdpa = self.attention_policy in (
            "auto",
            "fp32_sdpa",
        ) and self._can_use_fp32_sdpa(
            value,
            sequence_length,
            causal,
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
                score_mask,
                causal,
            )

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.d_model)
        )
        output = self.out_proj(context)
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
        self.ffn_policy = "exact"

    def configure_runtime_policy(
        self,
        qkv_layout: str,
        attention: str,
        use_packed_ffn: bool,
        use_triton_residual: bool,
        ffn_policy: str,
    ) -> None:
        """Apply one resolved policy to the block and its attention module."""

        self.attention.configure_runtime_policy(qkv_layout, attention)
        self.use_packed_ffn = use_packed_ffn
        self.use_triton_residual = use_triton_residual
        self.ffn_policy = ffn_policy

    def _can_use_wide_epilogue(self, value: torch.Tensor) -> bool:
        """Limit the approximate cuBLASLt epilogue to the measured Wide shape."""

        return (
            value.is_cuda
            and value.dtype == torch.bfloat16
            and value.shape == (16, 256, 1024)
            and self.ffn_in.weight.shape == (4096, 1024)
            and self.ffn_in.bias is not None
        )

    def _ffn_hidden(self, value: torch.Tensor) -> torch.Tensor:
        """Run the first FFN projection through the selected bounded candidate."""

        normalized = self.norm2(value)
        if self.ffn_policy == "cublaslt_tanh_gelu_epilogue" and (
            self._can_use_wide_epilogue(normalized)
        ):
            flattened = normalized.reshape(-1, normalized.shape[-1])
            hidden = torch.ops.aten._addmm_activation.default(
                self.ffn_in.bias,
                flattened,
                self.ffn_in.weight.t(),
                beta=1,
                alpha=1,
                use_gelu=True,
            )
            return hidden.view(*normalized.shape[:-1], self.ffn_in.out_features)
        return F.gelu(self.ffn_in(normalized), approximate="none")

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
        score_mask: torch.Tensor | None,
        invalid_query_mask: torch.Tensor | None,
        causal: bool,
    ) -> torch.Tensor:
        value = value + self.attention(
            self.norm1(value),
            valid_token_mask,
            score_mask,
            causal,
        )
        if self.use_packed_ffn and valid_token_mask is not None:
            return self._packed_token_ffn(value, valid_token_mask)

        ffn_update = self.ffn_out(self._ffn_hidden(value))
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
        self._cuda_graph_replay: CudaGraphReplay | None = None

        causal_mask = None
        if config.causal:
            causal_mask = torch.ones(
                config.seq_len,
                config.seq_len,
                dtype=torch.bool,
            ).triu(diagonal=1)
        self.register_buffer("_causal_mask", causal_mask, persistent=False)

        requested_policy = os.environ.get("TRANSFORMER_OPT_POLICY", "dispatch")
        self._dispatcher: OfflineDispatcher | None = None
        self._dispatch_signature: tuple[object, ...] | None = None
        self.dispatch_policy: str | None = None
        if requested_policy.strip().lower() == "dispatch":
            self._dispatcher = OfflineDispatcher()
            self._configure_named_policy("auto")
            self.requested_policy = "dispatch"
            self.dispatch_policy = "auto"
        else:
            self._configure_named_policy(requested_policy)

    def _apply(self, function: Any, recurse: bool = True) -> UserOptimizedTransformer:
        """Invalidate captured parameter addresses after a module transform."""

        result = super()._apply(function, recurse=recurse)
        self._cuda_graph_replay = None
        self._dispatch_signature = None
        return result

    def _resolve_dispatch(self, value: torch.Tensor | None = None) -> None:
        """Resolve an offline route once per static input signature."""

        if self._dispatcher is None:
            return
        if value is None:
            parameter = next(self.parameters())
            device = parameter.device
            dtype = parameter.dtype
            shape = (
                self.config.batch_size,
                self.config.seq_len,
                self.config.d_model,
            )
        else:
            device = value.device
            dtype = value.dtype
            shape = tuple(value.shape)
        signature = (device.type, device.index, dtype, shape)
        if signature == self._dispatch_signature:
            return
        policy = self._dispatcher.resolve(
            self.config,
            device=device,
            dtype=dtype,
            shape=shape,
        )
        self._configure_named_policy(policy)
        self.requested_policy = "dispatch"
        self.dispatch_policy = policy
        self._dispatch_signature = signature

    def _apply_runtime_policy(
        self,
        *,
        requested_policy: str,
        qkv_layout: str,
        attention: str,
        use_packed_ffn: bool,
        use_triton_residual: bool,
        ffn_policy: str,
        use_cuda_graph: bool,
    ) -> None:
        self.requested_policy = requested_policy
        self.requested_qkv_layout = qkv_layout
        self.requested_attention = attention
        self.use_packed_ffn = use_packed_ffn
        self.use_triton_residual = use_triton_residual
        self.requested_ffn = ffn_policy
        self.use_cuda_graph = use_cuda_graph
        for layer in self.layers:
            layer.configure_runtime_policy(
                qkv_layout,
                attention,
                use_packed_ffn,
                use_triton_residual,
                ffn_policy,
            )

    def _configure_named_policy(self, policy: str) -> None:
        """Resolve the project-level policy selected by the benchmark process."""

        normalized = policy.strip().lower()
        named_policies = {
            "auto": ("auto", "auto", False, False, "exact", False),
            "reference": ("view", "reference", False, False, "exact", False),
            "torch": (
                "torch_contiguous",
                "auto",
                False,
                False,
                "exact",
                False,
            ),
            "triton": (
                "triton",
                "triton_softmax",
                False,
                False,
                "exact",
                False,
            ),
            "preprocess": (
                "view",
                "triton_preprocess",
                False,
                False,
                "exact",
                False,
            ),
            "long-pv": ("view", "triton_pv", False, False, "exact", False),
            "wide-epilogue": (
                "view",
                "auto",
                False,
                False,
                "cublaslt_tanh_gelu_epilogue",
                False,
            ),
            "cuda-graph": ("auto", "auto", False, False, "exact", True),
            "padding": ("view", "auto", False, True, "exact", False),
            "packed": ("view", "auto", True, False, "exact", False),
        }
        if normalized not in named_policies:
            choices = ", ".join(sorted(named_policies))
            raise ValueError(
                f"unknown TRANSFORMER_OPT_POLICY={policy!r}; expected one of {choices}"
            )
        (
            qkv_layout,
            attention,
            use_packed_ffn,
            use_triton_residual,
            ffn_policy,
            use_cuda_graph,
        ) = named_policies[normalized]
        self._apply_runtime_policy(
            requested_policy=normalized,
            qkv_layout=qkv_layout,
            attention=attention,
            use_packed_ffn=use_packed_ffn,
            use_triton_residual=use_triton_residual,
            ffn_policy=ffn_policy,
            use_cuda_graph=use_cuda_graph,
        )

    def configure_runtime_policy(
        self,
        *,
        qkv_layout: str = "auto",
        attention: str = "auto",
        ffn: str = "exact",
    ) -> None:
        """Select concrete candidates for controlled benchmark comparisons."""

        qkv_layout_choices = {"auto", "view", "triton", "torch_contiguous"}
        attention_choices = {
            "auto",
            "explicit",
            "reference",
            "fp32_sdpa",
            "triton_preprocess",
            "triton_pv",
            "triton_softmax",
        }
        ffn_choices = {"exact", "cublaslt_tanh_gelu_epilogue"}
        if qkv_layout not in qkv_layout_choices:
            choices = ", ".join(sorted(qkv_layout_choices))
            raise ValueError(f"unknown qkv_layout={qkv_layout!r}; expected {choices}")
        if attention not in attention_choices:
            choices = ", ".join(sorted(attention_choices))
            raise ValueError(f"unknown attention={attention!r}; expected {choices}")
        if ffn not in ffn_choices:
            choices = ", ".join(sorted(ffn_choices))
            raise ValueError(f"unknown ffn={ffn!r}; expected {choices}")
        self._apply_runtime_policy(
            requested_policy="custom",
            qkv_layout=qkv_layout,
            attention=attention,
            use_packed_ffn=False,
            use_triton_residual=False,
            ffn_policy=ffn,
            use_cuda_graph=False,
        )

    def describe_execution_path(self) -> dict[str, Any]:
        """Describe the intended path without claiming an observed SDPA backend."""

        self._resolve_dispatch()
        parameter = next(self.parameters())
        fp32_sdpa_eligible = (
            parameter.is_cuda
            and parameter.dtype == torch.float32
            and not self.config.causal
            and self.config.seq_len <= 128
        )
        use_fp32_sdpa = (
            self.requested_attention in ("auto", "fp32_sdpa") and fp32_sdpa_eligible
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
        triton_preprocess_eligible = (
            TRITON_ATTENTION_PREPROCESS_AVAILABLE
            and parameter.is_cuda
            and parameter.dtype == torch.float16
            and (self.config.seq_len, head_dim) in {(64, 32), (2048, 64)}
        )
        triton_pv_eligible = (
            TRITON_ATTENTION_PV_AVAILABLE
            and triton_preprocess_eligible
            and self.config.batch_size == 1
            and self.config.seq_len == 2048
            and self.config.num_heads == 8
            and head_dim == 64
        )
        wide_epilogue_eligible = (
            parameter.is_cuda
            and parameter.dtype == torch.bfloat16
            and self.config.batch_size == 16
            and self.config.seq_len == 256
            and self.config.d_model == 1024
            and self.config.num_heads == 8
            and self.config.ffn_dim == 4096
            and self.config.num_layers == 6
            and not self.config.causal
        )
        launch_cuda_graph_eligible = (
            parameter.is_cuda
            and parameter.dtype == torch.float16
            and self.config.batch_size == 1
            and self.config.seq_len == 64
            and self.config.d_model == 256
            and self.config.num_heads == 8
            and self.config.ffn_dim == 1024
            and self.config.num_layers == 4
            and not self.config.causal
            and not self.training
        )
        triton_residual_eligible = (
            TRITON_RESIDUAL_AVAILABLE
            and parameter.is_cuda
            and parameter.dtype in (torch.float16, torch.bfloat16, torch.float32)
        )
        effective_policy = self.dispatch_policy or self.requested_policy
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

        if self.requested_attention == "triton_softmax" and triton_softmax_eligible:
            resolved_attention = "explicit_qk_triton_softmax_pv"
        elif use_fp32_sdpa:
            resolved_attention = "fp32_sdpa"
        elif self.requested_attention == "triton_pv" and triton_pv_eligible:
            resolved_attention = (
                "explicit_qk_triton_preprocess_native_softmax_triton_pv"
            )
        elif (
            self.requested_attention == "triton_preprocess"
            or (self.requested_attention == "auto" and self.config.seq_len == 2048)
        ) and triton_preprocess_eligible:
            resolved_attention = "explicit_qk_triton_preprocess_native_softmax_pv"
        elif (
            self.requested_attention in ("auto", "explicit", "triton_preprocess")
            and parameter.dtype in (torch.float16, torch.bfloat16)
            and self.config.seq_len <= 512
        ):
            resolved_attention = "explicit_qk_native_fp32_dtype_softmax_pv"
        else:
            resolved_attention = "explicit_reference_order"
            if self.requested_attention == "fp32_sdpa":
                fallback_reasons.append("fp32_sdpa_not_eligible")
            elif self.requested_attention == "triton_softmax":
                fallback_reasons.append("triton_attention_softmax_not_eligible")
            elif self.requested_attention == "triton_preprocess":
                fallback_reasons.append("triton_attention_preprocess_not_eligible")
            elif self.requested_attention == "triton_pv":
                fallback_reasons.append("triton_attention_pv_not_eligible")
        if self.use_triton_residual and not triton_residual_eligible:
            fallback_reasons.append("triton_residual_fusion_not_eligible")
        if (
            self.requested_ffn == "cublaslt_tanh_gelu_epilogue"
            and not wide_epilogue_eligible
        ):
            fallback_reasons.append("wide_ffn_epilogue_not_eligible")
        if self.use_cuda_graph and not launch_cuda_graph_eligible:
            fallback_reasons.append("launch_cuda_graph_not_eligible")

        if effective_policy == "triton" and not (
            triton_layout_eligible and triton_softmax_eligible
        ):
            selected_policy = (
                "triton_partial"
                if triton_layout_eligible or triton_softmax_eligible
                else "torch_fallback"
            )
        elif (
            (effective_policy == "preprocess" and not triton_preprocess_eligible)
            or (effective_policy == "long-pv" and not triton_pv_eligible)
            or (effective_policy == "wide-epilogue" and not wide_epilogue_eligible)
            or (effective_policy == "cuda-graph" and not launch_cuda_graph_eligible)
        ):
            selected_policy = "torch_fallback"
        else:
            selected_policy = effective_policy
        if effective_policy == "cuda-graph" and launch_cuda_graph_eligible:
            shape_route = "launch_fp16_eager_cuda_graph"
        elif (
            resolved_attention
            == "explicit_qk_triton_preprocess_native_softmax_triton_pv"
        ):
            shape_route = "long_fp16_fused_preprocess_and_pv_candidate"
        elif resolved_attention == "explicit_qk_triton_softmax_pv":
            shape_route = "triton_long_or_masked_attention"
        elif resolved_attention == "explicit_qk_triton_preprocess_native_softmax_pv":
            shape_route = "long_fp16_fused_preprocess_native_softmax"
        elif resolved_attention == "explicit_qk_native_fp32_dtype_softmax_pv":
            shape_route = "low_precision_native_dtype_softmax"
        elif use_fp32_sdpa:
            shape_route = "short_fp32_sdpa"
        elif self.config.seq_len >= 512:
            shape_route = "long_or_masked_reference_attention"
        elif parameter.dtype == torch.bfloat16 and self.config.d_model >= 1024:
            shape_route = "wide_bf16_reference_attention"
        else:
            shape_route = "general_reference_attention"
        resolved_ffn = (
            "cublaslt_tanh_gelu_epilogue"
            if self.requested_ffn == "cublaslt_tanh_gelu_epilogue"
            and wide_epilogue_eligible
            else "torch_exact_gelu"
        )
        return {
            "requested_policy": self.requested_policy,
            "selected_policy": selected_policy,
            "dispatch_source": (
                str(self._dispatcher.path) if self._dispatcher is not None else None
            ),
            "dispatch_policy": self.dispatch_policy,
            "runtime_wrapper": (
                "solution_eager_cuda_graph"
                if self.use_cuda_graph and launch_cuda_graph_eligible
                else "eager"
            ),
            "qkv_projection": "packed",
            "requested_qkv_layout": self.requested_qkv_layout,
            "resolved_qkv_layout": resolved_qkv_layout,
            "qkv_head_layout": resolved_qkv_layout,
            "requested_attention": self.requested_attention,
            "resolved_attention": resolved_attention,
            "attention_policy": resolved_attention,
            "selected_attention_backend": (
                "triton_preprocess_native_softmax_triton_pv"
                if resolved_attention
                == "explicit_qk_triton_preprocess_native_softmax_triton_pv"
                else "triton_softmax"
                if resolved_attention == "explicit_qk_triton_softmax_pv"
                else "triton_preprocess_native_softmax"
                if resolved_attention
                == "explicit_qk_triton_preprocess_native_softmax_pv"
                else "native_fp32_dtype_softmax"
                if resolved_attention == "explicit_qk_native_fp32_dtype_softmax_pv"
                else "auto"
                if use_fp32_sdpa
                else "explicit"
            ),
            "attention_candidate_status": (
                "experimental_requires_correctness_gate"
                if self.requested_attention
                in {"triton_softmax", "triton_preprocess", "triton_pv"}
                else "validated_route"
            ),
            "requested_ffn": self.requested_ffn,
            "resolved_ffn": resolved_ffn,
            "ffn_candidate_status": (
                "experimental_requires_correctness_gate"
                if self.requested_ffn == "cublaslt_tanh_gelu_epilogue"
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
            "token_mask_preprocessing": (
                "triton_direct_causal_and_key_mask"
                if resolved_attention
                in {
                    "explicit_qk_triton_preprocess_native_softmax_pv",
                    "explicit_qk_triton_preprocess_native_softmax_triton_pv",
                    "explicit_qk_triton_softmax_pv",
                }
                else "shared_causal_padding_union"
                if self.config.causal
                else "shared_broadcast_views"
            ),
        }

    def _uses_direct_score_masking(
        self,
        value: torch.Tensor,
        valid_token_mask: torch.Tensor | None,
    ) -> bool:
        """Predict routes that consume causal and key masks inside Triton."""

        sequence_length = value.shape[1]
        head_dim = self.config.d_model // self.config.num_heads
        mask_compatible = valid_token_mask is None or (
            valid_token_mask.is_cuda
            and valid_token_mask.device == value.device
            and valid_token_mask.dtype == torch.bool
            and valid_token_mask.is_contiguous()
            and valid_token_mask.shape == value.shape[:2]
        )
        if not mask_compatible or not value.is_cuda or value.dtype != torch.float16:
            return False
        use_auto_preprocess = (
            self.requested_attention == "auto"
            and sequence_length == 2048
            and head_dim == 64
        )
        use_explicit_preprocess = self.requested_attention in {
            "triton_preprocess",
            "triton_pv",
        } and (sequence_length, head_dim) in {(64, 32), (2048, 64)}
        if TRITON_ATTENTION_PREPROCESS_AVAILABLE and (
            use_auto_preprocess or use_explicit_preprocess
        ):
            return True
        return (
            self.requested_attention == "triton_softmax"
            and TRITON_ATTENTION_SOFTMAX_AVAILABLE
            and sequence_length in (512, 2048)
            and head_dim == 64
        )

    def _can_use_launch_cuda_graph(
        self,
        value: torch.Tensor,
        valid_token_mask: torch.Tensor | None,
    ) -> bool:
        mask_compatible = valid_token_mask is None or (
            valid_token_mask.is_cuda
            and valid_token_mask.device == value.device
            and valid_token_mask.dtype == torch.bool
            and valid_token_mask.is_contiguous()
            and valid_token_mask.shape == value.shape[:2]
        )
        return (
            self.use_cuda_graph
            and value.is_cuda
            and value.dtype == torch.float16
            and value.is_contiguous()
            and value.shape == (1, 64, 256)
            and self.config.num_heads == 8
            and self.config.ffn_dim == 1024
            and self.config.num_layers == 4
            and not self.config.causal
            and not self.training
            and not torch.is_grad_enabled()
            and mask_compatible
        )

    def _forward_eager(
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

        causal = self._causal_mask is not None
        direct_score_masking = self._uses_direct_score_masking(x, valid_token_mask)
        score_mask = None if direct_score_masking else invalid_key_mask
        if self._causal_mask is not None and not direct_score_masking:
            sequence_length = x.shape[1]
            causal_score_mask = self._causal_mask[
                None,
                None,
                :sequence_length,
                :sequence_length,
            ]
            score_mask = (
                causal_score_mask
                if score_mask is None
                else score_mask | causal_score_mask
            )

        for layer in self.layers:
            x = layer(
                x,
                valid_token_mask,
                score_mask,
                invalid_query_mask,
                causal,
            )
        x = self.final_norm(x)
        if invalid_query_mask is not None:
            x.masked_fill_(invalid_query_mask, 0)
        return x

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if torch.compiler.is_compiling():
            return self._forward_eager(x, valid_token_mask)
        self._resolve_dispatch(x)
        if self._can_use_launch_cuda_graph(x, valid_token_mask):
            if self._cuda_graph_replay is None:
                self._cuda_graph_replay = CudaGraphReplay()
            return self._cuda_graph_replay.run(
                self._forward_eager,
                x,
                valid_token_mask,
            )
        return self._forward_eager(x, valid_token_mask)


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
