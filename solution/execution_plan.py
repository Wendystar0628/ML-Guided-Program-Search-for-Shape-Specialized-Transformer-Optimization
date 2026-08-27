"""Pure execution-plan resolution shared by reporting and model execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .kernels import (
    TRITON_ATTENTION_PREPROCESS_AVAILABLE,
    TRITON_ATTENTION_SOFTMAX_AVAILABLE,
    TRITON_ONLINE_ATTENTION_AVAILABLE,
    TRITON_QKV_LAYOUT_AVAILABLE,
    TRITON_RESIDUAL_AVAILABLE,
)
from .policies import PolicySpec

_DIRECT_MASK_ATTENTION_PATHS = frozenset(
    {
        "explicit_qk_triton_preprocess_native_softmax_pv",
        "explicit_qk_triton_softmax_pv",
        "explicit_qk_triton_scale_mask_native_half_softmax_pv",
        "triton_two_pass_online_attention",
    }
)


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Runtime facts needed to resolve a bounded execution policy."""

    batch_size: int
    sequence_length: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool
    device: torch.device
    dtype: torch.dtype
    training: bool
    grad_enabled: bool
    input_contiguous: bool
    has_valid_token_mask: bool
    mask_compatible: bool

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


@dataclass(frozen=True, slots=True)
class LayerExecutionPlan:
    """Exact implementations selected for one Transformer layer."""

    qkv_layout: str
    attention: str
    ffn: str
    use_packed_ffn: bool
    use_triton_residual: bool


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """One immutable resolution of policy intent against runtime facts."""

    requested_policy: str
    effective_policy: str
    selected_policy: str
    requested_qkv_layout: str
    resolved_qkv_layout: str
    requested_attention: str
    resolved_attention: str
    selected_attention_backend: str
    requested_ffn: str
    resolved_ffn: str
    shape_route: str
    use_packed_ffn: bool
    use_triton_residual: bool
    use_cuda_graph: bool
    direct_score_masking: bool
    layers: tuple[LayerExecutionPlan, ...]
    fallback_reasons: tuple[str, ...]

    def describe(
        self,
        *,
        dispatch_source: str | None,
        dispatch_table_sha256: str | None,
        dispatch_policy: str | None,
        route_origin: str | None,
        causal: bool,
    ) -> dict[str, Any]:
        """Serialize the same immutable plan consumed by forward execution."""

        return {
            "requested_policy": self.requested_policy,
            "selected_policy": self.selected_policy,
            "dispatch_source": dispatch_source,
            "dispatch_table_sha256": dispatch_table_sha256,
            "dispatch_policy": dispatch_policy,
            "route_origin": route_origin,
            "runtime_wrapper": (
                "solution_eager_cuda_graph" if self.use_cuda_graph else "eager"
            ),
            "qkv_projection": "packed",
            "requested_qkv_layout": self.requested_qkv_layout,
            "resolved_qkv_layout": self.resolved_qkv_layout,
            "qkv_head_layout": self.resolved_qkv_layout,
            "requested_attention": self.requested_attention,
            "resolved_attention": self.resolved_attention,
            "attention_policy": self.resolved_attention,
            "selected_attention_backend": self.selected_attention_backend,
            "layer_attention": [layer.attention for layer in self.layers],
            "attention_candidate_status": (
                "experimental_requires_correctness_gate"
                if self.effective_policy == "long-tail-online"
                or self.requested_attention
                in {
                    "triton_softmax",
                    "s512_native_half_softmax",
                    "triton_preprocess",
                    "triton_online",
                }
                else "validated_route"
            ),
            "requested_ffn": self.requested_ffn,
            "resolved_ffn": self.resolved_ffn,
            "ffn_candidate_status": (
                "experimental_requires_correctness_gate"
                if self.requested_ffn == "inplace_exact_gelu"
                else "validated_route"
            ),
            "shape_route": self.shape_route,
            "block_fusion": (
                "triton_residual_add_padding_when_masked"
                if self.use_triton_residual
                else "torch_residual_fallback"
                if "triton_residual_fusion_not_eligible" in self.fallback_reasons
                else "none"
            ),
            "padding_route": (
                "packed_valid_token_ffn"
                if self.use_packed_ffn
                else "full_ffn_with_fused_padding_residual"
                if self.use_triton_residual
                else "shared_mask_only"
            ),
            "fallback_reason": list(self.fallback_reasons) or None,
            "causal_mask": "shared_buffer" if causal else "none",
            "token_mask_preprocessing": (
                "triton_direct_causal_and_key_mask"
                if self.direct_score_masking
                else "shared_causal_padding_union"
                if causal
                else "shared_broadcast_views"
            ),
        }


def _supports_online_attention(context: ExecutionContext) -> bool:
    if not (
        TRITON_ONLINE_ATTENTION_AVAILABLE
        and context.device.type == "cuda"
        and context.dtype == torch.float16
        and context.batch_size == 1
        and context.sequence_length == 2048
        and context.d_model == 512
        and context.num_heads == 8
        and context.head_dim == 64
        and not context.training
        and not context.grad_enabled
        and context.mask_compatible
    ):
        return False
    if torch.version.cuda is None:
        return False
    try:
        return torch.cuda.get_device_capability(context.device)[0] >= 8
    except (AssertionError, RuntimeError):
        return False


@dataclass(frozen=True, slots=True)
class _ExecutionCapabilities:
    fp32_sdpa: bool
    triton_layout: bool
    triton_softmax: bool
    s512_native_softmax: bool
    triton_preprocess: bool
    triton_online: bool
    tail_online: bool
    wide_inplace: bool
    triton_residual: bool
    cuda_graph: bool


def _resolve_capabilities(
    context: ExecutionContext,
    effective_policy: str,
) -> _ExecutionCapabilities:
    """Collect the static side of the exact tensor guards used by kernels."""

    head_dim = context.head_dim
    cuda = context.device.type == "cuda"
    mask_compatible = context.mask_compatible
    triton_softmax = (
        TRITON_ATTENTION_SOFTMAX_AVAILABLE
        and cuda
        and context.dtype == torch.float16
        and context.sequence_length in (512, 2048)
        and head_dim == 64
        and mask_compatible
    )
    triton_preprocess = (
        TRITON_ATTENTION_PREPROCESS_AVAILABLE
        and cuda
        and context.dtype == torch.float16
        and (context.sequence_length, head_dim) in {(64, 32), (2048, 64)}
        and mask_compatible
    )
    triton_online = _supports_online_attention(context)
    tail_online = (
        effective_policy == "long-tail-online"
        and triton_online
        and context.ffn_dim == 2048
        and context.num_layers == 4
    )
    launch_graph = (
        context.input_contiguous
        and mask_compatible
        and not context.causal
        and not context.training
        and not context.grad_enabled
        and context.dtype == torch.float16
        and cuda
        and (
            context.batch_size,
            context.sequence_length,
            context.d_model,
            context.num_heads,
            context.ffn_dim,
            context.num_layers,
        )
        == (1, 64, 256, 8, 1024, 4)
    )
    balanced_graph = (
        context.input_contiguous
        and mask_compatible
        and not context.causal
        and not context.training
        and not context.grad_enabled
        and context.dtype == torch.float16
        and cuda
        and (
            context.batch_size,
            context.sequence_length,
            context.d_model,
            context.num_heads,
            context.ffn_dim,
            context.num_layers,
        )
        == (8, 128, 512, 8, 2048, 6)
    )
    return _ExecutionCapabilities(
        fp32_sdpa=(
            cuda
            and context.dtype == torch.float32
            and not context.causal
            and context.sequence_length <= 128
        ),
        triton_layout=(
            TRITON_QKV_LAYOUT_AVAILABLE
            and cuda
            and context.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and 16 <= head_dim <= 128
            and head_dim & (head_dim - 1) == 0
        ),
        triton_softmax=triton_softmax,
        s512_native_softmax=(
            triton_softmax
            and context.batch_size == 8
            and context.sequence_length == 512
            and context.d_model == 512
            and context.num_heads == 8
        ),
        triton_preprocess=triton_preprocess,
        triton_online=triton_online,
        tail_online=tail_online,
        wide_inplace=(
            cuda
            and context.dtype == torch.bfloat16
            and context.batch_size == 16
            and context.sequence_length == 256
            and context.d_model == 1024
            and context.num_heads == 8
            and context.ffn_dim == 4096
            and context.num_layers == 6
            and not context.causal
            and not context.training
            and not context.grad_enabled
        ),
        triton_residual=(
            TRITON_RESIDUAL_AVAILABLE
            and cuda
            and context.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and context.has_valid_token_mask
            and mask_compatible
        ),
        cuda_graph=(effective_policy == "cuda-graph" and launch_graph)
        or (effective_policy == "balanced-cuda-graph" and balanced_graph),
    )


def _resolve_qkv_layout(
    spec: PolicySpec,
    capabilities: _ExecutionCapabilities,
) -> tuple[str, str | None]:
    if spec.qkv_layout == "triton":
        if capabilities.triton_layout:
            return "triton_single_pass", None
        return "view_fallback", "triton_qkv_layout_not_available_or_compatible"
    if spec.qkv_layout == "torch_contiguous":
        return "torch_three_contiguous_copies", None
    return "torch_zero_copy_view", None


def _resolve_attention(
    requested_attention: str,
    context: ExecutionContext,
    capabilities: _ExecutionCapabilities,
) -> tuple[str, str | None]:
    use_fp32_sdpa = (
        requested_attention in {"auto", "fp32_sdpa"} and capabilities.fp32_sdpa
    )
    if requested_attention == "triton_online" and capabilities.triton_online:
        return "triton_two_pass_online_attention", None
    if (
        requested_attention == "s512_native_half_softmax"
        and capabilities.s512_native_softmax
    ):
        return "explicit_qk_triton_scale_mask_native_half_softmax_pv", None
    if requested_attention == "triton_softmax" and capabilities.triton_softmax:
        return "explicit_qk_triton_softmax_pv", None
    if use_fp32_sdpa:
        return "fp32_sdpa", None
    if (
        requested_attention == "triton_preprocess"
        or (requested_attention == "auto" and context.sequence_length == 2048)
    ) and capabilities.triton_preprocess:
        return "explicit_qk_triton_preprocess_native_softmax_pv", None
    if (
        requested_attention in {"auto", "explicit", "triton_preprocess"}
        and context.dtype in (torch.float16, torch.bfloat16)
        and context.sequence_length <= 512
    ):
        return "explicit_qk_native_fp32_dtype_softmax_pv", None
    reason = {
        "fp32_sdpa": "fp32_sdpa_not_eligible",
        "s512_native_half_softmax": "s512_native_half_softmax_not_eligible",
        "triton_softmax": "triton_attention_softmax_not_eligible",
        "triton_preprocess": "triton_attention_preprocess_not_eligible",
        "triton_online": "triton_online_attention_not_eligible",
    }.get(requested_attention)
    return "explicit_reference_order", reason


def _fallback_reasons(
    spec: PolicySpec,
    effective_policy: str,
    context: ExecutionContext,
    capabilities: _ExecutionCapabilities,
    qkv_reason: str | None,
    attention_reason: str | None,
) -> tuple[str, ...]:
    reasons = [
        reason for reason in (qkv_reason, attention_reason) if reason is not None
    ]
    if spec.use_packed_ffn and not context.has_valid_token_mask:
        reasons.append("packed_ffn_requires_valid_token_mask")
    if spec.use_triton_residual and not capabilities.triton_residual:
        reasons.append("triton_residual_fusion_not_eligible")
    if effective_policy == "long-tail-online" and not capabilities.tail_online:
        reasons.append("triton_tail_online_attention_not_eligible")
    if spec.ffn == "inplace_exact_gelu" and not capabilities.wide_inplace:
        reasons.append("wide_ffn_epilogue_not_eligible")
    if spec.use_cuda_graph and not capabilities.cuda_graph:
        reasons.append("cuda_graph_policy_not_eligible")
    return tuple(reasons)


def _resolve_selected_policy(
    effective_policy: str,
    capabilities: _ExecutionCapabilities,
) -> str:
    if effective_policy == "triton" and not (
        capabilities.triton_layout and capabilities.triton_softmax
    ):
        if capabilities.triton_layout or capabilities.triton_softmax:
            return "triton_partial"
        return "torch_fallback"
    specialized_policy_eligible = {
        "preprocess": capabilities.triton_preprocess,
        "s512-native-softmax": capabilities.s512_native_softmax,
        "long-tail-online": capabilities.tail_online,
        "wide-triton-inplace": (
            capabilities.wide_inplace and capabilities.triton_layout
        ),
        "cuda-graph": capabilities.cuda_graph,
        "balanced-cuda-graph": capabilities.cuda_graph,
    }
    if (
        effective_policy in specialized_policy_eligible
        and not specialized_policy_eligible[effective_policy]
    ):
        return "torch_fallback"
    return effective_policy


def _shape_route(
    effective_policy: str,
    resolved_attention: str,
    context: ExecutionContext,
    capabilities: _ExecutionCapabilities,
) -> str:
    if effective_policy == "cuda-graph" and capabilities.cuda_graph:
        return "launch_fp16_eager_cuda_graph"
    if effective_policy == "balanced-cuda-graph" and capabilities.cuda_graph:
        return "balanced_fp16_eager_cuda_graph"
    attention_routes = {
        "three_explicit_layers_tail_online_attention": (
            "long_fp16_tail_layer_online_attention"
        ),
        "triton_two_pass_online_attention": (
            "long_fp16_all_layers_online_attention_candidate"
        ),
        "explicit_qk_triton_scale_mask_native_half_softmax_pv": (
            "s512_fp16_scale_mask_native_half_softmax"
        ),
        "explicit_qk_triton_softmax_pv": "triton_long_or_masked_attention",
        "explicit_qk_triton_preprocess_native_softmax_pv": (
            "long_fp16_fused_preprocess_native_softmax"
        ),
        "explicit_qk_native_fp32_dtype_softmax_pv": (
            "low_precision_native_dtype_softmax"
        ),
        "fp32_sdpa": "short_fp32_sdpa",
    }
    if resolved_attention in attention_routes:
        return attention_routes[resolved_attention]
    if context.sequence_length >= 512:
        return "long_or_masked_reference_attention"
    if context.dtype == torch.bfloat16 and context.d_model >= 1024:
        return "wide_bf16_reference_attention"
    return "general_reference_attention"


def _attention_backend(resolved_attention: str) -> str:
    return {
        "three_explicit_layers_tail_online_attention": "triton_tail_layer_online",
        "triton_two_pass_online_attention": "triton_two_pass_online",
        "explicit_qk_triton_scale_mask_native_half_softmax_pv": (
            "triton_scale_mask_native_half_softmax"
        ),
        "explicit_qk_triton_softmax_pv": "triton_softmax",
        "explicit_qk_triton_preprocess_native_softmax_pv": (
            "triton_preprocess_native_softmax"
        ),
        "explicit_qk_native_fp32_dtype_softmax_pv": "native_fp32_dtype_softmax",
        "fp32_sdpa": "auto",
    }.get(resolved_attention, "explicit")


def resolve_execution_plan(
    spec: PolicySpec,
    context: ExecutionContext,
    *,
    requested_policy: str,
    dispatch_policy: str | None,
) -> ExecutionPlan:
    """Resolve one policy without mutating the model or global runtime state."""

    effective_policy = dispatch_policy or requested_policy
    capabilities = _resolve_capabilities(context, effective_policy)
    resolved_qkv_layout, qkv_reason = _resolve_qkv_layout(spec, capabilities)
    base_attention, attention_reason = _resolve_attention(
        spec.attention,
        context,
        capabilities,
    )
    layer_attention = [base_attention] * context.num_layers
    resolved_attention = base_attention
    if capabilities.tail_online and spec.tail_attention == "triton_online":
        layer_attention[-1] = "triton_two_pass_online_attention"
        resolved_attention = "three_explicit_layers_tail_online_attention"

    resolved_ffn = "torch_exact_gelu"
    if capabilities.wide_inplace and spec.ffn == "inplace_exact_gelu":
        resolved_ffn = "torch_inplace_exact_gelu"
    use_packed_ffn = spec.use_packed_ffn and context.has_valid_token_mask
    use_triton_residual = spec.use_triton_residual and capabilities.triton_residual
    layer_plans = tuple(
        LayerExecutionPlan(
            qkv_layout=resolved_qkv_layout,
            attention=attention,
            ffn=resolved_ffn,
            use_packed_ffn=use_packed_ffn,
            use_triton_residual=use_triton_residual,
        )
        for attention in layer_attention
    )
    fallback_reasons = _fallback_reasons(
        spec,
        effective_policy,
        context,
        capabilities,
        qkv_reason,
        attention_reason,
    )

    return ExecutionPlan(
        requested_policy=requested_policy,
        effective_policy=effective_policy,
        selected_policy=_resolve_selected_policy(effective_policy, capabilities),
        requested_qkv_layout=spec.qkv_layout,
        resolved_qkv_layout=resolved_qkv_layout,
        requested_attention=spec.attention,
        resolved_attention=resolved_attention,
        selected_attention_backend=_attention_backend(resolved_attention),
        requested_ffn=spec.ffn,
        resolved_ffn=resolved_ffn,
        shape_route=_shape_route(
            effective_policy,
            resolved_attention,
            context,
            capabilities,
        ),
        use_packed_ffn=use_packed_ffn,
        use_triton_residual=use_triton_residual,
        use_cuda_graph=spec.use_cuda_graph and capabilities.cuda_graph,
        direct_score_masking=all(
            layer.attention in _DIRECT_MASK_ATTENTION_PATHS for layer in layer_plans
        ),
        layers=layer_plans,
        fallback_reasons=fallback_reasons,
    )


__all__ = [
    "ExecutionContext",
    "ExecutionPlan",
    "LayerExecutionPlan",
    "resolve_execution_plan",
]
