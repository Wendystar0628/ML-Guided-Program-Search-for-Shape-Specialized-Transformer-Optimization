"""Pure execution-plan resolution shared by reporting and model execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .kernels import (
    supports_s512_native_half_softmax,
    supports_triton_attention_preprocess,
    supports_triton_attention_softmax,
    supports_triton_online_attention,
    supports_triton_qkv_layout,
    supports_triton_residual,
    supports_wide_exact_gelu,
)
from .policies import ExecutionComponent, PolicySpec

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
    required_components: tuple[str, ...]
    resolved_components: tuple[str, ...]
    missing_components: tuple[str, ...]
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
            "required_components": list(self.required_components),
            "resolved_components": list(self.resolved_components),
            "missing_components": list(self.missing_components),
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
                if bool(
                    {
                        ExecutionComponent.TRITON_ATTENTION_SOFTMAX.value,
                        ExecutionComponent.TRITON_ATTENTION_PREPROCESS.value,
                        ExecutionComponent.S512_NATIVE_HALF_SOFTMAX.value,
                        ExecutionComponent.TAIL_ONLINE_ATTENTION.value,
                    }
                    & set(self.required_components)
                )
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


@dataclass(frozen=True, slots=True)
class _ExecutionCapabilities:
    fp32_sdpa: bool
    triton_layout: bool
    triton_softmax: bool
    s512_native_softmax: bool
    triton_preprocess: bool
    triton_online: bool
    wide_inplace: bool
    triton_residual: bool
    cuda_graph: bool


def _resolve_capabilities(
    context: ExecutionContext,
    spec: PolicySpec,
) -> _ExecutionCapabilities:
    """Collect the static side of the exact tensor guards used by kernels."""

    triton_softmax = supports_triton_attention_softmax(
        device_type=context.device.type,
        dtype=context.dtype,
        sequence_length=context.sequence_length,
        head_dim=context.head_dim,
        mask_compatible=context.mask_compatible,
    )
    triton_preprocess = supports_triton_attention_preprocess(
        device_type=context.device.type,
        dtype=context.dtype,
        sequence_length=context.sequence_length,
        head_dim=context.head_dim,
        mask_compatible=context.mask_compatible,
    )
    triton_online = supports_triton_online_attention(
        device=context.device,
        dtype=context.dtype,
        batch_size=context.batch_size,
        num_heads=context.num_heads,
        sequence_length=context.sequence_length,
        head_dim=context.head_dim,
        grad_or_training=context.training or context.grad_enabled,
        mask_compatible=context.mask_compatible,
    )
    workload_shape = (
        context.batch_size,
        context.sequence_length,
        context.d_model,
        context.num_heads,
        context.ffn_dim,
        context.num_layers,
    )
    cuda_graph = (
        spec.cuda_graph_shape is not None
        and workload_shape == spec.cuda_graph_shape
        and context.input_contiguous
        and context.mask_compatible
        and not context.causal
        and not context.training
        and not context.grad_enabled
        and context.dtype == torch.float16
        and context.device.type == "cuda"
    )
    return _ExecutionCapabilities(
        fp32_sdpa=(
            context.device.type == "cuda"
            and context.dtype == torch.float32
            and not context.causal
            and context.sequence_length <= 128
        ),
        triton_layout=supports_triton_qkv_layout(
            device_type=context.device.type,
            dtype=context.dtype,
            model_width=context.d_model,
            num_heads=context.num_heads,
        ),
        triton_softmax=triton_softmax,
        s512_native_softmax=supports_s512_native_half_softmax(
            device_type=context.device.type,
            dtype=context.dtype,
            batch_size=context.batch_size,
            num_heads=context.num_heads,
            sequence_length=context.sequence_length,
            head_dim=context.head_dim,
            mask_compatible=context.mask_compatible,
        ),
        triton_preprocess=triton_preprocess,
        triton_online=triton_online,
        wide_inplace=supports_wide_exact_gelu(
            device_type=context.device.type,
            dtype=context.dtype,
            input_shape=(
                context.batch_size,
                context.sequence_length,
                context.d_model,
            ),
            weight_shape=(context.ffn_dim, context.d_model),
            bias_shape=(context.ffn_dim,),
            grad_enabled=context.training or context.grad_enabled,
        ),
        triton_residual=supports_triton_residual(
            device_type=context.device.type,
            dtype=context.dtype,
            has_valid_token_mask=context.has_valid_token_mask,
            mask_compatible=context.mask_compatible,
        ),
        cuda_graph=cuda_graph,
    )


def _resolve_qkv_layout(
    spec: PolicySpec,
    capabilities: _ExecutionCapabilities,
) -> str:
    if spec.qkv_layout == "triton":
        if capabilities.triton_layout:
            return "triton_single_pass"
        return "view_fallback"
    if spec.qkv_layout == "torch_contiguous":
        return "torch_three_contiguous_copies"
    return "torch_zero_copy_view"


def _resolve_attention(
    requested_attention: str,
    context: ExecutionContext,
    capabilities: _ExecutionCapabilities,
) -> str:
    use_fp32_sdpa = (
        requested_attention in {"auto", "fp32_sdpa"} and capabilities.fp32_sdpa
    )
    if requested_attention == "triton_online" and capabilities.triton_online:
        return "triton_two_pass_online_attention"
    if (
        requested_attention == "s512_native_half_softmax"
        and capabilities.s512_native_softmax
    ):
        return "explicit_qk_triton_scale_mask_native_half_softmax_pv"
    if requested_attention == "triton_softmax" and capabilities.triton_softmax:
        return "explicit_qk_triton_softmax_pv"
    if use_fp32_sdpa:
        return "fp32_sdpa"
    if (
        requested_attention == "triton_preprocess"
        or (requested_attention == "auto" and context.sequence_length == 2048)
    ) and capabilities.triton_preprocess:
        return "explicit_qk_triton_preprocess_native_softmax_pv"
    if (
        requested_attention in {"auto", "explicit", "triton_preprocess"}
        and context.dtype in (torch.float16, torch.bfloat16)
        and context.sequence_length <= 512
    ):
        return "explicit_qk_native_fp32_dtype_softmax_pv"
    return "explicit_reference_order"


_ATTENTION_COMPONENTS = {
    "explicit_qk_triton_softmax_pv": (
        ExecutionComponent.TRITON_ATTENTION_SOFTMAX
    ),
    "explicit_qk_triton_preprocess_native_softmax_pv": (
        ExecutionComponent.TRITON_ATTENTION_PREPROCESS
    ),
    "explicit_qk_triton_scale_mask_native_half_softmax_pv": (
        ExecutionComponent.S512_NATIVE_HALF_SOFTMAX
    ),
}

_COMPONENT_FALLBACK_REASONS = {
    ExecutionComponent.TRITON_QKV_LAYOUT: (
        "triton_qkv_layout_not_available_or_compatible"
    ),
    ExecutionComponent.TRITON_ATTENTION_SOFTMAX: (
        "triton_attention_softmax_not_eligible"
    ),
    ExecutionComponent.TRITON_ATTENTION_PREPROCESS: (
        "triton_attention_preprocess_not_eligible"
    ),
    ExecutionComponent.S512_NATIVE_HALF_SOFTMAX: (
        "s512_native_half_softmax_not_eligible"
    ),
    ExecutionComponent.TAIL_ONLINE_ATTENTION: (
        "triton_tail_online_attention_not_eligible"
    ),
    ExecutionComponent.WIDE_INPLACE_FFN: "wide_ffn_epilogue_not_eligible",
    ExecutionComponent.PACKED_FFN: "packed_ffn_requires_valid_token_mask",
    ExecutionComponent.TRITON_RESIDUAL: (
        "triton_residual_fusion_not_eligible"
    ),
    ExecutionComponent.CUDA_GRAPH: "cuda_graph_policy_not_eligible",
}


def _resolved_components(
    *,
    qkv_layout: str,
    layer_attention: list[str],
    resolved_ffn: str,
    use_packed_ffn: bool,
    use_triton_residual: bool,
    use_cuda_graph: bool,
) -> frozenset[ExecutionComponent]:
    """Describe specialized components in the exact plan being returned."""

    components: set[ExecutionComponent] = set()
    if qkv_layout == "triton_single_pass":
        components.add(ExecutionComponent.TRITON_QKV_LAYOUT)
    components.update(
        component
        for attention in layer_attention
        if (component := _ATTENTION_COMPONENTS.get(attention)) is not None
    )
    if "triton_two_pass_online_attention" in layer_attention:
        components.add(ExecutionComponent.TAIL_ONLINE_ATTENTION)
    if resolved_ffn == "torch_inplace_exact_gelu":
        components.add(ExecutionComponent.WIDE_INPLACE_FFN)
    if use_packed_ffn:
        components.add(ExecutionComponent.PACKED_FFN)
    if use_triton_residual:
        components.add(ExecutionComponent.TRITON_RESIDUAL)
    if use_cuda_graph:
        components.add(ExecutionComponent.CUDA_GRAPH)
    return frozenset(components)


def _selected_policy(
    spec: PolicySpec,
    resolved_components: frozenset[ExecutionComponent],
) -> str:
    missing = spec.required_components - resolved_components
    if not missing:
        return spec.policy_id
    if spec.allow_partial_application and (
        spec.required_components & resolved_components
    ):
        return f"{spec.policy_id}_partial"
    return "torch_fallback"


def _fallback_reasons(
    missing_components: frozenset[ExecutionComponent],
    *,
    partial_application_disallowed: bool,
) -> tuple[str, ...]:
    reasons = [
        _COMPONENT_FALLBACK_REASONS[component]
        for component in sorted(missing_components, key=str)
    ]
    if partial_application_disallowed:
        reasons.append("policy_requires_complete_application")
    return tuple(reasons)


def _shape_route(
    spec: PolicySpec,
    resolved_attention: str,
    context: ExecutionContext,
    use_cuda_graph: bool,
) -> str:
    if use_cuda_graph:
        assert spec.cuda_graph_route is not None
        return spec.cuda_graph_route
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
    capabilities = _resolve_capabilities(context, spec)
    resolved_qkv_layout = _resolve_qkv_layout(spec, capabilities)
    base_attention = _resolve_attention(
        spec.attention,
        context,
        capabilities,
    )
    layer_attention = [base_attention] * context.num_layers
    resolved_attention = base_attention
    if capabilities.triton_online and spec.tail_attention == "triton_online":
        layer_attention[-1] = "triton_two_pass_online_attention"
        resolved_attention = "three_explicit_layers_tail_online_attention"

    resolved_ffn = "torch_exact_gelu"
    if capabilities.wide_inplace and spec.ffn == "inplace_exact_gelu":
        resolved_ffn = "torch_inplace_exact_gelu"
    use_packed_ffn = spec.use_packed_ffn and context.has_valid_token_mask
    use_triton_residual = spec.use_triton_residual and capabilities.triton_residual
    use_cuda_graph = spec.use_cuda_graph and capabilities.cuda_graph
    resolved_components = _resolved_components(
        qkv_layout=resolved_qkv_layout,
        layer_attention=layer_attention,
        resolved_ffn=resolved_ffn,
        use_packed_ffn=use_packed_ffn,
        use_triton_residual=use_triton_residual,
        use_cuda_graph=use_cuda_graph,
    )
    unavailable_components = spec.required_components - resolved_components
    partial_application_disallowed = bool(
        unavailable_components
        and spec.required_components & resolved_components
        and not spec.allow_partial_application
    )
    if partial_application_disallowed:
        # A policy that requires all of its specialized pieces falls back as a
        # unit. This avoids reporting one policy while silently running a
        # hybrid that was never registered or measured.
        resolved_qkv_layout = "torch_zero_copy_view"
        base_attention = _resolve_attention("auto", context, capabilities)
        layer_attention = [base_attention] * context.num_layers
        resolved_attention = base_attention
        resolved_ffn = "torch_exact_gelu"
        use_packed_ffn = False
        use_triton_residual = False
        use_cuda_graph = False
        resolved_components = _resolved_components(
            qkv_layout=resolved_qkv_layout,
            layer_attention=layer_attention,
            resolved_ffn=resolved_ffn,
            use_packed_ffn=use_packed_ffn,
            use_triton_residual=use_triton_residual,
            use_cuda_graph=use_cuda_graph,
        )

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
    missing_components = spec.required_components - resolved_components
    fallback_reasons = _fallback_reasons(
        unavailable_components,
        partial_application_disallowed=partial_application_disallowed,
    )

    return ExecutionPlan(
        requested_policy=requested_policy,
        effective_policy=effective_policy,
        selected_policy=_selected_policy(spec, resolved_components),
        required_components=tuple(sorted(map(str, spec.required_components))),
        resolved_components=tuple(sorted(map(str, resolved_components))),
        missing_components=tuple(sorted(map(str, missing_components))),
        requested_qkv_layout=spec.qkv_layout,
        resolved_qkv_layout=resolved_qkv_layout,
        requested_attention=spec.attention,
        resolved_attention=resolved_attention,
        selected_attention_backend=_attention_backend(resolved_attention),
        requested_ffn=spec.ffn,
        resolved_ffn=resolved_ffn,
        shape_route=_shape_route(
            spec,
            resolved_attention,
            context,
            use_cuda_graph,
        ),
        use_packed_ffn=use_packed_ffn,
        use_triton_residual=use_triton_residual,
        use_cuda_graph=use_cuda_graph,
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
