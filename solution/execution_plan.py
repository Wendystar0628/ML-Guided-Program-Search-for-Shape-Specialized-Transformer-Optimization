"""Resolve one immutable execution plan for reporting and model execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from policy_registry import ExecutionComponent, PolicySpec


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Runtime facts that can change execution eligibility."""

    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    causal: bool
    device: torch.device
    dtype: torch.dtype
    training: bool
    grad_enabled: bool
    input_contiguous: bool
    has_valid_token_mask: bool
    mask_compatible: bool


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Single source of truth for every branch consumed by forward."""

    requested_policy: str
    selected_policy: str
    attention_backend: str
    runtime_wrapper: str
    residual_norm_backend: str
    has_valid_token_mask: bool
    required_components: tuple[str, ...]
    resolved_components: tuple[str, ...]
    missing_components: tuple[str, ...]
    fallback_reasons: tuple[str, ...]

    @property
    def use_cuda_graph(self) -> bool:
        return self.runtime_wrapper == "cuda_graph"

    def describe(
        self,
        *,
        dispatch_source: str | None,
        dispatch_table_sha256: str | None,
        dispatch_policy: str | None,
        route_origin: str | None,
        causal: bool,
    ) -> dict[str, Any]:
        """Serialize the same immutable plan consumed by forward."""

        if not causal:
            causal_mask = "none"
        elif self.attention_backend in {
            "causal_sdpa",
            "mixed_fp16_efficient",
        }:
            causal_mask = "implicit_sdpa"
        else:
            causal_mask = "query_block"
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
            "qkv_projection": "packed",
            "attention_backend": self.attention_backend,
            "runtime_wrapper": self.runtime_wrapper,
            "residual_norm_backend": self.residual_norm_backend,
            "causal_mask": causal_mask,
            "valid_token_mask": (
                "direct_key_mask" if self.has_valid_token_mask else "none"
            ),
            "fallback_reasons": list(self.fallback_reasons),
        }


@dataclass(frozen=True, slots=True)
class _Capabilities:
    causal_sdpa: bool
    cuda_graph: bool
    compiled_residual_layer_norm: bool
    mixed_fp16_efficient_attention: bool


def _capabilities(context: ExecutionContext) -> _Capabilities:
    inference = not context.training and not context.grad_enabled
    causal_sdpa = (
        context.causal
        and context.mask_compatible
        and context.d_model % context.num_heads == 0
        # Native BF16 SDPA exceeded the official elementwise comparator on the
        # development GPU, so BF16 stays on the reference-order safe path.
        and context.dtype in {torch.float16, torch.float32}
        and context.device.type in {"cpu", "cuda"}
    )
    head_dim = context.d_model // context.num_heads
    mixed_shape = (context.seq_len >= 1024 and head_dim == 32) or (
        context.batch_size in {64, 128}
        and context.seq_len == 128
        and context.d_model in {32, 128}
    )
    compiler = getattr(torch, "compile", None)
    mixed_runtime_available = False
    if context.device.type == "cuda":
        mixed_runtime_available = bool(
            torch.backends.cuda.mem_efficient_sdp_enabled()
            and torch.cuda.get_device_capability(context.device) >= (8, 0)
        )
    compiled_residual_base = (
        causal_sdpa
        and inference
        and context.device.type == "cuda"
        and context.dtype == torch.float32
        and context.input_contiguous
        and not context.has_valid_token_mask
        and callable(compiler)
        and not torch.compiler.is_compiling()
    )
    return _Capabilities(
        causal_sdpa=causal_sdpa,
        cuda_graph=(
            causal_sdpa
            and inference
            and context.device.type == "cuda"
            and context.input_contiguous
        ),
        compiled_residual_layer_norm=compiled_residual_base,
        mixed_fp16_efficient_attention=(
            inference
            and context.causal
            and context.device.type == "cuda"
            and context.dtype == torch.float32
            and context.input_contiguous
            and not context.has_valid_token_mask
            and context.mask_compatible
            and mixed_shape
            and mixed_runtime_available
        ),
    )


def _resolved_components(
    spec: PolicySpec,
    capabilities: _Capabilities,
) -> frozenset[ExecutionComponent]:
    resolved: set[ExecutionComponent] = set()
    if spec.attention == "causal_sdpa" and capabilities.causal_sdpa:
        resolved.add(ExecutionComponent.CAUSAL_SDPA)
    if spec.use_cuda_graph and capabilities.cuda_graph:
        resolved.add(ExecutionComponent.CUDA_GRAPH)
    if (
        spec.use_compiled_residual_layer_norm
        and capabilities.compiled_residual_layer_norm
    ):
        resolved.add(ExecutionComponent.COMPILED_RESIDUAL_LAYER_NORM)
    if (
        spec.attention == "mixed_fp16_efficient"
        and capabilities.mixed_fp16_efficient_attention
    ):
        resolved.add(ExecutionComponent.MIXED_FP16_EFFICIENT_ATTENTION)
    return frozenset(resolved)


def resolve_execution_plan(
    spec: PolicySpec,
    context: ExecutionContext,
    *,
    requested_policy: str,
) -> ExecutionPlan:
    """Resolve a policy atomically; unsupported policies become ``safe``."""

    capabilities = _capabilities(context)
    resolved = _resolved_components(spec, capabilities)
    missing = spec.required_components - resolved
    fallback_reasons = tuple(
        f"{component.value}_not_eligible"
        for component in sorted(missing, key=lambda item: item.value)
    )
    fully_applied = not missing
    if fully_applied:
        attention_backend = spec.attention
        selected_policy = spec.policy_id
    else:
        attention_backend = "safe_streaming"
        selected_policy = "safe"
        resolved = frozenset()

    use_cuda_graph = fully_applied and spec.use_cuda_graph
    use_compiled_residual_layer_norm = (
        fully_applied and spec.use_compiled_residual_layer_norm
    )
    return ExecutionPlan(
        requested_policy=requested_policy,
        selected_policy=selected_policy,
        attention_backend=attention_backend,
        runtime_wrapper="cuda_graph" if use_cuda_graph else "eager",
        residual_norm_backend=(
            "compiled_residual_layer_norm"
            if use_compiled_residual_layer_norm
            else "torch"
        ),
        has_valid_token_mask=context.has_valid_token_mask,
        required_components=tuple(
            sorted(component.value for component in spec.required_components)
        ),
        resolved_components=tuple(sorted(component.value for component in resolved)),
        missing_components=tuple(sorted(component.value for component in missing)),
        fallback_reasons=fallback_reasons,
    )


__all__ = ["ExecutionContext", "ExecutionPlan", "resolve_execution_plan"]
