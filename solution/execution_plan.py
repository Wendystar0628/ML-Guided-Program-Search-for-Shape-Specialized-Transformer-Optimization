"""Resolve one immutable execution plan for reporting and model execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from policy_registry import (
    ExecutionComponent,
    PolicySpec,
    ResidualNormBackend,
    RuntimeWrapper,
)

from .kernels import (
    triton_mixed_residual_layer_norm_available,
    triton_residual_layer_norm_available,
    triton_shape13_causal_attention_available,
)
from .shape_families import (
    is_compiled_forward_candidate_workload,
    is_mixed_fp16_core_efficient_runtime_family,
    is_shape06_batch_tiled_workload,
    is_shape13_triton_attention_workload,
    is_streamed_mixed_fp16_core_cudnn_slice,
)


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
    ffn_dim: int | None = None
    num_layers: int | None = None


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Single source of truth for every branch consumed by forward."""

    requested_policy: str
    selected_policy: str
    attention_backend: str
    attention_compute_dtype: str
    linear_backend: str
    linear_compute_dtype: str
    runtime_wrapper: str
    compile_mode: str | None
    batch_tile_size: int | None
    residual_norm_backend: str
    has_valid_token_mask: bool
    required_components: tuple[str, ...]
    resolved_components: tuple[str, ...]
    missing_components: tuple[str, ...]
    fallback_reasons: tuple[str, ...]

    @property
    def use_cuda_graph(self) -> bool:
        return self.runtime_wrapper == "cuda_graph"

    @property
    def use_batch_tiled_cuda_graph(self) -> bool:
        return self.runtime_wrapper == "batch_tiled_cuda_graph"

    @property
    def use_compiled_forward(self) -> bool:
        return self.runtime_wrapper == "compiled_forward"

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
        elif self.attention_backend == "triton_shape13_causal_attention":
            causal_mask = "online_causal"
        elif self.attention_backend in {
            "causal_sdpa",
            "mixed_fp16_cudnn",
            "mixed_fp16_efficient",
        }:
            causal_mask = "implicit_sdpa"
        else:
            causal_mask = "query_block"
        description = {
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
            "attention_compute_dtype": self.attention_compute_dtype,
            "linear_backend": self.linear_backend,
            "linear_compute_dtype": self.linear_compute_dtype,
            "runtime_wrapper": self.runtime_wrapper,
            "compile_mode": self.compile_mode,
            "residual_norm_backend": self.residual_norm_backend,
            "causal_mask": causal_mask,
            "valid_token_mask": (
                "direct_key_mask" if self.has_valid_token_mask else "none"
            ),
            "fallback_reasons": list(self.fallback_reasons),
        }
        if self.batch_tile_size is not None:
            description["batch_tile_size"] = self.batch_tile_size
        return description


@dataclass(frozen=True, slots=True)
class _Capabilities:
    causal_sdpa: bool
    cuda_graph: bool
    batch_tiled_cuda_graph: bool
    compiled_forward: bool
    compiled_residual_layer_norm: bool
    triton_residual_layer_norm: bool
    triton_mixed_residual_layer_norm: bool
    triton_shape13_causal_attention: bool
    mixed_fp16_cudnn_attention: bool
    mixed_fp16_efficient_attention: bool
    mixed_fp16_core: bool


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
    efficient_core_shape = is_mixed_fp16_core_efficient_runtime_family(
        batch_size=context.batch_size,
        seq_len=context.seq_len,
        num_heads=context.num_heads,
        head_dim=head_dim,
    )
    cudnn_core_shape = is_streamed_mixed_fp16_core_cudnn_slice(
        batch_size=context.batch_size,
        seq_len=context.seq_len,
        num_heads=context.num_heads,
        head_dim=head_dim,
        ffn_dim=context.ffn_dim or 0,
        num_layers=context.num_layers or 0,
    )
    mixed_fp16_core_shape = efficient_core_shape or cudnn_core_shape
    mixed_shape = (
        (context.seq_len >= 1024 and head_dim in {32, 64})
        or (
            context.batch_size in {64, 128}
            and context.seq_len == 128
            and context.d_model in {32, 128}
        )
        or efficient_core_shape
    )
    cudnn_mixed_shape = context.seq_len >= 1024 and head_dim == 64
    compiler = getattr(torch, "compile", None)
    mixed_runtime_available = False
    cudnn_runtime_available = False
    compute_capability: tuple[int, int] | None = None
    if context.device.type == "cuda":
        compute_capability = torch.cuda.get_device_capability(context.device)
        mixed_runtime_available = bool(
            torch.backends.cuda.mem_efficient_sdp_enabled()
            and compute_capability >= (8, 0)
        )
        cudnn_runtime_available = bool(
            torch.backends.cuda.cudnn_sdp_enabled()
            and torch.backends.cudnn.is_available()
            and compute_capability >= (8, 0)
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
    triton_residual_base = (
        inference
        and context.device.type == "cuda"
        and context.dtype == torch.float32
        and context.input_contiguous
        and not context.has_valid_token_mask
        and context.d_model == 128
        and context.batch_size * context.seq_len >= 1_000_000
        and not torch.compiler.is_compiling()
        and triton_residual_layer_norm_available()
    )
    batch_tiled_cuda_graph = bool(
        causal_sdpa
        and inference
        and context.device.type == "cuda"
        and context.dtype == torch.float32
        and context.input_contiguous
        and not context.has_valid_token_mask
        and context.mask_compatible
        and not torch.compiler.is_compiling()
        and is_shape06_batch_tiled_workload(
            batch_size=context.batch_size,
            seq_len=context.seq_len,
            d_model=context.d_model,
            num_heads=context.num_heads,
            ffn_dim=context.ffn_dim or 0,
            num_layers=context.num_layers or 0,
        )
    )
    triton_mixed_residual_base = bool(
        batch_tiled_cuda_graph
        and context.d_model == 128
        and triton_mixed_residual_layer_norm_available()
    )
    exact_shape = {
        "batch_size": context.batch_size,
        "seq_len": context.seq_len,
        "d_model": context.d_model,
        "num_heads": context.num_heads,
        "ffn_dim": context.ffn_dim or 0,
        "num_layers": context.num_layers or 0,
    }
    compiled_forward = bool(
        causal_sdpa
        and inference
        and context.device.type == "cuda"
        and context.dtype == torch.float32
        and context.input_contiguous
        and not context.has_valid_token_mask
        and context.mask_compatible
        and callable(compiler)
        and not torch.compiler.is_compiling()
        and is_compiled_forward_candidate_workload(**exact_shape)
    )
    triton_shape13_attention = bool(
        compiled_forward
        and compute_capability is not None
        and compute_capability >= (8, 0)
        and is_shape13_triton_attention_workload(**exact_shape)
        and triton_shape13_causal_attention_available()
    )
    return _Capabilities(
        causal_sdpa=causal_sdpa,
        cuda_graph=(
            causal_sdpa
            and inference
            and context.device.type == "cuda"
            and context.input_contiguous
        ),
        batch_tiled_cuda_graph=batch_tiled_cuda_graph,
        compiled_forward=compiled_forward,
        compiled_residual_layer_norm=compiled_residual_base,
        triton_residual_layer_norm=triton_residual_base,
        triton_mixed_residual_layer_norm=triton_mixed_residual_base,
        triton_shape13_causal_attention=triton_shape13_attention,
        mixed_fp16_cudnn_attention=(
            inference
            and context.causal
            and context.device.type == "cuda"
            and context.dtype == torch.float32
            and context.input_contiguous
            and not context.has_valid_token_mask
            and context.mask_compatible
            and cudnn_mixed_shape
            and cudnn_runtime_available
        ),
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
        mixed_fp16_core=(
            inference
            and context.causal
            and context.device.type == "cuda"
            and context.dtype == torch.float32
            and context.input_contiguous
            and not context.has_valid_token_mask
            and context.mask_compatible
            and mixed_fp16_core_shape
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
        spec.runtime is RuntimeWrapper.BATCH_TILED_CUDA_GRAPH
        and capabilities.batch_tiled_cuda_graph
    ):
        resolved.add(ExecutionComponent.BATCH_TILED_CUDA_GRAPH)
    if (
        spec.runtime is RuntimeWrapper.COMPILED_FORWARD
        and capabilities.compiled_forward
    ):
        resolved.add(ExecutionComponent.COMPILED_FORWARD)
    if (
        spec.residual_norm is ResidualNormBackend.COMPILED
        and capabilities.compiled_residual_layer_norm
    ):
        resolved.add(ExecutionComponent.COMPILED_RESIDUAL_LAYER_NORM)
    if (
        spec.residual_norm is ResidualNormBackend.TRITON
        and capabilities.triton_residual_layer_norm
    ):
        resolved.add(ExecutionComponent.TRITON_RESIDUAL_LAYER_NORM)
    if (
        spec.residual_norm is ResidualNormBackend.TRITON_MIXED
        and capabilities.triton_mixed_residual_layer_norm
    ):
        resolved.add(ExecutionComponent.TRITON_MIXED_RESIDUAL_LAYER_NORM)
    if spec.attention == "mixed_fp16_cudnn" and capabilities.mixed_fp16_cudnn_attention:
        resolved.add(ExecutionComponent.MIXED_FP16_CUDNN_ATTENTION)
    if (
        spec.attention == "mixed_fp16_efficient"
        and capabilities.mixed_fp16_efficient_attention
    ):
        resolved.add(ExecutionComponent.MIXED_FP16_EFFICIENT_ATTENTION)
    if (
        spec.attention == "triton_shape13_causal_attention"
        and capabilities.triton_shape13_causal_attention
    ):
        resolved.add(ExecutionComponent.TRITON_SHAPE13_CAUSAL_ATTENTION)
    if spec.linear_compute == "float16" and capabilities.mixed_fp16_core:
        resolved.add(ExecutionComponent.MIXED_FP16_CORE)
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
        linear_backend = (
            "autocast_fp16" if spec.linear_compute == "float16" else "torch"
        )
        linear_compute_dtype = (
            "float16"
            if spec.linear_compute == "float16"
            else str(context.dtype).removeprefix("torch.")
        )
        attention_compute_dtype = (
            "float16"
            if spec.attention
            in {
                "mixed_fp16_cudnn",
                "mixed_fp16_efficient",
                "triton_shape13_causal_attention",
            }
            else str(context.dtype).removeprefix("torch.")
        )
    else:
        attention_backend = "safe_streaming"
        selected_policy = "safe"
        attention_compute_dtype = str(context.dtype).removeprefix("torch.")
        linear_backend = "torch"
        linear_compute_dtype = str(context.dtype).removeprefix("torch.")
        resolved = frozenset()

    runtime_wrapper = (
        spec.runtime.value if fully_applied else RuntimeWrapper.EAGER.value
    )
    return ExecutionPlan(
        requested_policy=requested_policy,
        selected_policy=selected_policy,
        attention_backend=attention_backend,
        attention_compute_dtype=attention_compute_dtype,
        linear_backend=linear_backend,
        linear_compute_dtype=linear_compute_dtype,
        runtime_wrapper=runtime_wrapper,
        compile_mode=(spec.compile_mode if fully_applied else None),
        batch_tile_size=(spec.batch_tile_size if fully_applied else None),
        residual_norm_backend=(
            spec.residual_norm.value
            if fully_applied
            else ResidualNormBackend.TORCH.value
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
