"""Resolve one immutable execution plan for reporting and forward execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .policies import ExecutionComponent, PolicySpec

# Batch tiling is a capacity strategy, not an official-shape constant.  The
# budget caps the input elements processed by one tile and naturally adapts to
# batch, sequence, and model width.
_BATCH_TILE_ELEMENT_BUDGET = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Runtime facts that can change execution eligibility."""

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
    """Concrete backends consumed by one Transformer layer."""

    attention_backend: str
    block_backend: str


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Single source of truth for the selected model execution path."""

    requested_policy: str
    selected_policy: str
    attention_backend: str
    runtime_wrapper: str
    batch_strategy: str
    batch_tile_size: int | None
    block_backend: str
    required_components: tuple[str, ...]
    resolved_components: tuple[str, ...]
    missing_components: tuple[str, ...]
    layers: tuple[LayerExecutionPlan, ...]
    fallback_reasons: tuple[str, ...]

    @property
    def use_cuda_graph(self) -> bool:
        return self.runtime_wrapper == "cuda_graph"

    @property
    def use_batch_tiling(self) -> bool:
        return self.batch_strategy == "tiled"

    @property
    def use_inplace_exact_gelu(self) -> bool:
        return self.block_backend == "inplace_exact_gelu"

    def describe(
        self,
        *,
        dispatch_source: str | None,
        dispatch_table_sha256: str | None,
        dispatch_policy: str | None,
        route_origin: str | None,
        causal: bool,
    ) -> dict[str, Any]:
        """Serialize the same plan that forward consumes."""

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
            "batch_strategy": self.batch_strategy,
            "batch_tile_size": self.batch_tile_size,
            "block_backend": self.block_backend,
            "layer_backends": [
                {
                    "attention": layer.attention_backend,
                    "block": layer.block_backend,
                }
                for layer in self.layers
            ],
            "causal_mask": "implicit" if causal else "none",
            "valid_token_mask": "direct_key_mask",
            "fallback_reason": list(self.fallback_reasons) or None,
        }


@dataclass(frozen=True, slots=True)
class _Capabilities:
    causal_sdpa: bool
    cuda_graph: bool
    batch_tile_size: int | None
    inplace_exact_gelu: bool


def _batch_tile_size(context: ExecutionContext) -> int | None:
    elements_per_sample = context.sequence_length * context.d_model
    if elements_per_sample <= 0:
        return None
    tile_size = max(1, _BATCH_TILE_ELEMENT_BUDGET // elements_per_sample)
    tile_size = min(context.batch_size, tile_size)
    return tile_size if tile_size < context.batch_size else None


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
    return _Capabilities(
        causal_sdpa=causal_sdpa,
        cuda_graph=(
            causal_sdpa
            and inference
            and context.device.type == "cuda"
            and context.input_contiguous
        ),
        batch_tile_size=(
            _batch_tile_size(context)
            if causal_sdpa and inference and context.input_contiguous
            else None
        ),
        inplace_exact_gelu=(causal_sdpa and inference),
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
    if spec.use_batch_tiling and capabilities.batch_tile_size is not None:
        resolved.add(ExecutionComponent.BATCH_TILING)
    if spec.use_inplace_exact_gelu and capabilities.inplace_exact_gelu:
        resolved.add(ExecutionComponent.INPLACE_EXACT_GELU)
    return frozenset(resolved)


def resolve_execution_plan(
    spec: PolicySpec,
    context: ExecutionContext,
    *,
    requested_policy: str,
    dispatch_policy: str | None,
) -> ExecutionPlan:
    """Resolve policy intent atomically; unsupported policies fall back to safe."""

    capabilities = _capabilities(context)
    resolved = _resolved_components(spec, capabilities)
    missing = spec.required_components - resolved
    fallback_reasons = tuple(
        f"{component.value}_not_eligible"
        for component in sorted(missing, key=lambda item: item.value)
    )

    # Explicit policies are atomic.  This keeps observed policy identity honest
    # and prevents a partially-applied candidate from being promoted.
    fully_applied = not missing
    if spec.policy_id == "auto":
        attention_backend = (
            "causal_sdpa" if capabilities.causal_sdpa else "safe_streaming"
        )
        selected_policy = "auto"
    elif fully_applied:
        attention_backend = (
            "causal_sdpa"
            if spec.attention == "causal_sdpa"
            else "safe_streaming"
        )
        selected_policy = spec.policy_id
    else:
        attention_backend = "safe_streaming"
        selected_policy = "safe"
        resolved = frozenset()

    use_cuda_graph = fully_applied and spec.use_cuda_graph
    use_batch_tiling = fully_applied and spec.use_batch_tiling
    use_inplace_exact_gelu = fully_applied and spec.use_inplace_exact_gelu
    batch_tile_size = (
        capabilities.batch_tile_size if use_batch_tiling else None
    )
    runtime_wrapper = "cuda_graph" if use_cuda_graph else "eager"
    batch_strategy = "tiled" if use_batch_tiling else "full"
    block_backend = (
        "inplace_exact_gelu" if use_inplace_exact_gelu else "torch"
    )
    layers = tuple(
        LayerExecutionPlan(attention_backend, block_backend)
        for _ in range(context.num_layers)
    )

    return ExecutionPlan(
        requested_policy=requested_policy,
        selected_policy=selected_policy,
        attention_backend=attention_backend,
        runtime_wrapper=runtime_wrapper,
        batch_strategy=batch_strategy,
        batch_tile_size=batch_tile_size,
        block_backend=block_backend,
        required_components=tuple(
            sorted(component.value for component in spec.required_components)
        ),
        resolved_components=tuple(
            sorted(component.value for component in resolved)
        ),
        missing_components=tuple(
            sorted(component.value for component in missing)
        ),
        layers=layers,
        fallback_reasons=fallback_reasons,
    )


__all__ = [
    "ExecutionContext",
    "ExecutionPlan",
    "LayerExecutionPlan",
    "resolve_execution_plan",
]
