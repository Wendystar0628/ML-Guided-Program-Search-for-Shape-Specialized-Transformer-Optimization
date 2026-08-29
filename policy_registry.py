"""Single, shape-independent registry for Transformer execution policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class ExecutionComponent(StrEnum):
    """Observable capabilities that make one policy distinct from another."""

    CAUSAL_SDPA = "causal_sdpa"
    CUDA_GRAPH = "cuda_graph"
    BATCH_TILED_CUDA_GRAPH = "batch_tiled_cuda_graph"
    COMPILED_FORWARD = "compiled_forward"
    COMPILED_RESIDUAL_LAYER_NORM = "compiled_residual_layer_norm"
    TRITON_RESIDUAL_LAYER_NORM = "triton_residual_layer_norm"
    TRITON_MIXED_RESIDUAL_LAYER_NORM = "triton_mixed_residual_layer_norm"
    TRITON_SHAPE13_CAUSAL_ATTENTION = "triton_shape13_causal_attention"
    MIXED_FP16_EFFICIENT_ATTENTION = "mixed_fp16_efficient_attention"
    MIXED_FP16_CUDNN_ATTENTION = "mixed_fp16_cudnn_attention"
    MIXED_FP16_CORE = "mixed_fp16_core"
    FP16_SHADOW_WEIGHTS = "fp16_shadow_weights"


class ResidualNormBackend(StrEnum):
    """Supported residual-plus-LayerNorm implementations."""

    TORCH = "torch"
    COMPILED = "compiled_residual_layer_norm"
    TRITON = "triton_residual_layer_norm"
    TRITON_MIXED = "triton_mixed_residual_layer_norm"


class RuntimeWrapper(StrEnum):
    """Supported outer execution schedules."""

    EAGER = "eager"
    CUDA_GRAPH = "cuda_graph"
    BATCH_TILED_CUDA_GRAPH = "batch_tiled_cuda_graph"
    COMPILED_FORWARD = "compiled_forward"


DEFAULT_COMPILED_FORWARD_MODE = "max-autotune"
COMPILED_FORWARD_MODES = frozenset(
    {
        DEFAULT_COMPILED_FORWARD_MODE,
        "max-autotune-no-cudagraphs",
    }
)


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """One execution composition, independent of workload shape and hardware."""

    policy_id: str
    attention: str = "safe_streaming"
    linear_compute: str = "input"
    residual_norm: ResidualNormBackend = ResidualNormBackend.TORCH
    runtime: RuntimeWrapper = RuntimeWrapper.EAGER
    compile_mode: str | None = None
    batch_tile_size: int | None = None
    reuse_unchanged_input: bool = False
    routable: bool = True

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id must not be empty")
        if self.attention not in {
            "safe_streaming",
            "causal_sdpa",
            "mixed_fp16_efficient",
            "mixed_fp16_cudnn",
            "triton_shape13_causal_attention",
        }:
            raise ValueError(f"unsupported attention backend: {self.attention}")
        if self.linear_compute not in {"input", "float16", "float16_shadow"}:
            raise ValueError(f"unsupported linear compute mode: {self.linear_compute}")
        try:
            backend = ResidualNormBackend(self.residual_norm)
        except ValueError as exc:
            raise ValueError(
                f"unsupported residual norm backend: {self.residual_norm}"
            ) from exc
        object.__setattr__(self, "residual_norm", backend)
        try:
            runtime = RuntimeWrapper(self.runtime)
        except ValueError as exc:
            raise ValueError(f"unsupported runtime wrapper: {self.runtime}") from exc
        object.__setattr__(self, "runtime", runtime)
        compile_mode = self.compile_mode
        if runtime is RuntimeWrapper.COMPILED_FORWARD:
            if compile_mode is None:
                compile_mode = DEFAULT_COMPILED_FORWARD_MODE
            if compile_mode not in COMPILED_FORWARD_MODES:
                raise ValueError(f"unsupported compiled-forward mode: {compile_mode}")
            object.__setattr__(self, "compile_mode", compile_mode)
        elif compile_mode is not None:
            raise ValueError(
                "compile_mode is valid only for the compiled-forward runtime"
            )
        if runtime is RuntimeWrapper.BATCH_TILED_CUDA_GRAPH:
            if (
                isinstance(self.batch_tile_size, bool)
                or not isinstance(self.batch_tile_size, int)
                or self.batch_tile_size <= 0
            ):
                raise ValueError(
                    "batch-tiled CUDA Graph runtime requires a positive tile size"
                )
        elif self.batch_tile_size is not None:
            raise ValueError(
                "batch_tile_size is valid only for batch-tiled CUDA Graph runtime"
            )
        if not isinstance(self.reuse_unchanged_input, bool):
            raise TypeError("reuse_unchanged_input must be a bool")
        if self.reuse_unchanged_input and runtime is not RuntimeWrapper.CUDA_GRAPH:
            raise ValueError(
                "reuse_unchanged_input is valid only for the CUDA Graph runtime"
            )

    @property
    def use_cuda_graph(self) -> bool:
        """Return whether the runtime owns a CUDA Graph."""

        return self.runtime in {
            RuntimeWrapper.CUDA_GRAPH,
            RuntimeWrapper.BATCH_TILED_CUDA_GRAPH,
        }

    @property
    def required_components(self) -> frozenset[ExecutionComponent]:
        """Derive capabilities from behavior instead of duplicating policy state."""

        components: set[ExecutionComponent] = set()
        if self.attention == "causal_sdpa":
            components.add(ExecutionComponent.CAUSAL_SDPA)
        elif self.attention == "mixed_fp16_efficient":
            components.add(ExecutionComponent.MIXED_FP16_EFFICIENT_ATTENTION)
        elif self.attention == "mixed_fp16_cudnn":
            components.add(ExecutionComponent.MIXED_FP16_CUDNN_ATTENTION)
        elif self.attention == "triton_shape13_causal_attention":
            components.add(ExecutionComponent.TRITON_SHAPE13_CAUSAL_ATTENTION)
        if self.linear_compute in {"float16", "float16_shadow"}:
            components.add(ExecutionComponent.MIXED_FP16_CORE)
        if self.linear_compute == "float16_shadow":
            components.add(ExecutionComponent.FP16_SHADOW_WEIGHTS)
        if self.runtime is RuntimeWrapper.CUDA_GRAPH:
            components.add(ExecutionComponent.CUDA_GRAPH)
        elif self.runtime is RuntimeWrapper.BATCH_TILED_CUDA_GRAPH:
            components.update(
                {
                    ExecutionComponent.CUDA_GRAPH,
                    ExecutionComponent.BATCH_TILED_CUDA_GRAPH,
                }
            )
        elif self.runtime is RuntimeWrapper.COMPILED_FORWARD:
            components.add(ExecutionComponent.COMPILED_FORWARD)
        if self.residual_norm is ResidualNormBackend.COMPILED:
            components.add(ExecutionComponent.COMPILED_RESIDUAL_LAYER_NORM)
        elif self.residual_norm is ResidualNormBackend.TRITON:
            components.add(ExecutionComponent.TRITON_RESIDUAL_LAYER_NORM)
        elif self.residual_norm is ResidualNormBackend.TRITON_MIXED:
            components.add(ExecutionComponent.TRITON_MIXED_RESIDUAL_LAYER_NORM)
        return frozenset(components)


_POLICY_SPECS = {
    "eager-sdpa": PolicySpec(
        "eager-sdpa",
        attention="causal_sdpa",
    ),
    "safe": PolicySpec("safe", routable=False),
    "graph": PolicySpec(
        "graph",
        attention="causal_sdpa",
        runtime=RuntimeWrapper.CUDA_GRAPH,
    ),
    "graph-fused-norm": PolicySpec(
        "graph-fused-norm",
        attention="causal_sdpa",
        runtime=RuntimeWrapper.CUDA_GRAPH,
        residual_norm=ResidualNormBackend.COMPILED,
    ),
    "mixed-fp16-efficient": PolicySpec(
        "mixed-fp16-efficient",
        attention="mixed_fp16_efficient",
    ),
    "mixed-fp16-cudnn": PolicySpec(
        "mixed-fp16-cudnn",
        attention="mixed_fp16_cudnn",
    ),
    "mixed-fp16-core-efficient": PolicySpec(
        "mixed-fp16-core-efficient",
        attention="mixed_fp16_efficient",
        linear_compute="float16",
    ),
    "mixed-fp16-core-efficient-triton-norm": PolicySpec(
        "mixed-fp16-core-efficient-triton-norm",
        attention="mixed_fp16_efficient",
        linear_compute="float16",
        residual_norm=ResidualNormBackend.TRITON,
    ),
    "mixed-fp16-core-cudnn": PolicySpec(
        "mixed-fp16-core-cudnn",
        attention="mixed_fp16_cudnn",
        linear_compute="float16",
    ),
    "graph-mixed-fp16-efficient": PolicySpec(
        "graph-mixed-fp16-efficient",
        attention="mixed_fp16_efficient",
        runtime=RuntimeWrapper.CUDA_GRAPH,
    ),
    "graph-mixed-fp16-efficient-compiled-norm": PolicySpec(
        "graph-mixed-fp16-efficient-compiled-norm",
        attention="mixed_fp16_efficient",
        residual_norm=ResidualNormBackend.COMPILED,
        runtime=RuntimeWrapper.CUDA_GRAPH,
    ),
    "graph-mixed-fp16-core-efficient-compiled-norm": PolicySpec(
        "graph-mixed-fp16-core-efficient-compiled-norm",
        attention="mixed_fp16_efficient",
        linear_compute="float16",
        residual_norm=ResidualNormBackend.COMPILED,
        runtime=RuntimeWrapper.CUDA_GRAPH,
    ),
    "graph-mixed-fp16-core-efficient-triton-mixed-norm-reuse-input": PolicySpec(
        "graph-mixed-fp16-core-efficient-triton-mixed-norm-reuse-input",
        attention="mixed_fp16_efficient",
        linear_compute="float16",
        residual_norm=ResidualNormBackend.TRITON_MIXED,
        runtime=RuntimeWrapper.CUDA_GRAPH,
        reuse_unchanged_input=True,
    ),
    "batch-tiled-mixed-fp16-core-efficient-compiled-norm": PolicySpec(
        "batch-tiled-mixed-fp16-core-efficient-compiled-norm",
        attention="mixed_fp16_efficient",
        linear_compute="float16",
        residual_norm=ResidualNormBackend.COMPILED,
        runtime=RuntimeWrapper.BATCH_TILED_CUDA_GRAPH,
        batch_tile_size=128,
    ),
    "batch-tiled-mixed-fp16-core-efficient-triton-mixed-norm": PolicySpec(
        "batch-tiled-mixed-fp16-core-efficient-triton-mixed-norm",
        attention="mixed_fp16_efficient",
        linear_compute="float16",
        residual_norm=ResidualNormBackend.TRITON_MIXED,
        runtime=RuntimeWrapper.BATCH_TILED_CUDA_GRAPH,
        batch_tile_size=128,
    ),
    "compiled-mixed-fp16-core-efficient": PolicySpec(
        "compiled-mixed-fp16-core-efficient",
        attention="mixed_fp16_efficient",
        linear_compute="float16",
        runtime=RuntimeWrapper.COMPILED_FORWARD,
    ),
    "compiled-shape08-fp16-shadow-weights": PolicySpec(
        "compiled-shape08-fp16-shadow-weights",
        attention="mixed_fp16_efficient",
        linear_compute="float16_shadow",
        runtime=RuntimeWrapper.COMPILED_FORWARD,
    ),
    "compiled-mixed-fp16-core-shape13-triton-attention": PolicySpec(
        "compiled-mixed-fp16-core-shape13-triton-attention",
        attention="triton_shape13_causal_attention",
        linear_compute="float16",
        runtime=RuntimeWrapper.COMPILED_FORWARD,
        compile_mode="max-autotune-no-cudagraphs",
    ),
}

POLICY_SPECS: Mapping[str, PolicySpec] = MappingProxyType(_POLICY_SPECS)
ROUTABLE_POLICY_IDS = frozenset(
    policy_id for policy_id, spec in POLICY_SPECS.items() if spec.routable
)
POLICY_SELECTORS = frozenset({"dispatch"})


def get_policy_spec(policy: str) -> PolicySpec:
    """Return the registered definition for an explicit execution policy."""

    normalized = policy.strip().lower()
    try:
        return POLICY_SPECS[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(POLICY_SPECS))
        raise ValueError(
            f"unknown runtime policy={policy!r}; expected one of {choices}"
        ) from exc


def policy_ids() -> frozenset[str]:
    """Return every policy accepted by explicit execution."""

    return frozenset(POLICY_SPECS)


__all__ = [
    "COMPILED_FORWARD_MODES",
    "DEFAULT_COMPILED_FORWARD_MODE",
    "POLICY_SELECTORS",
    "POLICY_SPECS",
    "ROUTABLE_POLICY_IDS",
    "ExecutionComponent",
    "PolicySpec",
    "ResidualNormBackend",
    "RuntimeWrapper",
    "get_policy_spec",
    "policy_ids",
]
