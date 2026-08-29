"""Finite execution candidates for the official Transformer workload."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from policy_registry import (
    POLICY_SPECS,
    ROUTABLE_POLICY_IDS,
    ExecutionComponent,
    get_policy_spec,
    policy_ids,
)
from runner.contracts import RunVariant, TransformerShape
from solution.shape_families import (
    is_compiled_forward_candidate_workload,
    is_graph_mixed_fp16_core_candidate_workload,
    is_measured_fp16_shadow_workload,
    is_measured_mixed_fp16_core_efficient_workload,
    is_measured_streamed_mixed_fp16_core_cudnn_workload,
    is_measured_triton_residual_norm_workload,
    is_shape05_graph_mixed_residual_norm_workload,
    is_shape06_batch_tiled_workload,
    is_shape08_fp16_shadow_workload,
    is_shape11_triton_dh8_attention_workload,
    is_shape13_triton_attention_workload,
)


@dataclass(frozen=True)
class PathExpectation:
    """One accepted value for a field in the planned execution path."""

    field: str
    accepted_values: frozenset[object]


@dataclass(frozen=True)
class ObservedPathExpectation:
    """Allowed and required branch values from observed forwards."""

    field: str
    accepted_values: frozenset[str]
    required_values: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ExecutionEvidence:
    """Evidence required before a candidate counts as actually applied."""

    selected_policies: frozenset[str] | None = None
    path_expectations: tuple[PathExpectation, ...] = ()
    observed_expectations: tuple[ObservedPathExpectation, ...] = ()

    def matches(
        self,
        *,
        solution_policy: str,
        execution_path: Mapping[str, Any],
        expected_request_policy: str | None = None,
    ) -> bool:
        """Return whether a worker report proves the intended route ran."""

        if execution_path.get("requested_policy") != (
            expected_request_policy or solution_policy
        ):
            return False
        selected = execution_path.get("selected_policy")
        accepted = self.selected_policies or frozenset({solution_policy})
        if selected not in accepted:
            return False
        if not all(
            execution_path.get(expectation.field) in expectation.accepted_values
            for expectation in self.path_expectations
        ):
            return False

        observed = execution_path.get("observed_execution")
        if not isinstance(observed, Mapping) or observed.get("complete") is not True:
            return False
        for expectation in self.observed_expectations:
            values = observed.get(expectation.field)
            if (
                not isinstance(values, Sequence)
                or isinstance(values, (str, bytes))
                or not values
                or not all(
                    isinstance(value, str) and value in expectation.accepted_values
                    for value in values
                )
                or not expectation.required_values.issubset(set(values))
            ):
                return False
        return True


Applicability = Callable[[TransformerShape, RunVariant], bool]


@dataclass(frozen=True)
class CandidateSpec:
    """One bounded strategy measured by calibration."""

    candidate_id: str
    solution_policy: str
    applies_to: Applicability
    applicability_description: str
    evidence: ExecutionEvidence
    workload_execution_modes: frozenset[str] = frozenset({"resident"})

    @property
    def required_components(self) -> frozenset[ExecutionComponent]:
        """Derive runtime requirements from the policy truth source."""

        return get_policy_spec(self.solution_policy).required_components

    @property
    def deployable(self) -> bool:
        """Return whether the policy may participate in runtime selection."""

        return get_policy_spec(self.solution_policy).routable

    @property
    def exact_route_eligible(self) -> bool:
        """Return whether this candidate may be persisted as a resident route."""

        return self.deployable and "resident" in self.workload_execution_modes

    def applies(self, shape: TransformerShape, variant: RunVariant) -> bool:
        return self.applies_to(shape, variant)

    def evidence_matches(self, execution_path: Mapping[str, Any]) -> bool:
        return self.evidence.matches(
            solution_policy=self.solution_policy,
            execution_path=execution_path,
        )

    def dispatch_evidence_matches(self, execution_path: Mapping[str, Any]) -> bool:
        return execution_path.get(
            "dispatch_policy"
        ) == self.solution_policy and self.evidence.matches(
            solution_policy=self.solution_policy,
            execution_path=execution_path,
            expected_request_policy="dispatch",
        )


def _safe_candidate(_shape: TransformerShape, _variant: RunVariant) -> bool:
    return True


def _native_sdpa_candidate(
    shape: TransformerShape,
    variant: RunVariant,
) -> bool:
    return (
        shape.causal
        and variant.padding_ratio == 0
        and variant.dtype in {"float16", "float32"}
    )


def _compact_graph_fused_norm_candidate(
    shape: TransformerShape,
    variant: RunVariant,
) -> bool:
    return bool(
        _native_sdpa_candidate(shape, variant)
        and variant.dtype == "float32"
        and shape.batch_size * shape.seq_len <= 2048
        and shape.d_model == 128
        and shape.ffn_dim == 128
    )


def _long_mixed_attention_candidate(
    shape: TransformerShape,
    variant: RunVariant,
) -> bool:
    return bool(
        _native_sdpa_candidate(shape, variant)
        and variant.dtype == "float32"
        and shape.seq_len >= 1024
        and shape.d_model // shape.num_heads in {32, 64}
    )


def _graph_mixed_attention_candidate(
    shape: TransformerShape,
    variant: RunVariant,
) -> bool:
    return bool(
        _native_sdpa_candidate(shape, variant)
        and variant.dtype == "float32"
        and shape.batch_size in {64, 128}
        and shape.seq_len == 128
        and shape.d_model in {32, 128}
    )


def _long_mixed_cudnn_candidate(
    shape: TransformerShape,
    variant: RunVariant,
) -> bool:
    return bool(
        _native_sdpa_candidate(shape, variant)
        and variant.dtype == "float32"
        and shape.seq_len >= 1024
        and shape.d_model // shape.num_heads == 64
    )


def _measured_mixed_fp16_core_candidate(
    shape: TransformerShape,
    variant: RunVariant,
) -> bool:
    """Limit full mixed-core execution to measured resident and streamed families."""

    return bool(
        _native_sdpa_candidate(shape, variant)
        and variant.dtype == "float32"
        and (
            is_measured_mixed_fp16_core_efficient_workload(
                batch_size=shape.batch_size,
                seq_len=shape.seq_len,
                d_model=shape.d_model,
                num_heads=shape.num_heads,
                ffn_dim=shape.ffn_dim,
                num_layers=shape.num_layers,
            )
            or is_measured_streamed_mixed_fp16_core_cudnn_workload(
                batch_size=shape.batch_size,
                seq_len=shape.seq_len,
                d_model=shape.d_model,
                num_heads=shape.num_heads,
                ffn_dim=shape.ffn_dim,
                num_layers=shape.num_layers,
            )
        )
    )


def _measured_streamed_mixed_fp16_core_cudnn_candidate(
    shape: TransformerShape,
    variant: RunVariant,
) -> bool:
    """Limit full mixed-core cuDNN execution to the measured streamed case."""

    return bool(
        _native_sdpa_candidate(shape, variant)
        and variant.dtype == "float32"
        and is_measured_streamed_mixed_fp16_core_cudnn_workload(
            batch_size=shape.batch_size,
            seq_len=shape.seq_len,
            d_model=shape.d_model,
            num_heads=shape.num_heads,
            ffn_dim=shape.ffn_dim,
            num_layers=shape.num_layers,
        )
    )


def _measured_triton_residual_norm_candidate(
    shape: TransformerShape,
    variant: RunVariant,
) -> bool:
    """Limit the custom residual-norm kernel to its measured Shape 06 family."""

    return bool(
        _native_sdpa_candidate(shape, variant)
        and variant.dtype == "float32"
        and is_measured_triton_residual_norm_workload(
            batch_size=shape.batch_size,
            seq_len=shape.seq_len,
            d_model=shape.d_model,
            num_heads=shape.num_heads,
            ffn_dim=shape.ffn_dim,
            num_layers=shape.num_layers,
        )
    )


def _graph_mixed_fp16_core_candidate(
    shape: TransformerShape,
    variant: RunVariant,
) -> bool:
    """Expose one bounded mixed-core Graph experiment for short workloads."""

    return bool(
        _native_sdpa_candidate(shape, variant)
        and variant.dtype == "float32"
        and is_graph_mixed_fp16_core_candidate_workload(
            batch_size=shape.batch_size,
            seq_len=shape.seq_len,
            d_model=shape.d_model,
            num_heads=shape.num_heads,
            ffn_dim=shape.ffn_dim,
            num_layers=shape.num_layers,
        )
    )


def _shape05_graph_mixed_residual_norm_candidate(
    shape: TransformerShape,
    variant: RunVariant,
) -> bool:
    """Expose the resident mixed residual-norm graph only for Shape 05."""

    return bool(
        _native_sdpa_candidate(shape, variant)
        and variant.dtype == "float32"
        and is_shape05_graph_mixed_residual_norm_workload(
            batch_size=shape.batch_size,
            seq_len=shape.seq_len,
            d_model=shape.d_model,
            num_heads=shape.num_heads,
            ffn_dim=shape.ffn_dim,
            num_layers=shape.num_layers,
        )
    )


def _shape06_batch_tiled_candidate(
    shape: TransformerShape,
    variant: RunVariant,
) -> bool:
    """Expose cache-blocked batch execution only for exact Shape 06."""

    return bool(
        _native_sdpa_candidate(shape, variant)
        and variant.dtype == "float32"
        and is_shape06_batch_tiled_workload(
            batch_size=shape.batch_size,
            seq_len=shape.seq_len,
            d_model=shape.d_model,
            num_heads=shape.num_heads,
            ffn_dim=shape.ffn_dim,
            num_layers=shape.num_layers,
        )
    )


def _compiled_forward_candidate(
    shape: TransformerShape,
    variant: RunVariant,
) -> bool:
    """Expose fixed-plan compilation only for the measured resident family."""

    return bool(
        _native_sdpa_candidate(shape, variant)
        and variant.dtype == "float32"
        and is_compiled_forward_candidate_workload(
            batch_size=shape.batch_size,
            seq_len=shape.seq_len,
            d_model=shape.d_model,
            num_heads=shape.num_heads,
            ffn_dim=shape.ffn_dim,
            num_layers=shape.num_layers,
        )
    )


def _shape08_fp16_shadow_candidate(
    shape: TransformerShape,
    variant: RunVariant,
) -> bool:
    """Expose prebuilt FP16 linear shadows only for exact Shape 08."""

    return bool(
        _native_sdpa_candidate(shape, variant)
        and variant.dtype == "float32"
        and is_shape08_fp16_shadow_workload(
            batch_size=shape.batch_size,
            seq_len=shape.seq_len,
            d_model=shape.d_model,
            num_heads=shape.num_heads,
            ffn_dim=shape.ffn_dim,
            num_layers=shape.num_layers,
        )
    )


def _short_graph_fp16_shadow_candidate(
    shape: TransformerShape,
    variant: RunVariant,
) -> bool:
    """Expose measured B64/D128 graph shadows without widening to Shape 07."""

    return bool(
        _native_sdpa_candidate(shape, variant)
        and variant.dtype == "float32"
        and shape.batch_size == 64
        and shape.seq_len == 128
        and shape.d_model == 128
        and shape.num_heads in {1, 2, 4}
        and is_measured_fp16_shadow_workload(
            batch_size=shape.batch_size,
            seq_len=shape.seq_len,
            d_model=shape.d_model,
            num_heads=shape.num_heads,
            ffn_dim=shape.ffn_dim,
            num_layers=shape.num_layers,
        )
    )


def _shape11_triton_dh8_attention_candidate(
    shape: TransformerShape,
    variant: RunVariant,
) -> bool:
    """Expose the exact padded-D16 online Attention specialization."""

    return bool(
        _native_sdpa_candidate(shape, variant)
        and variant.dtype == "float32"
        and is_shape11_triton_dh8_attention_workload(
            batch_size=shape.batch_size,
            seq_len=shape.seq_len,
            d_model=shape.d_model,
            num_heads=shape.num_heads,
            ffn_dim=shape.ffn_dim,
            num_layers=shape.num_layers,
        )
    )


def _shape13_triton_attention_candidate(
    shape: TransformerShape,
    variant: RunVariant,
) -> bool:
    """Expose the custom compiled path only for exact resident Shape 13."""

    return bool(
        _native_sdpa_candidate(shape, variant)
        and variant.dtype == "float32"
        and is_shape13_triton_attention_workload(
            batch_size=shape.batch_size,
            seq_len=shape.seq_len,
            d_model=shape.d_model,
            num_heads=shape.num_heads,
            ffn_dim=shape.ffn_dim,
            num_layers=shape.num_layers,
        )
    )


def _expect(field: str, *values: object) -> PathExpectation:
    return PathExpectation(field, frozenset(values))


def _observe(field: str, value: str) -> ObservedPathExpectation:
    return ObservedPathExpectation(
        field,
        frozenset({value}),
        frozenset({value}),
    )


def _native_evidence(
    *,
    policy: str,
    attention_backend: str = "causal_sdpa",
    runtime_wrapper: str = "eager",
    residual_norm_backend: str = "torch",
    attention_compute_dtype: str | None = None,
    linear_backend: str | None = None,
    linear_compute_dtype: str | None = None,
    batch_tile_size: int | None = None,
    reuse_unchanged_input: bool | None = None,
    use_triton_initial_fp16_norm: bool | None = None,
) -> ExecutionEvidence:
    observed = [
        _observe("attention_backends", attention_backend),
        _observe("residual_norm_backends", residual_norm_backend),
    ]
    if runtime_wrapper != "eager":
        observed.append(_observe("runtime_wrappers", runtime_wrapper))
    path_expectations = [
        _expect("attention_backend", attention_backend),
        _expect("runtime_wrapper", runtime_wrapper),
        _expect("residual_norm_backend", residual_norm_backend),
    ]
    compile_mode = get_policy_spec(policy).compile_mode
    if compile_mode is not None:
        path_expectations.append(_expect("compile_mode", compile_mode))
    if attention_compute_dtype is not None:
        path_expectations.append(
            _expect("attention_compute_dtype", attention_compute_dtype)
        )
        observed.append(_observe("attention_compute_dtypes", attention_compute_dtype))
    if linear_backend is not None:
        path_expectations.append(_expect("linear_backend", linear_backend))
        observed.append(_observe("linear_backends", linear_backend))
    if linear_compute_dtype is not None:
        path_expectations.append(_expect("linear_compute_dtype", linear_compute_dtype))
        observed.append(_observe("linear_compute_dtypes", linear_compute_dtype))
    if batch_tile_size is not None:
        path_expectations.append(_expect("batch_tile_size", batch_tile_size))
    if reuse_unchanged_input is not None:
        path_expectations.append(
            _expect("reuse_unchanged_input", reuse_unchanged_input)
        )
    if use_triton_initial_fp16_norm is not None:
        path_expectations.append(
            _expect(
                "use_triton_initial_fp16_norm",
                use_triton_initial_fp16_norm,
            )
        )
    return ExecutionEvidence(
        selected_policies=frozenset({policy}),
        path_expectations=tuple(path_expectations),
        observed_expectations=tuple(observed),
    )


_CANDIDATE_SPECS = (
    CandidateSpec(
        "eager-sdpa",
        "eager-sdpa",
        _native_sdpa_candidate,
        "causal FP16/FP32 shapes without padding",
        _native_evidence(policy="eager-sdpa"),
    ),
    CandidateSpec(
        "eager-safe",
        "safe",
        _safe_candidate,
        "all valid shapes and variants",
        ExecutionEvidence(
            selected_policies=frozenset({"safe"}),
            path_expectations=(
                _expect("attention_backend", "safe_streaming"),
                _expect("runtime_wrapper", "eager"),
                _expect("residual_norm_backend", "torch"),
            ),
            observed_expectations=(
                _observe("attention_backends", "safe_streaming"),
                _observe("residual_norm_backends", "torch"),
            ),
        ),
    ),
    CandidateSpec(
        "graph",
        "graph",
        _native_sdpa_candidate,
        "causal FP16/FP32 shapes without padding",
        _native_evidence(policy="graph", runtime_wrapper="cuda_graph"),
    ),
    CandidateSpec(
        "graph-fused-norm",
        "graph-fused-norm",
        _compact_graph_fused_norm_candidate,
        "up to 2048 FP32 width-128 tokens with full-forward Graph replay",
        _native_evidence(
            policy="graph-fused-norm",
            runtime_wrapper="cuda_graph",
            residual_norm_backend="compiled_residual_layer_norm",
        ),
    ),
    CandidateSpec(
        "mixed-fp16-efficient",
        "mixed-fp16-efficient",
        _long_mixed_attention_candidate,
        "long causal FP32 shapes with head dimension 32 or 64",
        _native_evidence(
            policy="mixed-fp16-efficient",
            attention_backend="mixed_fp16_efficient",
        ),
        frozenset({"resident", "batch_streamed"}),
    ),
    CandidateSpec(
        "mixed-fp16-cudnn",
        "mixed-fp16-cudnn",
        _long_mixed_cudnn_candidate,
        "long causal FP32 shapes with head dimension 64 and cuDNN SDPA",
        _native_evidence(
            policy="mixed-fp16-cudnn",
            attention_backend="mixed_fp16_cudnn",
        ),
        frozenset({"resident", "batch_streamed"}),
    ),
    CandidateSpec(
        "mixed-fp16-core-efficient",
        "mixed-fp16-core-efficient",
        _measured_mixed_fp16_core_candidate,
        "measured FP32 outer-state mixed core for Shapes 06, 08, 13, and streamed 14",
        _native_evidence(
            policy="mixed-fp16-core-efficient",
            attention_backend="mixed_fp16_efficient",
            attention_compute_dtype="float16",
            linear_backend="autocast_fp16",
            linear_compute_dtype="float16",
        ),
        frozenset({"resident", "batch_streamed"}),
    ),
    CandidateSpec(
        "mixed-fp16-core-efficient-triton-norm",
        "mixed-fp16-core-efficient-triton-norm",
        _measured_triton_residual_norm_candidate,
        "Shape 06 mixed FP16 core with measured Triton residual LayerNorm",
        _native_evidence(
            policy="mixed-fp16-core-efficient-triton-norm",
            attention_backend="mixed_fp16_efficient",
            residual_norm_backend="triton_residual_layer_norm",
            attention_compute_dtype="float16",
            linear_backend="autocast_fp16",
            linear_compute_dtype="float16",
        ),
    ),
    CandidateSpec(
        "mixed-fp16-core-cudnn",
        "mixed-fp16-core-cudnn",
        _measured_streamed_mixed_fp16_core_cudnn_candidate,
        "measured streamed FP32 outer-state mixed core for Shape 14",
        _native_evidence(
            policy="mixed-fp16-core-cudnn",
            attention_backend="mixed_fp16_cudnn",
            attention_compute_dtype="float16",
            linear_backend="autocast_fp16",
            linear_compute_dtype="float16",
        ),
        frozenset({"batch_streamed"}),
    ),
    CandidateSpec(
        "graph-mixed-fp16-efficient",
        "graph-mixed-fp16-efficient",
        _graph_mixed_attention_candidate,
        "B64/B128 S128 causal FP32 shapes with model width 32 or 128",
        _native_evidence(
            policy="graph-mixed-fp16-efficient",
            attention_backend="mixed_fp16_efficient",
            runtime_wrapper="cuda_graph",
        ),
    ),
    CandidateSpec(
        "graph-mixed-fp16-efficient-compiled-norm",
        "graph-mixed-fp16-efficient-compiled-norm",
        _graph_mixed_attention_candidate,
        "B64/B128 S128 FP32 mixed Attention with Graph and compiled residual norm",
        _native_evidence(
            policy="graph-mixed-fp16-efficient-compiled-norm",
            attention_backend="mixed_fp16_efficient",
            runtime_wrapper="cuda_graph",
            residual_norm_backend="compiled_residual_layer_norm",
        ),
    ),
    CandidateSpec(
        "graph-mixed-fp16-core-efficient-compiled-norm",
        "graph-mixed-fp16-core-efficient-compiled-norm",
        _graph_mixed_fp16_core_candidate,
        "B64/B128 S128 FP32 mixed core with Graph and compiled residual norm",
        _native_evidence(
            policy="graph-mixed-fp16-core-efficient-compiled-norm",
            attention_backend="mixed_fp16_efficient",
            runtime_wrapper="cuda_graph",
            residual_norm_backend="compiled_residual_layer_norm",
            attention_compute_dtype="float16",
            linear_backend="autocast_fp16",
            linear_compute_dtype="float16",
        ),
    ),
    CandidateSpec(
        "graph-fp16-shadow-efficient-compiled-norm",
        "graph-fp16-shadow-efficient-compiled-norm",
        _short_graph_fp16_shadow_candidate,
        "measured B64/D128 Graph path with persistent FP16 linear weights",
        _native_evidence(
            policy="graph-fp16-shadow-efficient-compiled-norm",
            attention_backend="mixed_fp16_efficient",
            runtime_wrapper="cuda_graph",
            residual_norm_backend="compiled_residual_layer_norm",
            attention_compute_dtype="float16",
            linear_backend="fp16_shadow",
            linear_compute_dtype="float16",
        ),
    ),
    CandidateSpec(
        "graph-mixed-fp16-core-efficient-triton-mixed-norm-reuse-input",
        "graph-mixed-fp16-core-efficient-triton-mixed-norm-reuse-input",
        _shape05_graph_mixed_residual_norm_candidate,
        "Shape 05 mixed residual-norm Graph with version-aware input staging",
        _native_evidence(
            policy=(
                "graph-mixed-fp16-core-efficient-triton-mixed-norm-reuse-input"
            ),
            attention_backend="mixed_fp16_efficient",
            runtime_wrapper="cuda_graph",
            residual_norm_backend="triton_mixed_residual_layer_norm",
            attention_compute_dtype="float16",
            linear_backend="autocast_fp16",
            linear_compute_dtype="float16",
            reuse_unchanged_input=True,
        ),
    ),
    CandidateSpec(
        "graph-fp16-shadow-efficient-triton-mixed-norm-reuse-input",
        "graph-fp16-shadow-efficient-triton-mixed-norm-reuse-input",
        _shape05_graph_mixed_residual_norm_candidate,
        "Shape 05 Graph with mixed norm, input reuse, and persistent FP16 weights",
        _native_evidence(
            policy="graph-fp16-shadow-efficient-triton-mixed-norm-reuse-input",
            attention_backend="mixed_fp16_efficient",
            runtime_wrapper="cuda_graph",
            residual_norm_backend="triton_mixed_residual_layer_norm",
            attention_compute_dtype="float16",
            linear_backend="fp16_shadow",
            linear_compute_dtype="float16",
            reuse_unchanged_input=True,
        ),
    ),
    CandidateSpec(
        "batch-tiled-mixed-fp16-core-efficient-compiled-norm",
        "batch-tiled-mixed-fp16-core-efficient-compiled-norm",
        _shape06_batch_tiled_candidate,
        "Shape 06 cache-blocked B128 full-model CUDA Graph tiles",
        _native_evidence(
            policy="batch-tiled-mixed-fp16-core-efficient-compiled-norm",
            attention_backend="mixed_fp16_efficient",
            runtime_wrapper="batch_tiled_cuda_graph",
            residual_norm_backend="compiled_residual_layer_norm",
            attention_compute_dtype="float16",
            linear_backend="autocast_fp16",
            linear_compute_dtype="float16",
            batch_tile_size=128,
        ),
    ),
    CandidateSpec(
        "batch-tiled-shape06-triton-mixed-norm-fp16-shadow",
        "batch-tiled-shape06-triton-mixed-norm-fp16-shadow",
        _shape06_batch_tiled_candidate,
        "Shape 06 B128 graph tiles with mixed norm and persistent FP16 weights",
        _native_evidence(
            policy="batch-tiled-shape06-triton-mixed-norm-fp16-shadow",
            attention_backend="mixed_fp16_efficient",
            runtime_wrapper="batch_tiled_cuda_graph",
            residual_norm_backend="triton_mixed_residual_layer_norm",
            attention_compute_dtype="float16",
            linear_backend="fp16_shadow",
            linear_compute_dtype="float16",
            batch_tile_size=128,
            use_triton_initial_fp16_norm=True,
        ),
    ),
    CandidateSpec(
        "compiled-mixed-fp16-core-efficient",
        "compiled-mixed-fp16-core-efficient",
        _compiled_forward_candidate,
        "fixed-plan full-stack compilation for measured Shapes 07, 08, 11, and 13",
        _native_evidence(
            policy="compiled-mixed-fp16-core-efficient",
            attention_backend="mixed_fp16_efficient",
            runtime_wrapper="compiled_forward",
            attention_compute_dtype="float16",
            linear_backend="autocast_fp16",
            linear_compute_dtype="float16",
        ),
    ),
    CandidateSpec(
        "compiled-shape08-fp16-shadow-weights",
        "compiled-shape08-fp16-shadow-weights",
        _shape08_fp16_shadow_candidate,
        "exact Shape 08 with compiled mixed core and prebuilt FP16 weights",
        _native_evidence(
            policy="compiled-shape08-fp16-shadow-weights",
            attention_backend="mixed_fp16_efficient",
            runtime_wrapper="compiled_forward",
            attention_compute_dtype="float16",
            linear_backend="fp16_shadow",
            linear_compute_dtype="float16",
        ),
    ),
    CandidateSpec(
        "compiled-shape11-dh8-triton-fp16-shadow",
        "compiled-shape11-dh8-triton-fp16-shadow",
        _shape11_triton_dh8_attention_candidate,
        "Shape 11 padded-D16 online Attention with direct BSD output",
        _native_evidence(
            policy="compiled-shape11-dh8-triton-fp16-shadow",
            attention_backend="triton_dh8_causal_attention_bsd",
            runtime_wrapper="compiled_forward",
            attention_compute_dtype="float16",
            linear_backend="fp16_shadow",
            linear_compute_dtype="float16",
        ),
    ),
    CandidateSpec(
        "compiled-shape13-triton-attention-fp16-shadow",
        "compiled-shape13-triton-attention-fp16-shadow",
        _shape13_triton_attention_candidate,
        "Shape 13 exact online Attention with persistent FP16 linear weights",
        _native_evidence(
            policy="compiled-shape13-triton-attention-fp16-shadow",
            attention_backend="triton_shape13_causal_attention",
            runtime_wrapper="compiled_forward",
            attention_compute_dtype="float16",
            linear_backend="fp16_shadow",
            linear_compute_dtype="float16",
        ),
    ),
)


def _build_registry(specs: Sequence[CandidateSpec]) -> Mapping[str, CandidateSpec]:
    registry: dict[str, CandidateSpec] = {}
    policy_owners: set[str] = set()
    for spec in specs:
        if spec.candidate_id in registry:
            raise RuntimeError(f"duplicate candidate id: {spec.candidate_id}")
        if spec.solution_policy in policy_owners:
            raise RuntimeError(
                f"multiple candidates map to policy {spec.solution_policy!r}"
            )
        if spec.solution_policy not in POLICY_SPECS:
            raise RuntimeError(
                f"candidate maps to unknown policy {spec.solution_policy!r}"
            )
        policy_owners.add(spec.solution_policy)
        registry[spec.candidate_id] = spec

    if policy_owners != policy_ids():
        missing = ", ".join(sorted(policy_ids() - policy_owners))
        extra = ", ".join(sorted(policy_owners - policy_ids()))
        raise RuntimeError(
            "candidate and policy registries disagree; "
            f"missing={missing}; extra={extra}"
        )
    deployable = {spec.solution_policy for spec in specs if spec.deployable}
    if deployable != ROUTABLE_POLICY_IDS:
        raise RuntimeError("candidate deployability disagrees with policy registry")
    return MappingProxyType(registry)


CANDIDATE_SPECS: Mapping[str, CandidateSpec] = _build_registry(_CANDIDATE_SPECS)


def candidate_spec(candidate_id: str) -> CandidateSpec | None:
    return CANDIDATE_SPECS.get(candidate_id)


def candidate_specs_for_shape(
    shape: TransformerShape,
    variant: RunVariant,
) -> tuple[CandidateSpec, ...]:
    """Return applicable candidates in stable calibration order."""

    return tuple(spec for spec in _CANDIDATE_SPECS if spec.applies(shape, variant))


def candidate_specs_for_execution_mode(
    shape: TransformerShape,
    variant: RunVariant,
    execution_mode: str,
) -> tuple[CandidateSpec, ...]:
    """Return deployable candidates compatible with one workload executor."""

    return tuple(
        spec
        for spec in candidate_specs_for_shape(shape, variant)
        if spec.deployable and execution_mode in spec.workload_execution_modes
    )


def candidate_spec_for_policy(
    shape: TransformerShape,
    variant: RunVariant,
    policy: str,
    *,
    deployable_only: bool = False,
) -> CandidateSpec | None:
    """Resolve the single candidate for a policy and shape variant."""

    matches = [
        spec
        for spec in candidate_specs_for_shape(shape, variant)
        if spec.solution_policy == policy and (not deployable_only or spec.deployable)
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple candidates map to policy {policy!r} for {shape.case_id}"
        )
    return matches[0] if matches else None


def deployable_policy_ids() -> frozenset[str]:
    return frozenset(
        spec.solution_policy for spec in _CANDIDATE_SPECS if spec.deployable
    )


def exact_route_policy_ids() -> frozenset[str]:
    """Return policies backed by a resident exact-route candidate."""

    return frozenset(
        spec.solution_policy for spec in _CANDIDATE_SPECS if spec.exact_route_eligible
    )


__all__ = [
    "CANDIDATE_SPECS",
    "CandidateSpec",
    "ExecutionEvidence",
    "ObservedPathExpectation",
    "PathExpectation",
    "candidate_spec",
    "candidate_spec_for_policy",
    "candidate_specs_for_execution_mode",
    "candidate_specs_for_shape",
    "deployable_policy_ids",
    "exact_route_policy_ids",
]
