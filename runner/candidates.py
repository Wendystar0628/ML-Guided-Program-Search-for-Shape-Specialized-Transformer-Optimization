"""Typed registry for the finite optimization candidates used by the runner.

The registry is deliberately project-specific.  It is the single runner-side
source for candidate identity, workload applicability, runtime requirements,
deployment eligibility, and the execution evidence that proves a requested
route actually ran.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from runner.contracts import WorkloadCase
from solution.policies import ROUTABLE_POLICY_IDS


class CapabilityTag(StrEnum):
    """Runtime capabilities required before measuring a candidate."""

    CUDA = "cuda"
    TRITON = "triton"
    CUDA_GRAPH = "cuda_graph"


class RoutingTag(StrEnum):
    """Transparent cost-model features used to rank active candidates."""

    SAFE_FALLBACK = "safe_fallback"
    REFERENCE_CONTROL = "reference_control"
    TORCH_CONTROL = "torch_control"
    GENERAL_TRITON = "general_triton"
    GRAPH = "graph"
    RUNNER_GRAPH = "runner_graph"
    SOLUTION_GRAPH = "solution_graph"
    BALANCED_GRAPH = "balanced_graph"
    COMPILE_DEFAULT = "compile_default"
    COMPILE_REDUCE_OVERHEAD = "compile_reduce_overhead"
    COMPILE_MAX_AUTOTUNE = "compile_max_autotune"
    ATTENTION_PREPROCESS = "attention_preprocess"
    ATTENTION_ONLINE = "attention_online"
    S512_NATIVE_SOFTMAX = "s512_native_softmax"
    PADDING_FUSED = "padding_fused"
    PADDING_PACKED = "padding_packed"
    WIDE_INPLACE = "wide_inplace"


@dataclass(frozen=True)
class PathExpectation:
    """One accepted value for a field in the reported execution path."""

    field: str
    accepted_values: frozenset[object]


@dataclass(frozen=True)
class ObservedPathExpectation:
    """Allowed and required branch values from one observed eager forward."""

    field: str
    accepted_values: frozenset[str]
    required_values: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ExecutionEvidence:
    """Evidence required before a tuning observation counts as applied."""

    selected_policies: frozenset[str] | None = None
    path_expectations: tuple[PathExpectation, ...] = ()
    requires_observed_execution: bool = False
    observed_expectations: tuple[ObservedPathExpectation, ...] = ()

    def matches(
        self,
        *,
        solution_policy: str,
        execution_path: Mapping[str, Any],
    ) -> bool:
        """Return whether a worker report proves the intended route ran."""

        if execution_path.get("requested_policy") != solution_policy:
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
        if not (self.requires_observed_execution or self.observed_expectations):
            return True
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


Applicability = Callable[[WorkloadCase], bool]


@dataclass(frozen=True)
class CandidateSpec:
    """Complete runner contract for one bounded optimization candidate."""

    candidate_id: str
    solution_policy: str
    applies_to: Applicability
    applicability_description: str
    hardware_case_support: Applicability | None = None
    hardware_case_support_description: str | None = None
    compile_solution: bool = False
    compile_mode: str = "default"
    cuda_graph_solution: bool = False
    capability_tags: frozenset[CapabilityTag] = frozenset()
    routing_tags: frozenset[RoutingTag] = frozenset()
    deployable: bool = True
    evidence: ExecutionEvidence = ExecutionEvidence()
    minimum_compute_capability: tuple[int, int] | None = None

    def applies(self, case: WorkloadCase) -> bool:
        """Return whether the candidate is meaningful for ``case``."""

        return self.applies_to(case)

    def supports_case_on_hardware(self, case: WorkloadCase) -> bool:
        """Return whether the bounded backend can execute this case directly."""

        return self.hardware_case_support is None or self.hardware_case_support(case)

    def evidence_matches(self, execution_path: Mapping[str, Any]) -> bool:
        """Return whether a worker report proves this candidate was applied."""

        return self.evidence.matches(
            solution_policy=self.solution_policy,
            execution_path=execution_path,
        )


def _always(_: WorkloadCase) -> bool:
    return True


def _short_static(case: WorkloadCase) -> bool:
    return case.seq_len <= 128


def _wide_family(case: WorkloadCase) -> bool:
    return case.d_model >= 1024 or case.ffn_dim >= 4096


def _compile_default(case: WorkloadCase) -> bool:
    return _short_static(case) or _wide_family(case)


def _balanced_graph(case: WorkloadCase) -> bool:
    return (
        case.batch_size == 8
        and case.seq_len == 128
        and case.d_model == 512
        and case.num_heads == 8
        and case.ffn_dim == 2048
        and case.num_layers == 6
        and case.dtype == "float16"
        and not case.causal
        and case.padding_ratio == 0
    )


def _attention_preprocess(case: WorkloadCase) -> bool:
    return case.dtype == "float16" and (
        case.seq_len,
        case.d_model // case.num_heads,
    ) in {(64, 32), (2048, 64)}


def _s512_family(case: WorkloadCase) -> bool:
    return (
        case.batch_size == 8
        and case.seq_len == 512
        and case.d_model == 512
        and case.num_heads == 8
        and case.ffn_dim == 2048
        and case.num_layers == 4
        and case.dtype == "float16"
    )


def _long_attention(case: WorkloadCase) -> bool:
    return (
        case.batch_size == 1
        and case.seq_len == 2048
        and case.d_model == 512
        and case.num_heads == 8
        and case.ffn_dim == 2048
        and case.num_layers == 4
        and case.dtype == "float16"
    )


def _launch_family(case: WorkloadCase) -> bool:
    return (
        case.batch_size == 1
        and case.seq_len == 64
        and case.d_model == 256
        and case.num_heads == 8
        and case.ffn_dim == 1024
        and case.num_layers == 4
        and case.dtype == "float16"
        and not case.causal
        and case.padding_ratio == 0
    )


def _padding_route(case: WorkloadCase) -> bool:
    # The benchmark always supplies a valid-token mask, including the all-true
    # case.  S512 full and padded cases therefore share the same route identity
    # and candidate family without relying on a case-id naming convention.
    return case.padding_ratio > 0 or _s512_family(case) or _launch_family(case)


def _runner_graph(case: WorkloadCase) -> bool:
    return _launch_family(case)


def _wide_exact(case: WorkloadCase) -> bool:
    return (
        case.batch_size == 16
        and case.seq_len == 256
        and case.d_model == 1024
        and case.num_heads == 8
        and case.ffn_dim == 4096
        and case.num_layers == 6
        and case.dtype == "bfloat16"
        and not case.causal
    )


def _expect(field: str, *values: object) -> PathExpectation:
    return PathExpectation(field, frozenset(values))


def _observe(
    field: str,
    *accepted_values: str,
    required_values: frozenset[str] = frozenset(),
) -> ObservedPathExpectation:
    return ObservedPathExpectation(
        field,
        frozenset(accepted_values),
        required_values,
    )


_CUDA = frozenset({CapabilityTag.CUDA})
_CUDA_TRITON = frozenset({CapabilityTag.CUDA, CapabilityTag.TRITON})
_CUDA_GRAPH = frozenset({CapabilityTag.CUDA, CapabilityTag.CUDA_GRAPH})


_CANDIDATE_SPECS = (
    CandidateSpec(
        "eager-reference",
        "reference",
        _always,
        "all validated workload shapes",
        routing_tags=frozenset({RoutingTag.REFERENCE_CONTROL}),
        evidence=ExecutionEvidence(
            path_expectations=(
                _expect("resolved_attention", "explicit_reference_order"),
            )
        ),
    ),
    CandidateSpec(
        "eager-torch",
        "torch",
        _always,
        "all validated workload shapes",
        routing_tags=frozenset({RoutingTag.TORCH_CONTROL}),
        evidence=ExecutionEvidence(
            path_expectations=(
                _expect("resolved_qkv_layout", "torch_three_contiguous_copies"),
            )
        ),
    ),
    CandidateSpec(
        "eager-auto",
        "auto",
        _always,
        "all validated workload shapes",
        routing_tags=frozenset({RoutingTag.SAFE_FALLBACK}),
    ),
    CandidateSpec(
        "eager-triton",
        "triton",
        _always,
        "all workload shapes as an explicit fallback-detection control",
        hardware_case_support=lambda case: (
            case.dtype == "float16"
            and case.seq_len in {512, 2048}
            and case.d_model // case.num_heads == 64
        ),
        hardware_case_support_description=(
            "FP16 S512/S2048 attention with head dimension 64"
        ),
        capability_tags=_CUDA_TRITON,
        routing_tags=frozenset({RoutingTag.GENERAL_TRITON}),
        evidence=ExecutionEvidence(
            selected_policies=frozenset({"triton", "triton_partial"}),
            requires_observed_execution=True,
            observed_expectations=(
                _observe("qkv_layouts", "triton_single_pass"),
                _observe("attention_backends", "explicit_qk_triton_softmax_pv"),
            ),
        ),
    ),
    CandidateSpec(
        "compile-default",
        "auto",
        _compile_default,
        "short static or wide GEMM-heavy workloads",
        compile_solution=True,
        capability_tags=_CUDA,
        routing_tags=frozenset({RoutingTag.COMPILE_DEFAULT}),
        deployable=False,
    ),
    CandidateSpec(
        "compile-reduce-overhead",
        "auto",
        _short_static,
        "short static workloads",
        compile_solution=True,
        compile_mode="reduce-overhead",
        capability_tags=_CUDA,
        routing_tags=frozenset({RoutingTag.COMPILE_REDUCE_OVERHEAD}),
        deployable=False,
    ),
    CandidateSpec(
        "balanced-cudagraph",
        "balanced-cuda-graph",
        _balanced_graph,
        "B8 S128 D512 H8 F2048 L6 FP16 non-causal static workload",
        capability_tags=_CUDA_GRAPH,
        routing_tags=frozenset(
            {RoutingTag.GRAPH, RoutingTag.SOLUTION_GRAPH, RoutingTag.BALANCED_GRAPH}
        ),
        evidence=ExecutionEvidence(
            requires_observed_execution=True,
            path_expectations=(
                _expect("runtime_wrapper", "solution_eager_cuda_graph"),
                _expect("shape_route", "balanced_fp16_eager_cuda_graph"),
            ),
        ),
    ),
    CandidateSpec(
        "attention-preprocess",
        "preprocess",
        _attention_preprocess,
        "FP16 S64/Dh32 or S2048/Dh64 attention",
        capability_tags=_CUDA_TRITON,
        routing_tags=frozenset({RoutingTag.ATTENTION_PREPROCESS}),
        evidence=ExecutionEvidence(
            requires_observed_execution=True,
            path_expectations=(
                _expect(
                    "resolved_attention",
                    "explicit_qk_triton_preprocess_native_softmax_pv",
                ),
            ),
            observed_expectations=(
                _observe(
                    "attention_backends",
                    "explicit_qk_triton_preprocess_native_softmax_pv",
                ),
            ),
        ),
    ),
    CandidateSpec(
        "s512-native-softmax",
        "s512-native-softmax",
        _s512_family,
        "B8 S512 D512 H8 F2048 L4 FP16 workload",
        capability_tags=_CUDA_TRITON,
        routing_tags=frozenset({RoutingTag.S512_NATIVE_SOFTMAX}),
        evidence=ExecutionEvidence(
            requires_observed_execution=True,
            path_expectations=(
                _expect(
                    "resolved_attention",
                    "explicit_qk_triton_scale_mask_native_half_softmax_pv",
                ),
            ),
            observed_expectations=(
                _observe(
                    "attention_backends",
                    "explicit_qk_triton_scale_mask_native_half_softmax_pv",
                ),
            ),
        ),
    ),
    CandidateSpec(
        "long-tail-online",
        "long-tail-online",
        _long_attention,
        "B1 S2048 D512 H8 F2048 L4 FP16 workload",
        capability_tags=_CUDA_TRITON,
        routing_tags=frozenset({RoutingTag.ATTENTION_ONLINE}),
        evidence=ExecutionEvidence(
            requires_observed_execution=True,
            path_expectations=(
                _expect(
                    "resolved_attention",
                    "three_explicit_layers_tail_online_attention",
                ),
            ),
            observed_expectations=(
                _observe(
                    "attention_backends",
                    "explicit_qk_triton_preprocess_native_softmax_pv",
                    "triton_two_pass_online_attention",
                    required_values=frozenset(
                        {
                            "explicit_qk_triton_preprocess_native_softmax_pv",
                            "triton_two_pass_online_attention",
                        }
                    ),
                ),
            ),
        ),
        minimum_compute_capability=(8, 0),
    ),
    CandidateSpec(
        "padding-fused",
        "padding",
        _padding_route,
        "padded workloads and full/padded route-sharing shape families",
        capability_tags=_CUDA_TRITON,
        routing_tags=frozenset({RoutingTag.PADDING_FUSED}),
        evidence=ExecutionEvidence(
            requires_observed_execution=True,
            path_expectations=(
                _expect(
                    "block_fusion",
                    "triton_residual_add_padding_when_masked",
                ),
            ),
            observed_expectations=(
                _observe("residual_backends", "triton_residual_add_padding"),
            ),
        ),
    ),
    CandidateSpec(
        "padding-packed",
        "packed",
        _padding_route,
        "padded workloads and full/padded route-sharing shape families",
        routing_tags=frozenset({RoutingTag.PADDING_PACKED}),
        evidence=ExecutionEvidence(
            path_expectations=(_expect("padding_route", "packed_valid_token_ffn"),),
            requires_observed_execution=True,
            observed_expectations=(
                _observe("ffn_backends", "packed_valid_token_ffn"),
                _observe("residual_backends", "packed_index_scatter_residual"),
            ),
        ),
    ),
    CandidateSpec(
        "eager-cudagraph",
        "auto",
        _runner_graph,
        "B1 S64 D256 H8 F1024 L4 FP16 non-causal static workload",
        cuda_graph_solution=True,
        capability_tags=_CUDA_GRAPH,
        routing_tags=frozenset({RoutingTag.GRAPH, RoutingTag.RUNNER_GRAPH}),
        deployable=False,
        evidence=ExecutionEvidence(
            path_expectations=(_expect("runtime_wrapper", "eager_cuda_graph"),)
        ),
    ),
    CandidateSpec(
        "launch-cudagraph",
        "cuda-graph",
        _launch_family,
        "B1 S64 D256 H8 F1024 L4 FP16 non-causal static workload",
        capability_tags=_CUDA_GRAPH,
        routing_tags=frozenset({RoutingTag.GRAPH, RoutingTag.SOLUTION_GRAPH}),
        evidence=ExecutionEvidence(
            requires_observed_execution=True,
            path_expectations=(
                _expect("runtime_wrapper", "solution_eager_cuda_graph"),
            ),
        ),
    ),
    CandidateSpec(
        "compile-max-autotune",
        "auto",
        _wide_family,
        "wide GEMM-heavy workloads",
        compile_solution=True,
        compile_mode="max-autotune",
        capability_tags=_CUDA,
        routing_tags=frozenset({RoutingTag.COMPILE_MAX_AUTOTUNE}),
        deployable=False,
    ),
    CandidateSpec(
        "wide-triton-inplace",
        "wide-triton-inplace",
        _wide_exact,
        "B16 S256 D1024 H8 F4096 L6 BF16 non-causal workload",
        capability_tags=_CUDA_TRITON,
        routing_tags=frozenset({RoutingTag.WIDE_INPLACE}),
        evidence=ExecutionEvidence(
            requires_observed_execution=True,
            path_expectations=(
                _expect("resolved_qkv_layout", "triton_single_pass"),
                _expect("resolved_ffn", "torch_inplace_exact_gelu"),
            ),
            observed_expectations=(
                _observe("qkv_layouts", "triton_single_pass"),
                _observe("ffn_backends", "torch_inplace_exact_gelu"),
            ),
        ),
    ),
)


def _build_registry(specs: Sequence[CandidateSpec]) -> Mapping[str, CandidateSpec]:
    registry: dict[str, CandidateSpec] = {}
    active_policy_ids: set[str] = set()
    for spec in specs:
        if spec.candidate_id in registry:
            raise RuntimeError(f"duplicate candidate id: {spec.candidate_id}")
        if spec.deployable:
            if spec.solution_policy in active_policy_ids:
                raise RuntimeError(
                    "multiple active deployable candidates map to policy "
                    f"{spec.solution_policy!r}"
                )
            active_policy_ids.add(spec.solution_policy)
        registry[spec.candidate_id] = spec
    represented_policies = {spec.solution_policy for spec in specs}
    if represented_policies != ROUTABLE_POLICY_IDS:
        missing = ", ".join(sorted(ROUTABLE_POLICY_IDS - represented_policies))
        extra = ", ".join(sorted(represented_policies - ROUTABLE_POLICY_IDS))
        raise RuntimeError(
            "candidate and Solution policy registries disagree; "
            f"missing={missing}; extra={extra}"
        )
    return MappingProxyType(registry)


CANDIDATE_SPECS: Mapping[str, CandidateSpec] = _build_registry(_CANDIDATE_SPECS)


def candidate_spec(candidate_id: str) -> CandidateSpec | None:
    """Return a registered candidate, or ``None`` for an unknown external id."""

    return CANDIDATE_SPECS.get(candidate_id)


def candidate_specs_for_case(
    case: WorkloadCase,
) -> tuple[CandidateSpec, ...]:
    """Return applicable registry entries in stable calibration order."""

    return tuple(spec for spec in _CANDIDATE_SPECS if spec.applies(case))


def candidate_spec_for_policy(
    case: WorkloadCase,
    policy: str,
    *,
    deployable_only: bool = False,
) -> CandidateSpec | None:
    """Resolve one unambiguous candidate for a Solution policy and workload."""

    matches = [
        spec
        for spec in candidate_specs_for_case(case)
        if spec.solution_policy == policy and (not deployable_only or spec.deployable)
    ]
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple candidates map to policy {policy!r} for {case.case_id}"
        )
    return matches[0] if matches else None


def deployable_policy_ids() -> frozenset[str]:
    """Return policies backed by an active, statically dispatchable candidate."""

    return frozenset(
        spec.solution_policy for spec in _CANDIDATE_SPECS if spec.deployable
    )


__all__ = [
    "CANDIDATE_SPECS",
    "CandidateSpec",
    "CapabilityTag",
    "ExecutionEvidence",
    "ObservedPathExpectation",
    "PathExpectation",
    "RoutingTag",
    "candidate_spec",
    "candidate_spec_for_policy",
    "candidate_specs_for_case",
    "deployable_policy_ids",
]
