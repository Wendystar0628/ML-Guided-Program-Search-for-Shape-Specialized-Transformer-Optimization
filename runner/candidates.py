"""Finite optimization candidates for the official causal Transformer shapes.

This registry is the runner-side truth for candidate identity, applicability,
hardware requirements, deployment eligibility, and execution evidence. It is
deliberately small: calibration compares complete execution strategies rather
than exposing every internal implementation detail as a candidate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from runner.contracts import RunVariant, TransformerShape
from solution.policies import ROUTABLE_POLICY_IDS


class CapabilityTag(StrEnum):
    """Runtime capabilities required before measuring a candidate."""

    CUDA = "cuda"
    CUDA_GRAPH = "cuda_graph"


class RoutingTag(StrEnum):
    """Cost-model features used to rank candidates on unseen hardware."""

    AUTO_CONTROL = "auto_control"
    SAFE_FALLBACK = "safe_fallback"
    CAUSAL_SDPA = "causal_sdpa"
    GRAPH = "graph"
    BATCH_TILED = "batch_tiled"
    INPLACE_BLOCK = "inplace_block"


@dataclass(frozen=True)
class PathExpectation:
    """One accepted value for a field in the reported execution path."""

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
    requires_observed_execution: bool = False
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
        if not (self.requires_observed_execution or self.observed_expectations):
            return True
        observed = execution_path.get("observed_execution")
        if not isinstance(observed, Mapping):
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
    """Complete runner contract for one bounded optimization strategy."""

    candidate_id: str
    solution_policy: str
    applies_to: Applicability
    applicability_description: str
    hardware_support: Applicability | None = None
    hardware_support_description: str | None = None
    capability_tags: frozenset[CapabilityTag] = frozenset()
    routing_tags: frozenset[RoutingTag] = frozenset()
    deployable: bool = True
    evidence: ExecutionEvidence = ExecutionEvidence()
    minimum_compute_capability: tuple[int, int] | None = None

    def applies(self, shape: TransformerShape, variant: RunVariant) -> bool:
        return self.applies_to(shape, variant)

    def supports_on_hardware(
        self,
        shape: TransformerShape,
        variant: RunVariant,
    ) -> bool:
        return self.hardware_support is None or self.hardware_support(shape, variant)

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


def _official_causal(shape: TransformerShape, variant: RunVariant) -> bool:
    return shape.causal and variant.padding_ratio == 0


def _graph_candidate(shape: TransformerShape, variant: RunVariant) -> bool:
    # Graph capture is useful for repeated static launch-heavy work, but the
    # extreme-batch and long-sequence cases have poor replay benefit relative
    # to their capture footprint.
    return (
        _official_causal(shape, variant)
        and shape.seq_len <= 128
        and shape.batch_size <= 128
    )


def _batch_tiled_candidate(shape: TransformerShape, variant: RunVariant) -> bool:
    return _official_causal(shape, variant) and shape.batch_size >= 1024


def _expect(field: str, *values: object) -> PathExpectation:
    return PathExpectation(field, frozenset(values))


def _observe(field: str, value: str) -> ObservedPathExpectation:
    return ObservedPathExpectation(
        field,
        frozenset({value}),
        frozenset({value}),
    )


_CUDA = frozenset({CapabilityTag.CUDA})
_CUDA_GRAPH = frozenset({CapabilityTag.CUDA, CapabilityTag.CUDA_GRAPH})


_CANDIDATE_SPECS = (
    CandidateSpec(
        "eager-auto",
        "auto",
        _official_causal,
        "official causal shapes",
        capability_tags=_CUDA,
        routing_tags=frozenset({RoutingTag.AUTO_CONTROL}),
    ),
    CandidateSpec(
        "eager-safe",
        "safe",
        _official_causal,
        "official causal shapes",
        capability_tags=_CUDA,
        routing_tags=frozenset({RoutingTag.SAFE_FALLBACK}),
        evidence=ExecutionEvidence(
            requires_observed_execution=True,
            path_expectations=(
                _expect("attention_backend", "safe_streaming"),
                _expect("runtime_wrapper", "eager"),
                _expect("batch_strategy", "full"),
                _expect("block_backend", "torch"),
            ),
            observed_expectations=(
                _observe("attention_backends", "safe_streaming"),
                _observe("block_backends", "torch"),
            ),
        ),
    ),
    CandidateSpec(
        "causal-sdpa",
        "causal-sdpa",
        _official_causal,
        "official causal shapes supported by the fused SDPA backend",
        capability_tags=_CUDA,
        routing_tags=frozenset({RoutingTag.CAUSAL_SDPA}),
        evidence=ExecutionEvidence(
            requires_observed_execution=True,
            path_expectations=(
                _expect("attention_backend", "causal_sdpa"),
                _expect("runtime_wrapper", "eager"),
                _expect("batch_strategy", "full"),
                _expect("block_backend", "torch"),
            ),
            observed_expectations=(
                _observe("attention_backends", "causal_sdpa"),
                _observe("block_backends", "torch"),
            ),
        ),
    ),
    CandidateSpec(
        "graph",
        "graph",
        _graph_candidate,
        "static causal shapes with B<=128 and S<=128",
        capability_tags=_CUDA_GRAPH,
        routing_tags=frozenset({RoutingTag.CAUSAL_SDPA, RoutingTag.GRAPH}),
        evidence=ExecutionEvidence(
            requires_observed_execution=True,
            path_expectations=(
                _expect("attention_backend", "causal_sdpa"),
                _expect("runtime_wrapper", "cuda_graph"),
                _expect("batch_strategy", "full"),
                _expect("block_backend", "torch"),
            ),
            observed_expectations=(
                _observe("attention_backends", "causal_sdpa"),
                _observe("block_backends", "torch"),
                _observe("runtime_wrappers", "cuda_graph"),
            ),
        ),
    ),
    CandidateSpec(
        "batch-tiled",
        "batch-tiled",
        _batch_tiled_candidate,
        "causal shapes with batch size at least 1024",
        capability_tags=_CUDA,
        routing_tags=frozenset({RoutingTag.CAUSAL_SDPA, RoutingTag.BATCH_TILED}),
        evidence=ExecutionEvidence(
            requires_observed_execution=True,
            path_expectations=(
                _expect("attention_backend", "causal_sdpa"),
                _expect("runtime_wrapper", "eager"),
                _expect("batch_strategy", "tiled"),
                _expect("block_backend", "torch"),
            ),
            observed_expectations=(
                _observe("attention_backends", "causal_sdpa"),
                _observe("block_backends", "torch"),
            ),
        ),
    ),
    CandidateSpec(
        "inplace-block",
        "inplace-block",
        _official_causal,
        "official causal shapes using exact in-place GELU",
        capability_tags=_CUDA,
        routing_tags=frozenset({RoutingTag.CAUSAL_SDPA, RoutingTag.INPLACE_BLOCK}),
        evidence=ExecutionEvidence(
            requires_observed_execution=True,
            path_expectations=(
                _expect("attention_backend", "causal_sdpa"),
                _expect("runtime_wrapper", "eager"),
                _expect("batch_strategy", "full"),
                _expect("block_backend", "inplace_exact_gelu"),
            ),
            observed_expectations=(
                _observe("attention_backends", "causal_sdpa"),
                _observe("block_backends", "inplace_exact_gelu"),
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
                    "multiple deployable candidates map to policy "
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
    return CANDIDATE_SPECS.get(candidate_id)


def candidate_specs_for_shape(
    shape: TransformerShape,
    variant: RunVariant,
) -> tuple[CandidateSpec, ...]:
    """Return applicable candidates in stable calibration order."""

    return tuple(spec for spec in _CANDIDATE_SPECS if spec.applies(shape, variant))


def candidate_spec_for_policy(
    shape: TransformerShape,
    variant: RunVariant,
    policy: str,
    *,
    deployable_only: bool = False,
) -> CandidateSpec | None:
    """Resolve one unambiguous candidate for a policy and shape variant."""

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
    "candidate_specs_for_shape",
    "deployable_policy_ids",
]
