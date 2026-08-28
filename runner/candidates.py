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

    @property
    def required_components(self) -> frozenset[ExecutionComponent]:
        """Derive runtime requirements from the policy truth source."""

        return get_policy_spec(self.solution_policy).required_components

    @property
    def deployable(self) -> bool:
        """Safe is an explicit diagnostic path, not a route target."""

        return get_policy_spec(self.solution_policy).routable

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
    runtime_wrapper: str = "eager",
    block_backend: str = "torch",
) -> ExecutionEvidence:
    observed = [
        _observe("attention_backends", "causal_sdpa"),
        _observe("block_backends", block_backend),
    ]
    if runtime_wrapper == "cuda_graph":
        observed.append(_observe("runtime_wrappers", "cuda_graph"))
    return ExecutionEvidence(
        selected_policies=frozenset({policy}),
        path_expectations=(
            _expect("attention_backend", "causal_sdpa"),
            _expect("runtime_wrapper", runtime_wrapper),
            _expect("block_backend", block_backend),
        ),
        observed_expectations=tuple(observed),
    )


_CANDIDATE_SPECS = (
    CandidateSpec(
        "eager-auto",
        "auto",
        _native_sdpa_candidate,
        "causal FP16/FP32 shapes without padding",
        _native_evidence(policy="auto"),
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
                _expect("block_backend", "torch"),
            ),
            observed_expectations=(
                _observe("attention_backends", "safe_streaming"),
                _observe("block_backends", "torch"),
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
        "inplace-block",
        "inplace-block",
        _native_sdpa_candidate,
        "causal FP16/FP32 shapes without padding",
        _native_evidence(
            policy="inplace-block",
            block_backend="inplace_exact_gelu",
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


__all__ = [
    "CANDIDATE_SPECS",
    "CandidateSpec",
    "ExecutionEvidence",
    "ObservedPathExpectation",
    "PathExpectation",
    "candidate_spec",
    "candidate_spec_for_policy",
    "candidate_specs_for_shape",
    "deployable_policy_ids",
]
