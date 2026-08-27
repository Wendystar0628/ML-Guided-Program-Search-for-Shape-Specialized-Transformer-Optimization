"""Finite, project-specific candidate screening for the performance mainline."""

from __future__ import annotations

import os
import platform
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Any

from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    WorkloadCase,
    atomic_write_json,
    new_run_id,
    solution_implementation_hash,
    utc_now,
)
from runner.route_promotion import (
    DEPLOYABLE_EAGER_POLICIES,
    select_deployable_winner,
)
from runner.supervisor import run_managed_benchmark

SOLUTION_POLICY_ENV = "TRANSFORMER_OPT_POLICY"
SOLUTION_POLICIES = (
    "dispatch",
    "auto",
    "reference",
    "torch",
    "triton",
    "preprocess",
    "s512-native-softmax",
    "long-pv",
    "long-tail-online",
    "wide-epilogue",
    "wide-triton-inplace",
    "cuda-graph",
    "balanced-cuda-graph",
    "padding",
    "packed",
)


@dataclass(frozen=True)
class TuningCandidate:
    """One bounded combination of model policy and compiler mode."""

    candidate_id: str
    solution_policy: str
    compile_solution: bool = False
    compile_mode: str = "default"
    cuda_graph_solution: bool = False


_EAGER_CANDIDATES = (
    TuningCandidate("eager-reference", "reference"),
    TuningCandidate("eager-torch", "torch"),
    TuningCandidate("eager-auto", "auto"),
    TuningCandidate("eager-triton", "triton"),
)
_COMPILE_DEFAULT = TuningCandidate(
    "compile-default",
    "auto",
    True,
    "default",
)
_COMPILE_REDUCE_OVERHEAD = TuningCandidate(
    "compile-reduce-overhead",
    "auto",
    True,
    "reduce-overhead",
)
_EAGER_CUDAGRAPH = TuningCandidate(
    "eager-cudagraph",
    "auto",
    cuda_graph_solution=True,
)
_SOLUTION_CUDAGRAPH = TuningCandidate("launch-cudagraph", "cuda-graph")
_BALANCED_CUDAGRAPH = TuningCandidate(
    "balanced-cudagraph",
    "balanced-cuda-graph",
)
_PADDING_FUSION_CANDIDATE = TuningCandidate("padding-fused", "padding")
_PADDING_PACKED_CANDIDATE = TuningCandidate("padding-packed", "packed")
_ATTENTION_PREPROCESS_CANDIDATE = TuningCandidate(
    "attention-preprocess",
    "preprocess",
)
_S512_NATIVE_SOFTMAX_CANDIDATE = TuningCandidate(
    "s512-native-softmax",
    "s512-native-softmax",
)
_LONG_TAIL_ONLINE_CANDIDATE = TuningCandidate(
    "long-tail-online",
    "long-tail-online",
)
_WIDE_TRITON_INPLACE_CANDIDATE = TuningCandidate(
    "wide-triton-inplace",
    "wide-triton-inplace",
)
_WIDE_CANDIDATE = TuningCandidate(
    "compile-max-autotune",
    "auto",
    True,
    "max-autotune",
)

_DEVICE_PROFILE_FIELDS = (
    "device_type",
    "device_name",
    "compute_capability",
    "architecture_family",
    "platform_system",
    "torch",
    "cuda_runtime",
    "triton",
    "driver",
)
_ROUTING_PLAN_FIELDS = (
    "source",
    "calibration_stage",
    "decision_scope",
    "requires_full_workload_measurement",
    "bottleneck_class",
    "routing_signals",
    "candidate_order",
    "selection_reasons",
    "capability_rejections",
    "screening_tuning_ids",
)

_ROUTE_VISIBLE_CASE_FIELDS = (
    "dtype",
    "batch_size",
    "seq_len",
    "d_model",
    "num_heads",
    "ffn_dim",
    "num_layers",
    "causal",
)


def _compact_device_profile(
    profile: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Keep only route identity and reproducibility facts in tuning output."""

    if profile is None:
        return None
    return {
        field: profile[field]
        for field in _DEVICE_PROFILE_FIELDS
        if profile.get(field) is not None
    }


def _compact_routing_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Drop workload estimates already recoverable from the stored case."""

    return {field: plan[field] for field in _ROUTING_PLAN_FIELDS if field in plan}


def _installed_triton_version() -> str:
    """Return the local optional Triton version for explicit tuning runs."""

    try:
        import triton
    except Exception:  # noqa: BLE001 - Triton is an optional candidate backend.
        return "unavailable"
    version = getattr(triton, "__version__", None)
    return str(version) if version is not None else "unknown"


@contextmanager
def solution_policy(policy: str) -> Iterator[None]:
    """Temporarily select a Solution execution policy for a serial worker run."""

    if policy not in SOLUTION_POLICIES:
        raise ContractError(f"unsupported solution policy: {policy}")
    previous = os.environ.get(SOLUTION_POLICY_ENV)
    os.environ[SOLUTION_POLICY_ENV] = policy
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(SOLUTION_POLICY_ENV, None)
        else:
            os.environ[SOLUTION_POLICY_ENV] = previous


def candidates_for_case(case: WorkloadCase) -> tuple[TuningCandidate, ...]:
    """Return the small candidate set that is meaningful for one workload case."""

    candidates = list(_EAGER_CANDIDATES)
    if case.seq_len <= 128:
        candidates.extend((_COMPILE_DEFAULT, _COMPILE_REDUCE_OVERHEAD))
    if (
        case.batch_size == 8
        and case.seq_len == 128
        and case.d_model == 512
        and case.num_heads == 8
        and case.ffn_dim == 2048
        and case.num_layers == 6
        and case.dtype == "float16"
        and not case.causal
        and case.padding_ratio == 0
    ):
        candidates.append(_BALANCED_CUDAGRAPH)
    if case.dtype == "float16" and (case.seq_len, case.d_model // case.num_heads) in {
        (64, 32),
        (2048, 64),
    }:
        candidates.append(_ATTENTION_PREPROCESS_CANDIDATE)
    if (
        case.batch_size == 8
        and case.seq_len == 512
        and case.d_model == 512
        and case.num_heads == 8
        and case.ffn_dim == 2048
        and case.num_layers == 4
        and case.dtype == "float16"
    ):
        candidates.append(_S512_NATIVE_SOFTMAX_CANDIDATE)
    if (
        case.batch_size == 1
        and case.seq_len == 2048
        and case.d_model == 512
        and case.num_heads == 8
        and case.dtype == "float16"
    ):
        candidates.append(_LONG_TAIL_ONLINE_CANDIDATE)
    if case.padding_ratio > 0 or case.case_id.startswith("mask_s512_"):
        candidates.extend((_PADDING_FUSION_CANDIDATE, _PADDING_PACKED_CANDIDATE))
    elif (
        case.batch_size == 1
        and case.seq_len == 64
        and case.d_model == 256
        and case.num_heads == 8
        and case.ffn_dim == 1024
        and case.dtype == "float16"
    ):
        candidates.extend(
            (
                _PADDING_FUSION_CANDIDATE,
                _EAGER_CUDAGRAPH,
                _SOLUTION_CUDAGRAPH,
            )
        )
    if case.d_model >= 1024 or case.ffn_dim >= 4096:
        if _COMPILE_DEFAULT not in candidates:
            candidates.append(_COMPILE_DEFAULT)
        candidates.append(_WIDE_CANDIDATE)
        if (
            case.batch_size == 16
            and case.seq_len == 256
            and case.d_model == 1024
            and case.num_heads == 8
            and case.ffn_dim == 4096
            and case.num_layers == 6
            and case.dtype == "bfloat16"
            and not case.causal
        ):
            candidates.extend((_WIDE_TRITON_INPLACE_CANDIDATE,))
    return tuple(candidates)


def is_deployable_candidate(candidate: TuningCandidate) -> bool:
    """Return whether a candidate can be represented by the static dispatcher."""

    return (
        not candidate.compile_solution
        and not candidate.cuda_graph_solution
        and candidate.solution_policy in DEPLOYABLE_EAGER_POLICIES
    )


def _deployable_candidates_by_policy(
    case: WorkloadCase,
    *,
    rejected_ids: set[str] | None = None,
) -> dict[str, TuningCandidate]:
    rejected = rejected_ids or set()
    by_policy: dict[str, TuningCandidate] = {}
    for candidate in candidates_for_case(case):
        if not is_deployable_candidate(candidate) or candidate.candidate_id in rejected:
            continue
        if candidate.solution_policy in by_policy:
            raise ContractError(
                f"multiple deployable candidates map to policy "
                f"{candidate.solution_policy!r} for {case.case_id}"
            )
        by_policy[candidate.solution_policy] = candidate
    return by_policy


def calibration_route_key(case: WorkloadCase) -> tuple[Any, ...]:
    """Return the workload fields visible to one calibrated dispatch route."""

    return tuple(getattr(case, field) for field in _ROUTE_VISIBLE_CASE_FIELDS)


def align_shared_smoke_plans(
    cases: Sequence[WorkloadCase],
    plans: Sequence[Mapping[str, Any]],
    incumbent_candidate_ids: Sequence[str | None],
    *,
    candidate_limit: int,
) -> list[dict[str, Any]]:
    """Give cases sharing one route key a common deployable Smoke shortlist."""

    if len(cases) != len(plans) or len(cases) != len(incumbent_candidate_ids):
        raise ContractError("Smoke plan inputs must have the same length")
    aligned = [dict(plan) for plan in plans]
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, case in enumerate(cases):
        groups.setdefault(calibration_route_key(case), []).append(index)

    for indices in groups.values():
        if len(indices) == 1:
            continue
        policy_candidates: list[dict[str, TuningCandidate]] = []
        ordered_policies: list[list[str]] = []
        for index in indices:
            case = cases[index]
            rejections = aligned[index].get("capability_rejections", {})
            rejected_ids = set(rejections) if isinstance(rejections, Mapping) else set()
            by_policy = _deployable_candidates_by_policy(
                case,
                rejected_ids=rejected_ids,
            )
            by_id = {
                candidate.candidate_id: candidate for candidate in by_policy.values()
            }
            raw_order = aligned[index].get("candidate_order")
            if (
                not isinstance(raw_order, Sequence)
                or isinstance(raw_order, (str, bytes))
            ):
                raise ContractError(f"Smoke plan for {case.case_id} has no order")
            order: list[str] = []
            for candidate_id in raw_order:
                candidate = by_id.get(candidate_id)
                if candidate is not None and candidate.solution_policy not in order:
                    order.append(candidate.solution_policy)
            policy_candidates.append(by_policy)
            ordered_policies.append(order)

        common_policies = set(policy_candidates[0])
        for by_policy in policy_candidates[1:]:
            common_policies.intersection_update(by_policy)
        if "auto" not in common_policies:
            raise ContractError("a shared Smoke route group has no eager-auto control")

        incumbents: list[str] = []
        for index, by_policy in zip(indices, policy_candidates, strict=True):
            incumbent_id = incumbent_candidate_ids[index]
            if incumbent_id is None:
                continue
            candidate = next(
                (
                    item
                    for item in by_policy.values()
                    if item.candidate_id == incumbent_id
                ),
                None,
            )
            if candidate is None:
                raise ContractError(
                    f"incumbent {incumbent_id!r} is unavailable for "
                    f"{cases[index].case_id}"
                )
            if candidate.solution_policy not in incumbents:
                incumbents.append(candidate.solution_policy)
        non_auto_incumbents = [policy for policy in incumbents if policy != "auto"]
        if len(non_auto_incumbents) > 1:
            raise ContractError("cases sharing one route key have different incumbents")

        required_policies = ["auto", *non_auto_incumbents]
        if len(required_policies) > candidate_limit:
            raise ContractError(
                "candidate limit is too small to retain auto and the current incumbent"
            )
        union_order = {
            policy for order in ordered_policies for policy in order
        } & common_policies
        registry_order = list(policy_candidates[0])
        pool = union_order

        def ordinal_score(
            policy: str,
            orders: tuple[tuple[str, ...], ...] = tuple(
                tuple(order) for order in ordered_policies
            ),
            registry: tuple[str, ...] = tuple(registry_order),
        ) -> tuple[int, int, str]:
            ranks: list[int] = []
            for order in orders:
                if policy in order:
                    ranks.append(order.index(policy))
                else:
                    ranks.append(len(order) + registry.index(policy))
            return max(ranks), sum(ranks), policy

        extras = sorted(
            pool - set(required_policies),
            key=ordinal_score,
        )[: candidate_limit - len(required_policies)]
        selected_policies = [*required_policies, *extras]
        for member_offset, index in enumerate(indices):
            by_policy = policy_candidates[member_offset]
            candidate_order = [
                by_policy[policy].candidate_id for policy in selected_policies
            ]
            raw_reasons = aligned[index].get("selection_reasons", {})
            reasons = {
                candidate_id: list(raw_reasons[candidate_id])
                for candidate_id in candidate_order
                if isinstance(raw_reasons, Mapping)
                and isinstance(raw_reasons.get(candidate_id), Sequence)
                and not isinstance(raw_reasons[candidate_id], (str, bytes))
            }
            for candidate_id in candidate_order:
                reasons.setdefault(
                    candidate_id,
                    ["selected jointly for cases sharing one dispatch route"],
                )
            aligned[index]["candidate_order"] = candidate_order
            aligned[index]["selection_reasons"] = reasons
            aligned[index]["decision_scope"] = "shared_route_smoke_shortlist"
    return aligned


def _eligible_smoke_observations(
    case: WorkloadCase,
    summary: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if summary.get("case_id") != case.case_id:
        raise ContractError(f"Smoke summary does not match {case.case_id}")
    protocol = summary.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("preset") != "smoke":
        raise ContractError(f"{case.case_id} finalist selection requires Smoke results")
    if (
        summary.get("complete") is not True
        or summary.get("source_consistent") is not True
        or summary.get("implementation_consistent") is not True
    ):
        raise ContractError(f"Smoke screening is incomplete for {case.case_id}")
    eligible: dict[str, Mapping[str, Any]] = {}
    for policy, candidate in _deployable_candidates_by_policy(case).items():
        try:
            observation = select_deployable_winner(
                summary,
                allowed_policies=frozenset({policy}),
            )
        except ContractError:
            continue
        if observation.get("candidate_id") != candidate.candidate_id:
            raise ContractError(
                f"Smoke observation candidate does not match policy {policy!r}"
            )
        eligible[policy] = observation
    return eligible


def build_formal_candidate_plans(
    cases: Sequence[WorkloadCase],
    smoke_summaries: Sequence[Mapping[str, Any]],
    incumbent_candidate_ids: Sequence[str | None],
) -> list[dict[str, Any]]:
    """Build measured Formal controls and one shared deployable challenger."""

    if (
        len(cases) != len(smoke_summaries)
        or len(cases) != len(incumbent_candidate_ids)
    ):
        raise ContractError("Formal finalist inputs must have the same length")
    implementation_hashes = {
        summary.get("source_implementation_sha256") for summary in smoke_summaries
    }
    source_hashes = {
        summary.get("source_solution_sha256") for summary in smoke_summaries
    }
    implementation_hash = next(iter(implementation_hashes), None)
    source_hash = next(iter(source_hashes), None)
    if (
        len(implementation_hashes) != 1
        or not isinstance(implementation_hash, str)
        or not implementation_hash
        or len(source_hashes) != 1
        or not isinstance(source_hash, str)
        or not source_hash
    ):
        raise ContractError("Smoke cases do not share one Solution implementation")
    eligible_by_case = [
        _eligible_smoke_observations(case, summary)
        for case, summary in zip(cases, smoke_summaries, strict=True)
    ]
    plans: list[dict[str, Any] | None] = [None] * len(cases)
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, case in enumerate(cases):
        groups.setdefault(calibration_route_key(case), []).append(index)

    for indices in groups.values():
        incumbent_policies: list[str] = []
        candidate_maps: list[dict[str, TuningCandidate]] = []
        for index in indices:
            candidates = _deployable_candidates_by_policy(cases[index])
            candidate_maps.append(candidates)
            incumbent_id = incumbent_candidate_ids[index]
            if incumbent_id is not None:
                incumbent = next(
                    (
                        candidate
                        for candidate in candidates.values()
                        if candidate.candidate_id == incumbent_id
                    ),
                    None,
                )
                if incumbent is None:
                    raise ContractError(
                        f"incumbent {incumbent_id!r} is unavailable for "
                        f"{cases[index].case_id}"
                    )
                if incumbent.solution_policy not in incumbent_policies:
                    incumbent_policies.append(incumbent.solution_policy)
        non_auto_incumbents = [
            policy for policy in incumbent_policies if policy != "auto"
        ]
        if len(non_auto_incumbents) > 1:
            raise ContractError("cases sharing one route key have different incumbents")
        control_policies = ["auto", *non_auto_incumbents]
        for index in indices:
            missing = set(control_policies) - set(eligible_by_case[index])
            if missing:
                raise ContractError(
                    f"Smoke controls failed for {cases[index].case_id}: "
                    f"{', '.join(sorted(missing))}"
                )

        common_policies = set(eligible_by_case[indices[0]])
        for index in indices[1:]:
            common_policies.intersection_update(eligible_by_case[index])
        challenger_policies = common_policies - set(control_policies)

        def challenger_rank(
            policy: str,
            group_indices: tuple[int, ...] = tuple(indices),
            controls: tuple[str, ...] = tuple(control_policies),
        ) -> tuple[float, float, float, str]:
            relative_gains: list[float] = []
            p90_values: list[float] = []
            for index in group_indices:
                observations = eligible_by_case[index]
                control_speedup = max(
                    float(observations[control]["conservative_speedup"])
                    for control in controls
                )
                challenger = observations[policy]
                relative_gains.append(
                    float(challenger["conservative_speedup"]) / control_speedup
                )
                p90_values.append(float(challenger["target_p90_ms"]))
            return (
                -min(relative_gains),
                -(sum(relative_gains) / len(relative_gains)),
                max(p90_values),
                policy,
            )

        challenger_policy = (
            min(challenger_policies, key=challenger_rank)
            if challenger_policies
            else None
        )
        selected_policies = [
            *control_policies,
            *([challenger_policy] if challenger_policy is not None else []),
        ]
        screening_ids: list[str] = []
        for index in indices:
            tuning_id = smoke_summaries[index].get("tuning_id")
            if not isinstance(tuning_id, str) or not tuning_id:
                raise ContractError(
                    f"Smoke summary is missing tuning_id for {cases[index].case_id}"
                )
            screening_ids.append(tuning_id)
        for member_offset, index in enumerate(indices):
            by_policy = candidate_maps[member_offset]
            candidate_order = [
                by_policy[policy].candidate_id for policy in selected_policies
            ]
            reasons = {candidate_order[0]: ["required eager-auto control"]}
            if non_auto_incumbents:
                reasons[candidate_order[1]] = ["current calibrated incumbent"]
            if challenger_policy is not None:
                reasons[candidate_order[-1]] = [
                    "best common deployable challenger from Smoke measurements"
                ]
            plans[index] = {
                "source": "smoke_measured_finalists",
                "calibration_stage": "formal",
                "decision_scope": "formal_route_selection",
                "requires_full_workload_measurement": True,
                "candidate_order": candidate_order,
                "selection_reasons": reasons,
                "screening_tuning_ids": screening_ids,
            }
    if any(plan is None for plan in plans):
        raise ContractError("unable to build all Formal finalist plans")
    return [dict(plan) for plan in plans if plan is not None]


def deployable_candidate_id_for_policy(
    case: WorkloadCase,
    policy: str,
) -> str | None:
    """Map one deployed policy back to its eager calibration candidate."""

    for candidate in candidates_for_case(case):
        if (
            candidate.solution_policy == policy
            and not candidate.compile_solution
            and not candidate.cuda_graph_solution
        ):
            return candidate.candidate_id
    return None


def select_candidates(
    case: WorkloadCase,
    requested_ids: Sequence[str] | None,
) -> tuple[TuningCandidate, ...]:
    """Select explicit candidates while rejecting names that do not fit the case."""

    available = {item.candidate_id: item for item in candidates_for_case(case)}
    if not requested_ids:
        return tuple(available.values())
    unknown = sorted(set(requested_ids) - set(available))
    if unknown:
        raise ContractError(
            f"candidates are not available for {case.case_id}: {unknown}; "
            f"available={sorted(available)}"
        )
    return tuple(available[candidate_id] for candidate_id in requested_ids)


def _candidate_protocol(
    base: MeasurementProtocol,
    candidate: TuningCandidate,
) -> MeasurementProtocol:
    protocol = replace(
        base,
        compile_baseline=False,
        compile_solution=candidate.compile_solution,
        cuda_graph_solution=candidate.cuda_graph_solution,
        compile_mode=candidate.compile_mode,
    )
    protocol.validate()
    return protocol


def _observation(
    candidate: TuningCandidate,
    result: dict[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    performance = result.get("performance")
    target = performance.get("target") if isinstance(performance, dict) else None
    baseline = performance.get("baseline") if isinstance(performance, dict) else None
    correctness = result.get("correctness")
    execution_path = result.get("execution_path")
    source = result.get("source")
    requested_policy = (
        execution_path.get("requested_policy")
        if isinstance(execution_path, dict)
        else None
    )
    selected_policy = (
        execution_path.get("selected_policy")
        if isinstance(execution_path, dict)
        else None
    )
    selected_matches = selected_policy == candidate.solution_policy
    if candidate.solution_policy == "triton":
        selected_matches = selected_policy in {"triton", "triton_partial"}
    route_matches = True
    if isinstance(execution_path, dict):
        if candidate.solution_policy == "torch":
            route_matches = (
                execution_path.get("resolved_qkv_layout")
                == "torch_three_contiguous_copies"
            )
        elif candidate.solution_policy == "padding":
            route_matches = (
                execution_path.get("block_fusion")
                == "triton_residual_add_padding_when_masked"
            )
        elif candidate.solution_policy == "packed":
            route_matches = (
                execution_path.get("padding_route") == "packed_valid_token_ffn"
            )
        elif candidate.solution_policy == "reference":
            route_matches = (
                execution_path.get("resolved_attention") == "explicit_reference_order"
            )
        elif candidate.solution_policy == "preprocess":
            route_matches = (
                execution_path.get("resolved_attention")
                == "explicit_qk_triton_preprocess_native_softmax_pv"
            )
        elif candidate.solution_policy == "s512-native-softmax":
            route_matches = (
                execution_path.get("resolved_attention")
                == "explicit_qk_triton_scale_mask_native_half_softmax_pv"
            )
        elif candidate.solution_policy == "long-pv":
            route_matches = (
                execution_path.get("resolved_attention")
                == "explicit_qk_triton_preprocess_native_softmax_triton_pv"
            )
        elif candidate.solution_policy == "long-tail-online":
            route_matches = (
                execution_path.get("resolved_attention")
                == "three_explicit_layers_tail_online_attention"
            )
        elif candidate.solution_policy == "wide-epilogue":
            route_matches = (
                execution_path.get("resolved_ffn") == "cublaslt_tanh_gelu_epilogue"
            )
        elif candidate.solution_policy == "wide-triton-inplace":
            route_matches = (
                execution_path.get("resolved_qkv_layout") == "triton_single_pass"
                and execution_path.get("resolved_ffn") == "torch_inplace_exact_gelu"
            )
        elif candidate.solution_policy == "cuda-graph":
            route_matches = (
                execution_path.get("runtime_wrapper") == "solution_eager_cuda_graph"
            )
        elif candidate.solution_policy == "balanced-cuda-graph":
            route_matches = (
                execution_path.get("runtime_wrapper") == "solution_eager_cuda_graph"
                and execution_path.get("shape_route")
                == "balanced_fp16_eager_cuda_graph"
            )
    policy_applied = (
        requested_policy == candidate.solution_policy
        and selected_matches
        and route_matches
    )
    if candidate.cuda_graph_solution:
        policy_applied = (
            policy_applied
            and isinstance(execution_path, dict)
            and (execution_path.get("runtime_wrapper") == "eager_cuda_graph")
        )
    baseline_rounds = (
        baseline.get("round_medians_ms") if isinstance(baseline, dict) else None
    )
    target_rounds = target.get("round_medians_ms") if isinstance(target, dict) else None
    paired_round_speedups: list[float] | None = None
    if (
        isinstance(baseline_rounds, list)
        and isinstance(target_rounds, list)
        and len(baseline_rounds) == len(target_rounds)
        and baseline_rounds
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(float(value))
            and float(value) > 0
            for value in (*baseline_rounds, *target_rounds)
        )
    ):
        paired_round_speedups = [
            float(baseline_value) / float(target_value)
            for baseline_value, target_value in zip(
                baseline_rounds,
                target_rounds,
                strict=True,
            )
        ]
    conservative_speedup = min(paired_round_speedups) if paired_round_speedups else None
    return {
        "candidate_id": candidate.candidate_id,
        "solution_policy": candidate.solution_policy,
        "compile_solution": candidate.compile_solution,
        "compile_mode": candidate.compile_mode,
        "cuda_graph_solution": candidate.cuda_graph_solution,
        "outcome": result.get("outcome"),
        "correctness_passed": (
            correctness.get("passed") if isinstance(correctness, dict) else None
        ),
        "failed_elements": (
            correctness.get("failed_elements")
            if isinstance(correctness, dict)
            else None
        ),
        "max_abs_error": (
            correctness.get("max_abs_error") if isinstance(correctness, dict) else None
        ),
        "baseline_median_ms": (
            baseline.get("median_ms") if isinstance(baseline, dict) else None
        ),
        "target_median_ms": (
            target.get("median_ms") if isinstance(target, dict) else None
        ),
        "target_p90_ms": target.get("p90_ms") if isinstance(target, dict) else None,
        "baseline_round_medians_ms": baseline_rounds,
        "target_round_medians_ms": target_rounds,
        "paired_round_speedups": paired_round_speedups,
        "conservative_speedup": conservative_speedup,
        "speedup": (
            performance.get("speedup") if isinstance(performance, dict) else None
        ),
        "policy_applied": policy_applied,
        "run_id": result.get("run_id"),
        "solution_sha256": (
            source.get("solution_sha256") if isinstance(source, dict) else None
        ),
        "execution_path": execution_path,
        "result_path": str(result_path),
    }


def run_tuning_case(
    project_root: Path,
    *,
    workload_set_id: str,
    workload_sha256: str,
    case: WorkloadCase,
    base_protocol: MeasurementProtocol,
    device: str,
    requested_candidates: Sequence[str] | None = None,
    routing_plan: Mapping[str, Any] | None = None,
    device_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run candidates serially and persist one compact screening summary."""

    observations: list[dict[str, Any]] = []
    tuning_id = new_run_id()
    compact_device_profile = _compact_device_profile(device_profile)
    solution_root = project_root / "solution"
    implementation_hash_before = (
        solution_implementation_hash(solution_root) if solution_root.is_dir() else None
    )
    candidates = select_candidates(case, requested_candidates)
    for candidate in candidates:
        protocol = _candidate_protocol(base_protocol, candidate)
        with solution_policy(candidate.solution_policy):
            result, result_path = run_managed_benchmark(
                project_root,
                workload_set_id=workload_set_id,
                case=case,
                protocol=protocol,
                device=device,
                target="solution",
                workload_sha256=workload_sha256,
                sweep_id=tuning_id,
                tuning_id=tuning_id,
                candidate_id=candidate.candidate_id,
            )
        observations.append(_observation(candidate, result, result_path))
        environment = result.get("environment")
        if compact_device_profile is None and isinstance(environment, dict):
            resolved_device = environment.get("device")
            compact_device_profile = {
                "device_type": (
                    resolved_device.split(":", maxsplit=1)[0]
                    if isinstance(resolved_device, str)
                    else None
                ),
                "device_name": environment.get("gpu"),
                "compute_capability": environment.get("compute_capability"),
                "platform_system": platform.system(),
                "torch": environment.get("torch"),
                "cuda_runtime": environment.get("cuda_runtime"),
                "triton": _installed_triton_version(),
                "driver": environment.get("driver"),
            }
        if result.get("outcome") == "cancelled":
            break

    eligible = [
        item
        for item in observations
        if item["outcome"] == "success"
        and item["correctness_passed"] is True
        and item["policy_applied"] is True
        and isinstance(item["speedup"], (int, float))
        and isfinite(float(item["speedup"]))
        and float(item["speedup"]) > 0
        and isinstance(item["target_median_ms"], (int, float))
        and isfinite(float(item["target_median_ms"]))
        and float(item["target_median_ms"]) > 0
    ]
    solution_hashes = {
        item["solution_sha256"]
        for item in observations
        if isinstance(item.get("solution_sha256"), str)
    }
    source_consistent = len(solution_hashes) == 1 and all(
        item.get("solution_sha256") in solution_hashes for item in observations
    )
    implementation_hash_after = (
        solution_implementation_hash(solution_root) if solution_root.is_dir() else None
    )
    implementation_consistent = (
        implementation_hash_before is not None
        and implementation_hash_before == implementation_hash_after
    )
    if not source_consistent:
        eligible = []

    def ranking_speedup(item: dict[str, Any]) -> float:
        conservative = item.get("conservative_speedup")
        return float(
            conservative if isinstance(conservative, (int, float)) else item["speedup"]
        )

    winner = max(eligible, key=ranking_speedup) if eligible else None
    deployable = [
        item
        for item in eligible
        if item["compile_solution"] is False and item["cuda_graph_solution"] is False
    ]
    deployable_winner = max(deployable, key=ranking_speedup) if deployable else None
    summary = {
        "schema_version": 1,
        "tuning_id": tuning_id,
        "created_at": utc_now(),
        "complete": len(observations) == len(candidates)
        and all(item.get("outcome") != "cancelled" for item in observations),
        "workload": {
            "set_id": workload_set_id,
            "sha256": workload_sha256,
            "case": case.as_dict(),
        },
        "requested_device": device,
        "device_profile": compact_device_profile,
        "routing_plan": (
            _compact_routing_plan(routing_plan)
            if routing_plan is not None
            else {
                "source": (
                    "explicit_candidates"
                    if requested_candidates is not None
                    else "fixed_candidate_set"
                ),
                "candidate_order": [item.candidate_id for item in candidates],
            }
        ),
        "protocol": base_protocol.as_dict(),
        "source_solution_sha256": (
            next(iter(solution_hashes)) if len(solution_hashes) == 1 else None
        ),
        "source_consistent": source_consistent,
        "source_implementation_sha256": (
            implementation_hash_before if implementation_consistent else None
        ),
        "implementation_consistent": implementation_consistent,
        "case_id": case.case_id,
        "winner_basis": "full_transformer_correctness_and_paired_timing",
        "observations": observations,
        "winner": winner,
        "deployable_winner": deployable_winner,
    }
    summary_path = project_root / "results" / "tuning" / f"{tuning_id}.json"
    summary["summary_path"] = str(summary_path)
    atomic_write_json(summary_path, summary)
    return summary
