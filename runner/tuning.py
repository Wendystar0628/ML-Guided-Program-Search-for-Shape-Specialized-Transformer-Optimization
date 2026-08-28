"""Finite, project-specific candidate screening for the performance mainline."""

from __future__ import annotations

import platform
from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import isfinite
from pathlib import Path
from typing import Any

from project_identity import solution_implementation_hash
from runner.candidates import (
    CandidateSpec,
    candidate_spec,
    candidate_spec_for_policy,
    candidate_specs_for_execution_mode,
    candidate_specs_for_shape,
)
from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    RunVariant,
    TransformerShape,
    atomic_write_json,
    new_run_id,
    utc_now,
)
from runner.locking import device_measurement_lease
from runner.routing_contracts import workload_route_identity
from runner.supervisor import CancellationToken, run_managed_benchmark
from runner.tuning_contracts import (
    TUNING_SCHEMA_VERSION,
    observation_latency_key,
    select_deployable_winner,
)
from runner.workload_execution import plan_workload_execution

_DEVICE_PROFILE_FIELDS = (
    "device_type",
    "device_name",
    "compute_capability",
    "architecture_family",
    "platform_system",
    "torch",
    "cuda_runtime",
    "driver",
    "matmul_precision",
    "allow_tf32",
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
    """Drop workload estimates recoverable from the stored shape and variant."""

    return {field: plan[field] for field in _ROUTING_PLAN_FIELDS if field in plan}


def candidates_for_shape(
    shape: TransformerShape,
    variant: RunVariant,
) -> tuple[CandidateSpec, ...]:
    """Return candidates supported by the shape's selected executor."""

    if not isinstance(shape, TransformerShape):
        raise TypeError("shape must be a TransformerShape")
    if not isinstance(variant, RunVariant):
        raise TypeError("variant must be a RunVariant")
    shape.validate()
    variant.validate()
    execution_mode = plan_workload_execution(shape, variant).execution_mode
    if execution_mode == "resident":
        return candidate_specs_for_shape(shape, variant)
    return candidate_specs_for_execution_mode(shape, variant, execution_mode)


def is_deployable_candidate(candidate: CandidateSpec) -> bool:
    """Return whether a candidate can be represented by the static dispatcher."""

    spec = candidate_spec(candidate.candidate_id)
    return bool(spec is not None and spec.exact_route_eligible and spec == candidate)


def _deployable_candidates_by_policy(
    shape: TransformerShape,
    variant: RunVariant,
    *,
    rejected_ids: set[str] | None = None,
) -> dict[str, CandidateSpec]:
    rejected = rejected_ids or set()
    by_policy: dict[str, CandidateSpec] = {}
    for candidate in candidates_for_shape(shape, variant):
        if not is_deployable_candidate(candidate) or candidate.candidate_id in rejected:
            continue
        if candidate.solution_policy in by_policy:
            raise ContractError(
                f"multiple deployable candidates map to policy "
                f"{candidate.solution_policy!r} for {shape.case_id}"
            )
        by_policy[candidate.solution_policy] = candidate
    return by_policy


def calibration_route_key(
    shape: TransformerShape,
    variant: RunVariant,
) -> tuple[Any, ...]:
    """Return the workload fields visible to one calibrated dispatch route."""

    return workload_route_identity(shape, variant)


def align_shared_smoke_plans(
    shapes: Sequence[TransformerShape],
    variant: RunVariant,
    plans: Sequence[Mapping[str, Any]],
    incumbent_candidate_ids: Sequence[str | None],
    *,
    candidate_limit: int,
) -> list[dict[str, Any]]:
    """Give cases sharing one route key a common deployable Smoke shortlist."""

    if len(shapes) != len(plans) or len(shapes) != len(incumbent_candidate_ids):
        raise ContractError("Smoke plan inputs must have the same length")
    aligned = [dict(plan) for plan in plans]
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, shape in enumerate(shapes):
        groups.setdefault(calibration_route_key(shape, variant), []).append(index)

    for indices in groups.values():
        if len(indices) == 1:
            continue
        policy_candidates: list[dict[str, CandidateSpec]] = []
        ordered_policies: list[list[str]] = []
        for index in indices:
            shape = shapes[index]
            rejections = aligned[index].get("capability_rejections", {})
            rejected_ids = set(rejections) if isinstance(rejections, Mapping) else set()
            by_policy = _deployable_candidates_by_policy(
                shape,
                variant,
                rejected_ids=rejected_ids,
            )
            by_id = {
                candidate.candidate_id: candidate for candidate in by_policy.values()
            }
            raw_order = aligned[index].get("candidate_order")
            if not isinstance(raw_order, Sequence) or isinstance(
                raw_order, (str, bytes)
            ):
                raise ContractError(f"Smoke plan for {shape.case_id} has no order")
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
        if "eager-sdpa" not in common_policies:
            raise ContractError("a shared Smoke route group has no eager-sdpa control")

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
                    f"{shapes[index].case_id}"
                )
            if candidate.solution_policy not in incumbents:
                incumbents.append(candidate.solution_policy)
        non_default_incumbents = [
            policy for policy in incumbents if policy != "eager-sdpa"
        ]
        if len(non_default_incumbents) > 1:
            raise ContractError("cases sharing one route key have different incumbents")

        required_policies = ["eager-sdpa", *non_default_incumbents]
        if len(required_policies) > candidate_limit:
            raise ContractError(
                "candidate limit is too small to retain eager-sdpa and the "
                "current incumbent"
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
    shape: TransformerShape,
    variant: RunVariant,
    summary: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if summary.get("case_id") != shape.case_id:
        raise ContractError(f"Smoke summary does not match {shape.case_id}")
    protocol = summary.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("preset") != "smoke":
        raise ContractError(
            f"{shape.case_id} finalist selection requires Smoke results"
        )
    if (
        summary.get("complete") is not True
        or summary.get("source_consistent") is not True
        or summary.get("implementation_consistent") is not True
        or summary.get("official_consistent") is not True
    ):
        raise ContractError(f"Smoke screening is incomplete for {shape.case_id}")
    eligible: dict[str, Mapping[str, Any]] = {}
    for policy, candidate in _deployable_candidates_by_policy(
        shape,
        variant,
    ).items():
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
    shapes: Sequence[TransformerShape],
    variant: RunVariant,
    smoke_summaries: Sequence[Mapping[str, Any]],
    incumbent_candidate_ids: Sequence[str | None],
) -> list[dict[str, Any]]:
    """Retain every shared candidate that passed Smoke for Formal measurement."""

    if len(shapes) != len(smoke_summaries) or len(shapes) != len(
        incumbent_candidate_ids
    ):
        raise ContractError("Formal finalist inputs must have the same length")
    implementation_hashes = {
        summary.get("source_implementation_sha256") for summary in smoke_summaries
    }
    source_hashes = {
        summary.get("source_solution_sha256") for summary in smoke_summaries
    }
    official_hashes = {
        summary.get("official_snapshot_sha256") for summary in smoke_summaries
    }
    implementation_hash = next(iter(implementation_hashes), None)
    source_hash = next(iter(source_hashes), None)
    official_hash = next(iter(official_hashes), None)
    if (
        len(implementation_hashes) != 1
        or not isinstance(implementation_hash, str)
        or not implementation_hash
        or len(source_hashes) != 1
        or not isinstance(source_hash, str)
        or not source_hash
        or len(official_hashes) != 1
        or not isinstance(official_hash, str)
        or not official_hash
    ):
        raise ContractError(
            "Smoke cases do not share one Solution and official snapshot"
        )
    eligible_by_case = [
        _eligible_smoke_observations(shape, variant, summary)
        for shape, summary in zip(shapes, smoke_summaries, strict=True)
    ]
    plans: list[dict[str, Any] | None] = [None] * len(shapes)
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, shape in enumerate(shapes):
        groups.setdefault(calibration_route_key(shape, variant), []).append(index)

    for indices in groups.values():
        incumbent_policies: list[str] = []
        candidate_maps: list[dict[str, CandidateSpec]] = []
        for index in indices:
            candidates = _deployable_candidates_by_policy(shapes[index], variant)
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
                        f"{shapes[index].case_id}"
                    )
                if incumbent.solution_policy not in incumbent_policies:
                    incumbent_policies.append(incumbent.solution_policy)
        non_default_incumbents = [
            policy for policy in incumbent_policies if policy != "eager-sdpa"
        ]
        if len(non_default_incumbents) > 1:
            raise ContractError("cases sharing one route key have different incumbents")
        control_policies = ["eager-sdpa", *non_default_incumbents]
        for index in indices:
            missing = set(control_policies) - set(eligible_by_case[index])
            if missing:
                raise ContractError(
                    f"Smoke controls failed for {shapes[index].case_id}: "
                    f"{', '.join(sorted(missing))}"
                )

        common_policies = set(eligible_by_case[indices[0]])
        for index in indices[1:]:
            common_policies.intersection_update(eligible_by_case[index])
        challenger_policies = [
            policy
            for policy in candidate_maps[0]
            if policy in common_policies and policy not in control_policies
        ]
        selected_policies = [
            *control_policies,
            *challenger_policies,
        ]
        screening_ids: list[str] = []
        for index in indices:
            tuning_id = smoke_summaries[index].get("tuning_id")
            if not isinstance(tuning_id, str) or not tuning_id:
                raise ContractError(
                    f"Smoke summary is missing tuning_id for {shapes[index].case_id}"
                )
            screening_ids.append(tuning_id)
        for member_offset, index in enumerate(indices):
            by_policy = candidate_maps[member_offset]
            candidate_order = [
                by_policy[policy].candidate_id for policy in selected_policies
            ]
            reasons = {candidate_order[0]: ["required eager-sdpa control"]}
            if non_default_incumbents:
                reasons[candidate_order[1]] = ["current calibrated incumbent"]
            for policy in challenger_policies:
                reasons[by_policy[policy].candidate_id] = [
                    "passed Smoke and retained for Formal measurement"
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
    shape: TransformerShape,
    variant: RunVariant,
    policy: str,
) -> str | None:
    """Map one deployed policy back to its eager calibration candidate."""

    spec = candidate_spec_for_policy(
        shape,
        variant,
        policy,
        deployable_only=True,
    )
    return spec.candidate_id if spec is not None and spec.exact_route_eligible else None


def select_candidates(
    shape: TransformerShape,
    variant: RunVariant,
    requested_ids: Sequence[str],
) -> tuple[CandidateSpec, ...]:
    """Select explicit candidates that fit the requested shape variant."""

    available = {
        item.candidate_id: item for item in candidates_for_shape(shape, variant)
    }
    if not requested_ids:
        raise ContractError("at least one explicit tuning candidate is required")
    unknown = sorted(set(requested_ids) - set(available))
    if unknown:
        raise ContractError(
            f"candidates are not available for {shape.case_id}: {unknown}; "
            f"available={sorted(available)}"
        )
    return tuple(available[candidate_id] for candidate_id in requested_ids)


def _tuning_protocol(base: MeasurementProtocol) -> MeasurementProtocol:
    """Use one uncompiled protocol for every deployable runtime policy."""

    protocol = replace(
        base,
        compile_baseline=False,
        compile_solution=False,
    )
    protocol.validate()
    return protocol


def _observation(
    candidate: CandidateSpec,
    result: dict[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    performance = result.get("performance")
    target = performance.get("target") if isinstance(performance, dict) else None
    baseline = performance.get("baseline") if isinstance(performance, dict) else None
    correctness = result.get("correctness")
    execution_path = result.get("execution_path")
    source = result.get("source")
    spec = candidate_spec(candidate.candidate_id)
    policy_applied = bool(
        spec is not None
        and spec == candidate
        and isinstance(execution_path, Mapping)
        and spec.evidence_matches(execution_path)
    )
    return {
        "candidate_id": candidate.candidate_id,
        "solution_policy": candidate.solution_policy,
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
        "speedup": (
            performance.get("speedup") if isinstance(performance, dict) else None
        ),
        "policy_applied": policy_applied,
        "run_id": result.get("run_id"),
        "solution_sha256": (
            source.get("solution_sha256") if isinstance(source, dict) else None
        ),
        "official_snapshot_sha256": (
            source.get("official_sha256") if isinstance(source, dict) else None
        ),
        "execution_path": execution_path,
        "result_path": str(result_path),
    }


def run_tuning_case(
    project_root: Path,
    *,
    workload_set_id: str,
    workload_sha256: str,
    shape: TransformerShape,
    variant: RunVariant,
    base_protocol: MeasurementProtocol,
    device: str,
    requested_candidates: Sequence[str],
    routing_plan: Mapping[str, Any] | None = None,
    device_profile: Mapping[str, Any] | None = None,
    cancellation_token: CancellationToken | None = None,
) -> dict[str, Any]:
    with device_measurement_lease(
        project_root,
        device,
        purpose=f"candidate tuning for {shape.case_id}",
    ):
        return _run_tuning_case(
            project_root,
            workload_set_id=workload_set_id,
            workload_sha256=workload_sha256,
            shape=shape,
            variant=variant,
            base_protocol=base_protocol,
            device=device,
            requested_candidates=requested_candidates,
            routing_plan=routing_plan,
            device_profile=device_profile,
            cancellation_token=cancellation_token,
        )


def _run_tuning_case(
    project_root: Path,
    *,
    workload_set_id: str,
    workload_sha256: str,
    shape: TransformerShape,
    variant: RunVariant,
    base_protocol: MeasurementProtocol,
    device: str,
    requested_candidates: Sequence[str],
    routing_plan: Mapping[str, Any] | None = None,
    device_profile: Mapping[str, Any] | None = None,
    cancellation_token: CancellationToken | None = None,
) -> dict[str, Any]:
    """Run candidates serially and persist one compact screening summary."""

    observations: list[dict[str, Any]] = []
    tuning_id = new_run_id()
    tuning_directory = project_root.resolve() / "results" / "tuning" / tuning_id
    runs_directory = tuning_directory / "runs"
    compact_device_profile = _compact_device_profile(device_profile)
    solution_root = project_root / "solution"
    implementation_hash_before = (
        solution_implementation_hash(solution_root) if solution_root.is_dir() else None
    )
    candidates = select_candidates(shape, variant, requested_candidates)
    protocol = _tuning_protocol(base_protocol)
    for candidate in candidates:
        result, result_path = run_managed_benchmark(
            project_root,
            workload_set_id=workload_set_id,
            shape=shape,
            variant=variant,
            protocol=protocol,
            device=device,
            target="solution",
            workload_sha256=workload_sha256,
            sweep_id=tuning_id,
            tuning_id=tuning_id,
            candidate_id=candidate.candidate_id,
            solution_policy=candidate.solution_policy,
            cancellation_token=cancellation_token,
            result_dir=runs_directory,
        )
        observation = _observation(candidate, result, result_path)
        observation["result_path"] = f"runs/{result_path.name}"
        observations.append(observation)
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
                "driver": environment.get("driver"),
                "matmul_precision": protocol.matmul_precision,
                "allow_tf32": protocol.allow_tf32,
            }
        if result.get("outcome") == "cancelled":
            break

    eligible = [
        item
        for item in observations
        if item["outcome"] == "success"
        and item["correctness_passed"] is True
        and item["policy_applied"] is True
        and isinstance(item["target_median_ms"], (int, float))
        and isfinite(float(item["target_median_ms"]))
        and float(item["target_median_ms"]) > 0
        and isinstance(item["target_p90_ms"], (int, float))
        and isfinite(float(item["target_p90_ms"]))
        and float(item["target_p90_ms"]) > 0
    ]
    solution_hashes = {
        item["solution_sha256"]
        for item in observations
        if isinstance(item.get("solution_sha256"), str)
    }
    source_consistent = len(solution_hashes) == 1 and all(
        item.get("solution_sha256") in solution_hashes for item in observations
    )
    official_hashes = {
        item["official_snapshot_sha256"]
        for item in observations
        if isinstance(item.get("official_snapshot_sha256"), str)
    }
    official_consistent = len(official_hashes) == 1 and all(
        item.get("official_snapshot_sha256") in official_hashes for item in observations
    )
    implementation_hash_after = (
        solution_implementation_hash(solution_root) if solution_root.is_dir() else None
    )
    implementation_consistent = (
        implementation_hash_before is not None
        and implementation_hash_before == implementation_hash_after
    )
    if not source_consistent or not official_consistent:
        eligible = []

    winner = min(eligible, key=observation_latency_key) if eligible else None
    deployable_candidate_ids = {
        candidate.candidate_id
        for candidate in candidates
        if candidate.exact_route_eligible
    }
    deployable = [
        item for item in eligible if item["candidate_id"] in deployable_candidate_ids
    ]
    deployable_winner = (
        min(deployable, key=observation_latency_key) if deployable else None
    )
    summary = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "tuning_id": tuning_id,
        "created_at": utc_now(),
        "complete": len(observations) == len(candidates)
        and all(item.get("outcome") != "cancelled" for item in observations),
        "workload": {
            "set_id": workload_set_id,
            "sha256": workload_sha256,
            "shape": shape.as_dict(),
            "variant": variant.as_dict(),
        },
        "requested_device": device,
        "device_profile": compact_device_profile,
        "routing_plan": (
            _compact_routing_plan(routing_plan)
            if routing_plan is not None
            else {
                "source": "explicit_candidates",
                "candidate_order": [item.candidate_id for item in candidates],
            }
        ),
        "protocol": protocol.as_dict(),
        "source_solution_sha256": (
            next(iter(solution_hashes)) if len(solution_hashes) == 1 else None
        ),
        "source_consistent": source_consistent,
        "official_snapshot_sha256": (
            next(iter(official_hashes)) if official_consistent else None
        ),
        "official_consistent": official_consistent,
        "source_implementation_sha256": (
            implementation_hash_before if implementation_consistent else None
        ),
        "implementation_consistent": implementation_consistent,
        "case_id": shape.case_id,
        "winner_basis": "target_median_ms_then_p90_then_candidate_id",
        "observations": observations,
        "winner": winner,
        "deployable_winner": deployable_winner,
    }
    summary_path = tuning_directory / "summary.json"
    summary["summary_path"] = str(summary_path)
    atomic_write_json(summary_path, summary)
    return summary
