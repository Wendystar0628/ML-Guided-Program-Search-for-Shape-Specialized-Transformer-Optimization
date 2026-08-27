"""Pure aggregation for ordered workload sweeps."""

from __future__ import annotations

import math
from typing import Any

from runner.contracts import ContractError, WorkloadGroup, WorkloadSet
from runner.result_contracts import (
    validate_benchmark_performance,
    validate_correctness,
)


def _geometric_mean(values: list[float], weights: list[float] | None = None) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ContractError("geometric mean inputs must be finite and positive")
    if weights is None:
        return math.exp(math.fsum(math.log(value) for value in values) / len(values))
    if len(values) != len(weights):
        raise ContractError("geometric mean values and weights must have equal length")
    total_weight = math.fsum(weights)
    if total_weight <= 0 or any(weight <= 0 for weight in weights):
        raise ContractError("geometric mean weights must be positive")
    return math.exp(
        math.fsum(weight * math.log(value) for value, weight in zip(values, weights))
        / total_weight
    )


def _outcome(run: dict[str, Any]) -> str:
    value = run.get("outcome")
    return value if isinstance(value, str) else "runtime_error"


def _validated_performance(
    run: dict[str, Any],
    target: str,
) -> tuple[float | None, str | None]:
    protocol = run.get("protocol")
    if not isinstance(protocol, dict):
        return None, "missing_protocol"
    performance, error = validate_benchmark_performance(
        run.get("performance"),
        target=target,
        repeats=protocol.get("repeats"),
        rounds=protocol.get("rounds"),
        expected_timer="cuda_event",
    )
    if error is not None or target == "baseline":
        return None, error
    assert performance is not None
    return performance.speedup, None


def _validated_correctness(
    run: dict[str, Any],
    target: str,
) -> str | None:
    if target == "baseline":
        return None
    protocol = run.get("protocol")
    if not isinstance(protocol, dict):
        return "invalid_correctness"
    return validate_correctness(
        run.get("correctness"), expected_trials=protocol.get("accuracy_trials")
    )


def _run_context(
    run: dict[str, Any],
    *,
    target: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if run.get("schema_version") != 2:
        return None, "unsupported_schema"
    if run.get("run_kind") != "benchmark":
        return None, "run_kind_mismatch"
    if run.get("target") != target:
        return None, "target_mismatch"
    sweep_id = run.get("sweep_id")
    if not isinstance(sweep_id, str) or not sweep_id:
        return None, "missing_sweep_id"
    source = run.get("source")
    if not isinstance(source, dict):
        return None, "missing_source"
    official_hash = source.get("official_sha256")
    if not isinstance(official_hash, str) or not official_hash:
        return None, "missing_official_snapshot_hash"
    solution_hash = source.get("solution_sha256")
    if target == "solution" and (
        not isinstance(solution_hash, str) or not solution_hash
    ):
        return None, "missing_solution_source_hash"
    protocol = run.get("protocol")
    if not isinstance(protocol, dict):
        return None, "missing_protocol"
    environment = run.get("environment")
    if not isinstance(environment, dict):
        return None, "missing_environment"
    resolved_device = environment.get("device")
    if not isinstance(resolved_device, str) or not resolved_device:
        return None, "missing_resolved_device"
    return {
        "sweep_id": sweep_id,
        "official_snapshot_sha256": official_hash,
        "solution_source_sha256": solution_hash if target == "solution" else None,
        "protocol": protocol,
        "environment": environment,
    }, None


def _context_mismatch(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> str | None:
    labels = {
        "sweep_id": "sweep_id_mismatch",
        "official_snapshot_sha256": "official_snapshot_mismatch",
        "solution_source_sha256": "solution_source_mismatch",
        "protocol": "protocol_mismatch",
        "environment": "environment_mismatch",
    }
    for key, reason in labels.items():
        if actual.get(key) != expected.get(key):
            return reason
    return None


def summarize_sweep(
    workload_set: WorkloadSet,
    runs: list[dict[str, Any]],
    *,
    target: str,
) -> dict[str, Any]:
    """Summarize a sweep without persisting a second result artifact."""

    if target not in {"baseline", "solution"}:
        raise ContractError(f"unsupported sweep target: {target}")

    expected_cases = workload_set.cases
    expected_ids = [case.case_id for case in expected_cases]
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    unexpected: list[str] = []
    for run in runs:
        workload = run.get("workload") or {}
        case = workload.get("case") if isinstance(workload, dict) else None
        case_id = case.get("case_id") if isinstance(case, dict) else None
        if not isinstance(case_id, str):
            unexpected.append("<missing-case-id>")
            continue
        if case_id not in expected_ids:
            unexpected.append(case_id)
            continue
        if case_id in indexed:
            duplicates.append(case_id)
            continue
        indexed[case_id] = run

    case_results: list[dict[str, Any]] = []
    failed_cases: list[dict[str, str]] = []
    speedups: dict[str, float] = {}
    context_anchor: dict[str, Any] | None = None
    for case, case_id in zip(expected_cases, expected_ids):
        run = indexed.get(case_id)
        if run is None:
            case_results.append(
                {"case_id": case_id, "outcome": "missing", "speedup": None}
            )
            failed_cases.append({"case_id": case_id, "outcome": "missing"})
            continue
        outcome = _outcome(run)
        reason: str | None = None
        if outcome != "success":
            reason = outcome
        if reason is None:
            reason = _validated_correctness(run, target)

        workload = run.get("workload")
        if reason is None and (
            not isinstance(workload, dict)
            or workload.get("set_id") != workload_set.workload_set_id
            or workload.get("sha256") != workload_set.sha256
            or workload.get("case") != case.as_dict()
        ):
            reason = "workload_mismatch"

        context, context_error = _run_context(run, target=target)
        if reason is None and context_error is not None:
            reason = context_error
        if context is not None:
            if context_anchor is None:
                context_anchor = context
            elif reason is None:
                reason = _context_mismatch(context_anchor, context)

        speedup: float | None = None
        if reason is None:
            speedup, reason = _validated_performance(run, target)
        case_results.append(
            {
                "case_id": case_id,
                "outcome": reason or outcome,
                "speedup": speedup,
            }
        )
        if reason is not None:
            failed_cases.append({"case_id": case_id, "outcome": reason})
        elif target == "solution" and speedup is not None:
            speedups[case_id] = speedup

    for case_id in duplicates:
        failed_cases.append({"case_id": case_id, "outcome": "duplicate"})
    for case_id in unexpected:
        failed_cases.append({"case_id": case_id, "outcome": "unexpected"})

    complete = not failed_cases and len(indexed) == len(expected_ids)
    summary: dict[str, Any] = {
        "workload_set_id": workload_set.workload_set_id,
        "workload_sha256": workload_set.sha256,
        "target": target,
        "sweep_outcome": "complete" if complete else "incomplete",
        "case_results": case_results,
        "failed_cases": failed_cases,
        "groups": [],
        "group_balanced_geomean_speedup": None,
        "worst_case_speedup": None,
    }
    if not complete or target != "solution":
        return summary

    group_results: list[dict[str, Any]] = []
    group_values: list[float] = []
    group_weights: list[float] = []
    for group in workload_set.groups:
        if not isinstance(group, WorkloadGroup):
            raise ContractError("workload_set groups must contain WorkloadGroup values")
        value = _geometric_mean([speedups[case_id] for case_id in group.case_ids])
        group_results.append(
            {
                "group_id": group.group_id,
                "display_name": group.display_name,
                "weight": group.weight,
                "geomean_speedup": value,
            }
        )
        group_values.append(value)
        group_weights.append(float(group.weight))

    summary["groups"] = group_results
    summary["group_balanced_geomean_speedup"] = _geometric_mean(
        group_values, group_weights
    )
    summary["worst_case_speedup"] = min(speedups.values())
    return summary
