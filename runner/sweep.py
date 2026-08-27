"""Execute, aggregate, and persist one isolated benchmark sweep."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    WorkloadCase,
    WorkloadGroup,
    WorkloadSet,
    atomic_write_json,
    load_workload_set,
    new_run_id,
    utc_now,
)
from runner.locking import device_measurement_lease
from runner.result_contracts import (
    validate_benchmark_performance,
    validate_correctness,
)
from runner.supervisor import CancellationToken, run_managed_benchmark

ManagedBenchmark = Callable[..., tuple[dict[str, Any], Path]]
SweepValidator = Callable[
    [WorkloadSet, Sequence[Mapping[str, Any]], Sequence[Path], dict[str, Any]],
    None,
]
CaseStarted = Callable[[int, int, WorkloadCase], None]
CaseCompleted = Callable[[dict[str, Any], Path], None]


@dataclass(frozen=True)
class BenchmarkSweepRequest:
    """Complete input for one ordered benchmark sweep."""

    project_root: Path
    workload_set_id: str
    protocol: MeasurementProtocol
    device: str
    target: str = "solution"
    solution_policy: str | None = "dispatch"
    output_root: Path | None = None
    sweep_id: str | None = None


@dataclass(frozen=True)
class BenchmarkSweepResult:
    """In-memory result and owned artifact paths for one sweep."""

    sweep_id: str
    sweep_directory: Path
    runs: tuple[dict[str, Any], ...]
    run_paths: tuple[Path, ...]
    summary: dict[str, Any]
    summary_path: Path


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
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return None, "missing_run_id"
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


def _case_id(run: Mapping[str, Any]) -> str | None:
    workload = run.get("workload")
    case = workload.get("case") if isinstance(workload, Mapping) else None
    value = case.get("case_id") if isinstance(case, Mapping) else None
    return value if isinstance(value, str) and value else None


def _compact_case_result(
    case_id: str,
    run: Mapping[str, Any] | None,
    *,
    outcome: str,
    speedup: float | None,
) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "case_id": case_id,
        "outcome": outcome,
        "run_id": run.get("run_id") if run is not None else None,
        "baseline_median_ms": None,
        "target_median_ms": None,
        "speedup": speedup,
        "dispatch_policy": None,
        "selected_policy": None,
        "route_origin": None,
        "policy_applied": None,
    }
    if run is None:
        return compact
    performance = run.get("performance")
    if isinstance(performance, Mapping):
        baseline = performance.get("baseline")
        target = performance.get("target")
        if isinstance(baseline, Mapping):
            compact["baseline_median_ms"] = baseline.get("median_ms")
        if isinstance(target, Mapping):
            compact["target_median_ms"] = target.get("median_ms")
    execution_path = run.get("execution_path")
    if isinstance(execution_path, Mapping):
        compact["dispatch_policy"] = execution_path.get("dispatch_policy")
        compact["selected_policy"] = execution_path.get("selected_policy")
        compact["route_origin"] = execution_path.get("route_origin")
    return compact


def _common_dispatch(runs: Sequence[Mapping[str, Any]]) -> dict[str, str] | None:
    pairs: set[tuple[str, str]] = set()
    for run in runs:
        execution_path = run.get("execution_path")
        if not isinstance(execution_path, Mapping):
            return None
        source = execution_path.get("dispatch_source")
        digest = execution_path.get("dispatch_table_sha256")
        if not isinstance(source, str) or not isinstance(digest, str):
            return None
        pairs.add((source, digest))
    if len(pairs) != 1:
        return None
    source, digest = next(iter(pairs))
    return {"source": source, "sha256": digest}


def summarize_sweep(
    workload_set: WorkloadSet,
    runs: list[dict[str, Any]],
    *,
    target: str,
) -> dict[str, Any]:
    """Build the single compact summary shared by CLI and verified runs."""

    if target not in {"baseline", "solution"}:
        raise ContractError(f"unsupported sweep target: {target}")

    expected_cases = workload_set.cases
    expected_ids = [case.case_id for case in expected_cases]
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    unexpected: list[str] = []
    for run in runs:
        case_id = _case_id(run)
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
                _compact_case_result(
                    case_id,
                    None,
                    outcome="missing",
                    speedup=None,
                )
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
            _compact_case_result(
                case_id,
                run,
                outcome=reason or outcome,
                speedup=speedup,
            )
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
    created_values = [
        value
        for value in (run.get("created_at") for run in runs)
        if isinstance(value, str) and value
    ]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "sweep_id": context_anchor.get("sweep_id") if context_anchor else None,
        "created_at": min(created_values) if created_values else None,
        "workload_set_id": workload_set.workload_set_id,
        "workload_sha256": workload_set.sha256,
        "target": target,
        "source": (
            {
                "official_sha256": context_anchor["official_snapshot_sha256"],
                "solution_sha256": context_anchor["solution_source_sha256"],
            }
            if context_anchor is not None
            else None
        ),
        "protocol": context_anchor.get("protocol") if context_anchor else None,
        "environment": context_anchor.get("environment") if context_anchor else None,
        "dispatch": _common_dispatch(runs),
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


def _resolve_output_root(request: BenchmarkSweepRequest) -> Path:
    project_root = request.project_root.resolve()
    output_root = request.output_root or project_root / "results" / "sweeps"
    if not output_root.is_absolute():
        output_root = project_root / output_root
    return output_root.resolve()


def _validated_sweep_id(value: str | None) -> str:
    sweep_id = value or new_run_id()
    if not sweep_id or Path(sweep_id).name != sweep_id or sweep_id in {".", ".."}:
        raise ContractError("sweep_id must be one non-empty path component")
    return sweep_id


class BenchmarkSweepService:
    """Run every workload case in its own worker and own the sweep artifacts."""

    def __init__(self, run_case: ManagedBenchmark | None = None) -> None:
        self._run_case = run_case or run_managed_benchmark

    def run(
        self,
        request: BenchmarkSweepRequest,
        *,
        validate_before_persist: SweepValidator | None = None,
        on_case_started: CaseStarted | None = None,
        on_case_completed: CaseCompleted | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> BenchmarkSweepResult:
        with device_measurement_lease(
            request.project_root,
            request.device,
            purpose="benchmark sweep",
        ):
            return self._run(
                request,
                validate_before_persist=validate_before_persist,
                on_case_started=on_case_started,
                on_case_completed=on_case_completed,
                cancellation_token=cancellation_token,
            )

    def _run(
        self,
        request: BenchmarkSweepRequest,
        *,
        validate_before_persist: SweepValidator | None,
        on_case_started: CaseStarted | None,
        on_case_completed: CaseCompleted | None,
        cancellation_token: CancellationToken | None,
    ) -> BenchmarkSweepResult:
        if request.target not in {"baseline", "solution"}:
            raise ContractError(f"unsupported sweep target: {request.target}")
        if request.target == "baseline" and request.solution_policy is not None:
            solution_policy = None
        else:
            solution_policy = request.solution_policy

        project_root = request.project_root.resolve()
        workload_set = load_workload_set(project_root, request.workload_set_id)
        sweep_id = _validated_sweep_id(request.sweep_id)
        sweep_directory = _resolve_output_root(request) / sweep_id
        if sweep_directory.exists():
            raise ContractError(f"refusing to reuse existing sweep: {sweep_directory}")
        runs_directory = sweep_directory / "runs"
        started_at = utc_now()
        runs: list[dict[str, Any]] = []
        run_paths: list[Path] = []
        cancelled = False

        total = len(workload_set.cases)
        for index, case in enumerate(workload_set.cases, start=1):
            if cancellation_token is not None and cancellation_token.is_cancelled:
                cancelled = True
                break
            if on_case_started is not None:
                on_case_started(index, total, case)
            if cancellation_token is not None and cancellation_token.is_cancelled:
                cancelled = True
                break
            run, run_path = self._run_case(
                project_root,
                workload_set_id=request.workload_set_id,
                case=case,
                protocol=request.protocol,
                device=request.device,
                target=request.target,
                workload_sha256=workload_set.sha256,
                sweep_id=sweep_id,
                result_dir=runs_directory,
                solution_policy=solution_policy,
                cancellation_token=cancellation_token,
            )
            run_path = run_path.resolve()
            if run_path.parent != runs_directory.resolve():
                raise ContractError(
                    f"benchmark result escaped its sweep directory: {run_path}"
                )
            runs.append(run)
            run_paths.append(run_path)
            if on_case_completed is not None:
                on_case_completed(run, run_path)
            if run.get("outcome") == "cancelled":
                cancelled = True
                break

        summary = summarize_sweep(
            workload_set,
            runs,
            target=request.target,
        )
        summary["sweep_id"] = sweep_id
        summary["created_at"] = started_at
        if cancelled:
            summary["sweep_outcome"] = "cancelled"
        if validate_before_persist is not None and not cancelled:
            validate_before_persist(workload_set, runs, run_paths, summary)
        summary_path = sweep_directory / "summary.json"
        atomic_write_json(summary_path, summary)
        return BenchmarkSweepResult(
            sweep_id=sweep_id,
            sweep_directory=sweep_directory,
            runs=tuple(runs),
            run_paths=tuple(run_paths),
            summary=summary,
            summary_path=summary_path,
        )


__all__ = [
    "BenchmarkSweepRequest",
    "BenchmarkSweepResult",
    "BenchmarkSweepService",
    "summarize_sweep",
]
