"""Execute and summarize one ordered official benchmark sweep."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    RunVariant,
    TransformerShape,
    WorkloadSet,
    atomic_write_json,
    load_workload_set,
    new_run_id,
    utc_now,
)
from runner.locking import device_measurement_lease
from runner.performance_metrics import LOGICAL_OPERATOR_TRAFFIC_SCOPE
from runner.result_contracts import (
    RUN_RESULT_SCHEMA_VERSION,
    ComparisonMode,
    validate_benchmark_performance,
    validate_correctness,
)
from runner.result_layout import intermediate_results_dir
from runner.supervisor import CancellationToken, run_managed_benchmark
from runner.workload_execution import (
    resident_benchmark_shapes,
    streamed_benchmark_shapes,
)

ManagedBenchmark = Callable[..., tuple[dict[str, Any], Path]]
SWEEP_SUMMARY_SCHEMA_VERSION = 6
SweepValidator = Callable[
    [WorkloadSet, Sequence[Mapping[str, Any]], Sequence[Path], dict[str, Any]],
    None,
]
ShapeStarted = Callable[[int, int, TransformerShape], None]
ShapeCompleted = Callable[[dict[str, Any], Path], None]


@dataclass(frozen=True)
class BenchmarkSweepRequest:
    """Complete input for one local sweep of the official workload."""

    project_root: Path
    workload_set_id: str
    protocol: MeasurementProtocol
    device: str
    variant: RunVariant = field(default_factory=RunVariant)
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


def _geometric_mean(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ContractError("geometric mean inputs must be finite and positive")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _outcome(run: Mapping[str, Any]) -> str:
    value = run.get("outcome")
    return value if isinstance(value, str) else "runtime_error"


def _validated_performance(
    run: Mapping[str, Any],
    target: str,
) -> tuple[float | None, str | None]:
    protocol = run.get("protocol")
    if not isinstance(protocol, Mapping):
        return None, "missing_protocol"
    expected_comparison_mode: ComparisonMode = (
        "baseline_only" if target == "baseline" else "paired"
    )
    if run.get("comparison_mode") != expected_comparison_mode:
        return None, "comparison_mode_mismatch"
    performance, error = validate_benchmark_performance(
        run.get("performance"),
        target=target,
        comparison_mode=expected_comparison_mode,
        repeats=protocol.get("repeats"),
        rounds=protocol.get("rounds"),
        expected_timer="cuda_event",
    )
    if error is not None or target == "baseline":
        return None, error
    assert performance is not None
    return performance.speedup, None


def _validated_correctness(run: Mapping[str, Any], target: str) -> str | None:
    if target == "baseline":
        return None
    protocol = run.get("protocol")
    if not isinstance(protocol, Mapping):
        return "invalid_correctness"
    return validate_correctness(
        run.get("correctness"), expected_trials=protocol.get("accuracy_trials")
    )


def _run_context(
    run: Mapping[str, Any],
    *,
    target: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if run.get("schema_version") != RUN_RESULT_SCHEMA_VERSION:
        return None, "unsupported_schema"
    if run.get("run_kind") != "benchmark":
        return None, "run_kind_mismatch"
    if run.get("target") != target:
        return None, "target_mismatch"
    run_id = run.get("run_id")
    sweep_id = run.get("sweep_id")
    if not isinstance(run_id, str) or not run_id:
        return None, "missing_run_id"
    if not isinstance(sweep_id, str) or not sweep_id:
        return None, "missing_sweep_id"
    source = run.get("source")
    if not isinstance(source, Mapping):
        return None, "missing_source"
    official_hash = source.get("official_sha256")
    if not isinstance(official_hash, str) or not official_hash:
        return None, "missing_official_snapshot_hash"
    solution_hash = source.get("solution_sha256")
    if target == "solution" and (
        not isinstance(solution_hash, str) or not solution_hash
    ):
        return None, "missing_solution_source_hash"
    selected_policy = run.get("selected_policy")
    if not isinstance(selected_policy, str) or not selected_policy:
        return None, "missing_selected_policy"
    if run.get("policy_applied") is not True:
        return None, "policy_not_applied"
    if run.get("actual_policy") != selected_policy:
        return None, "actual_policy_mismatch"
    protocol = run.get("protocol")
    environment = run.get("environment")
    if not isinstance(protocol, Mapping):
        return None, "missing_protocol"
    if not isinstance(environment, Mapping):
        return None, "missing_environment"
    resolved_device = environment.get("device")
    if not isinstance(resolved_device, str) or not resolved_device:
        return None, "missing_resolved_device"
    return {
        "sweep_id": sweep_id,
        "official_snapshot_sha256": official_hash,
        "solution_source_sha256": solution_hash if target == "solution" else None,
        "protocol": dict(protocol),
        "environment": dict(environment),
    }, None


def _context_mismatch(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
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


def _shape_id(run: Mapping[str, Any]) -> str | None:
    workload = run.get("workload")
    shape = workload.get("shape") if isinstance(workload, Mapping) else None
    value = shape.get("case_id") if isinstance(shape, Mapping) else None
    return value if isinstance(value, str) and value else None


def _timing_summary(value: Any) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    median = value.get("median_ms")
    p90 = value.get("p90_ms")
    if (
        not isinstance(median, (int, float))
        or isinstance(median, bool)
        or not isinstance(p90, (int, float))
        or isinstance(p90, bool)
    ):
        return None
    return {"median_ms": float(median), "p90_ms": float(p90)}


def _validated_metric_summary(
    value: Any,
    *,
    target: str,
) -> tuple[dict[str, int | float | str] | None, str | None]:
    """Validate the compact useful-work metrics derived by the supervisor."""

    if not isinstance(value, Mapping):
        return None, "missing_performance"
    useful_flops = value.get("useful_matmul_flops")
    attention_fraction = value.get("attention_flops_fraction")
    achieved_tflops = value.get("achieved_tflops")
    logical_bytes = value.get("estimated_logical_operator_bytes")
    logical_intensity = value.get(
        "logical_operator_arithmetic_intensity_flops_per_byte"
    )
    logical_traffic_gbps = value.get("estimated_logical_operator_traffic_gbps")
    logical_traffic_scope = value.get("logical_operator_traffic_scope")
    if (
        isinstance(useful_flops, bool)
        or not isinstance(useful_flops, int)
        or useful_flops <= 0
    ):
        return None, "invalid_useful_matmul_flops"
    if (
        isinstance(attention_fraction, bool)
        or not isinstance(attention_fraction, (int, float))
        or not math.isfinite(float(attention_fraction))
        or not 0.0 <= float(attention_fraction) <= 1.0
    ):
        return None, "invalid_attention_flops_fraction"
    if (
        isinstance(achieved_tflops, bool)
        or not isinstance(achieved_tflops, (int, float))
        or not math.isfinite(float(achieved_tflops))
        or float(achieved_tflops) <= 0
    ):
        return None, "invalid_achieved_tflops"
    if (
        isinstance(logical_bytes, bool)
        or not isinstance(logical_bytes, int)
        or logical_bytes <= 0
    ):
        return None, "invalid_estimated_logical_operator_bytes"
    if (
        isinstance(logical_intensity, bool)
        or not isinstance(logical_intensity, (int, float))
        or not math.isfinite(float(logical_intensity))
        or float(logical_intensity) <= 0
    ):
        return None, "invalid_logical_operator_arithmetic_intensity"
    if (
        isinstance(logical_traffic_gbps, bool)
        or not isinstance(logical_traffic_gbps, (int, float))
        or not math.isfinite(float(logical_traffic_gbps))
        or float(logical_traffic_gbps) <= 0
    ):
        return None, "invalid_estimated_logical_operator_traffic_gbps"
    if logical_traffic_scope != LOGICAL_OPERATOR_TRAFFIC_SCOPE:
        return None, "invalid_logical_operator_traffic_scope"

    measured_side = value.get("baseline" if target == "baseline" else "target")
    timing = _timing_summary(measured_side)
    if timing is None:
        return None, "missing_measured_timing"
    expected_tflops = useful_flops / (timing["median_ms"] * 1_000_000_000.0)
    if not math.isclose(
        float(achieved_tflops),
        expected_tflops,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return None, "achieved_tflops_mismatch"
    expected_intensity = useful_flops / logical_bytes
    if not math.isclose(
        float(logical_intensity),
        expected_intensity,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        return None, "logical_operator_arithmetic_intensity_mismatch"
    expected_traffic_gbps = logical_bytes / (timing["median_ms"] * 1_000_000.0)
    if not math.isclose(
        float(logical_traffic_gbps),
        expected_traffic_gbps,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return None, "estimated_logical_operator_traffic_gbps_mismatch"
    return (
        {
            "useful_matmul_flops": useful_flops,
            "attention_flops_fraction": float(attention_fraction),
            "achieved_tflops": float(achieved_tflops),
            "estimated_logical_operator_bytes": logical_bytes,
            "logical_operator_arithmetic_intensity_flops_per_byte": float(
                logical_intensity
            ),
            "estimated_logical_operator_traffic_gbps": float(logical_traffic_gbps),
            "logical_operator_traffic_scope": logical_traffic_scope,
        },
        None,
    )


def _accuracy_summary(run: Mapping[str, Any]) -> dict[str, Any] | None:
    value = run.get("correctness")
    if not isinstance(value, Mapping):
        return None
    return {
        key: value[key]
        for key in (
            "passed",
            "trial_count",
            "failed_elements",
            "max_abs_error",
            "max_relative_error",
            "diagnostic",
        )
        if value.get(key) is not None
    }


def _actual_policy(run: Mapping[str, Any], *, target: str) -> str | None:
    if target == "baseline":
        return "official-baseline"
    persisted = run.get("actual_policy")
    return persisted if isinstance(persisted, str) and persisted else None


def _failure_summary(run: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if run is None:
        return {"stage": "sweep", "type": "MissingResult", "message": "missing"}
    value = run.get("failure")
    if not isinstance(value, Mapping):
        return None
    return {
        key: value[key]
        for key in ("stage", "type", "message", "exit_code")
        if value.get(key) is not None
    }


def _compact_shape_result(
    shape_id: str,
    run: Mapping[str, Any] | None,
    *,
    target: str,
    outcome: str,
    speedup: float | None,
    metric_summary: Mapping[str, int | float | str] | None,
) -> dict[str, Any]:
    performance = run.get("performance") if run is not None else None
    baseline = None
    solution = None
    if isinstance(performance, Mapping):
        baseline = _timing_summary(performance.get("baseline"))
        solution = _timing_summary(performance.get("target"))
    failure = _failure_summary(run) if outcome != "success" else None
    if outcome != "success" and failure is None:
        failure = {
            "stage": "summary_validation",
            "type": "InvalidResult",
            "message": outcome,
        }
    compact = {
        "case_id": shape_id,
        "outcome": outcome,
        "baseline": baseline,
        "solution": solution,
        "speedup": speedup,
        "accuracy": _accuracy_summary(run) if run is not None else None,
        "selected_policy": run.get("selected_policy") if run is not None else None,
        "policy_applied": run.get("policy_applied") if run is not None else None,
        "actual_policy": _actual_policy(run, target=target)
        if run is not None
        else None,
        "failure": failure,
    }
    if metric_summary is not None:
        compact.update(metric_summary)
    if isinstance(performance, Mapping):
        peak_bytes = performance.get("peak_device_allocated_bytes")
        if (
            isinstance(peak_bytes, int)
            and not isinstance(peak_bytes, bool)
            and peak_bytes > 0
        ):
            compact["peak_device_allocated_bytes"] = peak_bytes
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
    variant: RunVariant | None = None,
) -> dict[str, Any]:
    """Build one compact, unweighted summary for resident benchmark shapes."""

    if target not in {"baseline", "solution"}:
        raise ContractError(f"unsupported sweep target: {target}")
    expected_variant = variant or RunVariant()
    expected_shapes = resident_benchmark_shapes(
        workload_set.shapes,
        expected_variant,
    )
    expected_ids = [shape.case_id for shape in expected_shapes]
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    unexpected: list[str] = []
    for run in runs:
        shape_id = _shape_id(run)
        if shape_id is None or shape_id not in expected_ids:
            unexpected.append(shape_id or "<missing-case-id>")
        elif shape_id in indexed:
            duplicates.append(shape_id)
        else:
            indexed[shape_id] = run

    shape_results: list[dict[str, Any]] = []
    failed_cases: list[dict[str, str]] = []
    speedups: list[float] = []
    context_anchor: dict[str, Any] | None = None
    for shape in expected_shapes:
        shape_id = shape.case_id
        run = indexed.get(shape_id)
        reason: str | None = "missing" if run is None else None
        outcome = "missing" if run is None else _outcome(run)
        if run is not None and outcome != "success":
            reason = outcome
        if run is not None and reason is None:
            reason = _validated_correctness(run, target)
            workload = run.get("workload")
            if reason is None and (
                not isinstance(workload, Mapping)
                or workload.get("set_id") != workload_set.workload_set_id
                or workload.get("sha256") != workload_set.sha256
                or workload.get("shape") != shape.as_dict()
                or workload.get("variant") != expected_variant.as_dict()
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
        metric_summary: dict[str, int | float | str] | None = None
        if run is not None and reason is None:
            speedup, reason = _validated_performance(run, target)
        if run is not None and reason is None:
            metric_summary, reason = _validated_metric_summary(
                run.get("performance"),
                target=target,
            )
        compact_outcome = reason or outcome
        shape_results.append(
            _compact_shape_result(
                shape_id,
                run,
                target=target,
                outcome=compact_outcome,
                speedup=speedup,
                metric_summary=metric_summary,
            )
        )
        if reason is not None:
            failed_cases.append({"case_id": shape_id, "outcome": reason})
        elif target == "solution" and speedup is not None:
            speedups.append(speedup)

    failed_cases.extend(
        {"case_id": shape_id, "outcome": "duplicate"} for shape_id in duplicates
    )
    failed_cases.extend(
        {"case_id": shape_id, "outcome": "unexpected"} for shape_id in unexpected
    )
    complete = not failed_cases and len(indexed) == len(expected_ids)
    created_values = [
        value
        for value in (run.get("created_at") for run in runs)
        if isinstance(value, str) and value
    ]
    return {
        "schema_version": SWEEP_SUMMARY_SCHEMA_VERSION,
        "sweep_id": context_anchor.get("sweep_id") if context_anchor else None,
        "created_at": min(created_values) if created_values else None,
        "workload_set_id": workload_set.workload_set_id,
        "workload_sha256": workload_set.sha256,
        "variant": expected_variant.as_dict(),
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
        "case_results": shape_results,
        "failed_cases": failed_cases,
        "streamed_case_ids": [
            shape.case_id
            for shape in streamed_benchmark_shapes(
                workload_set.shapes,
                expected_variant,
            )
        ],
        "geomean_speedup": _geometric_mean(speedups)
        if complete and target == "solution"
        else None,
    }


def _resolve_output_root(request: BenchmarkSweepRequest) -> Path:
    project_root = request.project_root.resolve()
    output_root = request.output_root or intermediate_results_dir(
        project_root,
        "sweeps",
    )
    if not output_root.is_absolute():
        output_root = project_root / output_root
    return output_root.resolve()


def _validated_sweep_id(value: str | None) -> str:
    sweep_id = value or new_run_id()
    if not sweep_id or Path(sweep_id).name != sweep_id or sweep_id in {".", ".."}:
        raise ContractError("sweep_id must be one non-empty path component")
    return sweep_id


class BenchmarkSweepService:
    """Run resident benchmark shapes in isolated workers."""

    def __init__(self, run_shape: ManagedBenchmark | None = None) -> None:
        self._run_shape = run_shape or run_managed_benchmark

    def run(
        self,
        request: BenchmarkSweepRequest,
        *,
        validate_before_persist: SweepValidator | None = None,
        on_case_started: ShapeStarted | None = None,
        on_case_completed: ShapeCompleted | None = None,
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
        on_case_started: ShapeStarted | None,
        on_case_completed: ShapeCompleted | None,
        cancellation_token: CancellationToken | None,
    ) -> BenchmarkSweepResult:
        if request.target not in {"baseline", "solution"}:
            raise ContractError(f"unsupported sweep target: {request.target}")
        solution_policy = (
            None if request.target == "baseline" else request.solution_policy
        )
        request.variant.validate()

        project_root = request.project_root.resolve()
        workload_set = load_workload_set(project_root, request.workload_set_id)
        shapes = resident_benchmark_shapes(
            workload_set.shapes,
            request.variant,
        )
        sweep_id = _validated_sweep_id(request.sweep_id)
        sweep_directory = _resolve_output_root(request) / sweep_id
        if sweep_directory.exists():
            raise ContractError(f"refusing to reuse existing sweep: {sweep_directory}")
        runs_directory = sweep_directory / "runs"
        started_at = utc_now()
        runs: list[dict[str, Any]] = []
        run_paths: list[Path] = []
        cancelled = False

        total = len(shapes)
        for index, shape in enumerate(shapes, start=1):
            if cancellation_token is not None and cancellation_token.is_cancelled:
                cancelled = True
                break
            if on_case_started is not None:
                on_case_started(index, total, shape)
            if cancellation_token is not None and cancellation_token.is_cancelled:
                cancelled = True
                break
            run, run_path = self._run_shape(
                project_root,
                workload_set_id=request.workload_set_id,
                shape=shape,
                variant=request.variant,
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
            variant=request.variant,
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
