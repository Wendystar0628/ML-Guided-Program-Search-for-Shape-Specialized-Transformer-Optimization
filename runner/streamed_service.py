"""Application service for memory-bounded streamed benchmark cases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    RunVariant,
    TransformerShape,
    WorkloadSet,
    load_workload_set,
    select_transformer_shape,
)
from runner.result_layout import intermediate_results_dir
from runner.supervisor import run_managed_benchmark
from runner.workload_execution import (
    STREAMED_POLICY_SELECTOR,
    plan_workload_execution,
    streamed_benchmark_shapes,
)


@dataclass(frozen=True, slots=True)
class StreamedBenchmarkRequest:
    """One explicit request for the independently measured streamed scope."""

    project_root: Path
    workload_set_id: str
    protocol: MeasurementProtocol
    device: str
    variant: RunVariant
    solution_policy: str = STREAMED_POLICY_SELECTOR
    case_ids: tuple[str, ...] = ()
    output_root: Path | None = None


@dataclass(frozen=True, slots=True)
class StreamedBenchmarkResult:
    """Completed streamed runs in workload order."""

    runs: tuple[dict[str, Any], ...]
    result_paths: tuple[Path, ...]


StreamedCaseStartedCallback = Callable[[int, int, TransformerShape], None]
StreamedCaseCompletedCallback = Callable[
    [int, int, TransformerShape, dict[str, Any], Path], None
]


def _select_streamed_shapes(
    workload_set: WorkloadSet,
    request: StreamedBenchmarkRequest,
) -> tuple[TransformerShape, ...]:
    if request.case_ids:
        if len(request.case_ids) != len(set(request.case_ids)):
            raise ContractError("streamed case_id values must not be repeated")
        shapes = tuple(
            select_transformer_shape(workload_set, case_id)
            for case_id in request.case_ids
        )
    else:
        shapes = streamed_benchmark_shapes(workload_set.shapes, request.variant)

    resident = tuple(
        shape.case_id
        for shape in shapes
        if not plan_workload_execution(shape, request.variant).is_streamed
    )
    if resident:
        raise ContractError(
            "benchmark-streamed accepts only shapes selected for batch-streamed "
            f"execution; resident={list(resident)}"
        )
    if not shapes:
        raise ContractError("the workload set contains no batch-streamed shapes")
    return shapes


class StreamedBenchmarkService:
    """Run streamed shapes without coupling them to the resident sweep."""

    def run(
        self,
        request: StreamedBenchmarkRequest,
        *,
        on_case_started: StreamedCaseStartedCallback | None = None,
        on_case_completed: StreamedCaseCompletedCallback | None = None,
    ) -> StreamedBenchmarkResult:
        request.protocol.validate()
        request.variant.validate()
        workload_set = load_workload_set(
            request.project_root,
            request.workload_set_id,
        )
        shapes = _select_streamed_shapes(workload_set, request)
        output_root = request.output_root or intermediate_results_dir(
            request.project_root,
            "streamed",
        )

        runs: list[dict[str, Any]] = []
        paths: list[Path] = []
        total = len(shapes)
        for index, shape in enumerate(shapes, start=1):
            if on_case_started is not None:
                on_case_started(index, total, shape)
            result, result_path = run_managed_benchmark(
                request.project_root,
                workload_set_id=request.workload_set_id,
                shape=shape,
                variant=request.variant,
                protocol=request.protocol,
                device=request.device,
                target="solution",
                workload_sha256=workload_set.sha256,
                result_dir=output_root,
                solution_policy=request.solution_policy,
            )
            runs.append(result)
            paths.append(result_path)
            if on_case_completed is not None:
                on_case_completed(index, total, shape, result, result_path)
            if result.get("outcome") == "cancelled":
                break
        return StreamedBenchmarkResult(tuple(runs), tuple(paths))


__all__ = [
    "StreamedBenchmarkRequest",
    "StreamedBenchmarkResult",
    "StreamedBenchmarkService",
]
