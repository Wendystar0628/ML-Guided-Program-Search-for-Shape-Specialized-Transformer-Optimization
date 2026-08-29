from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from autotune.service import SearchServiceRequest
from autotune.workflow import OptimizationLoopPolicy, OptimizationService
from benchmarking.protocols import (
    load_resident_shapes,
    load_shapes,
    load_streamed_shapes,
)
from cli import _optimization_case_ids


class _SearchService:
    def __init__(self, outcomes: list[tuple[int, tuple[bool, ...]]]) -> None:
        self.outcomes = iter(outcomes)
        self.requests: list[SearchServiceRequest] = []

    def run(self, request: SearchServiceRequest) -> object:
        self.requests.append(request)
        exit_code, updates = next(self.outcomes)
        return SimpleNamespace(
            exit_code=exit_code,
            shape_results=tuple(
                SimpleNamespace(deployment_updated=updated) for updated in updates
            ),
        )


def _request() -> SearchServiceRequest:
    return SearchServiceRequest(
        project_root=Path("."),
        case_ids=("official_01",),
        budget_seconds=1.0,
        seed=100,
    )


def test_deployment_resets_patience_before_plateau_stop() -> None:
    search = _SearchService(
        [
            (0, (False,)),
            (0, (False,)),
            (0, (True,)),
            (0, (False,)),
            (0, (False,)),
            (0, (False,)),
        ]
    )

    result = OptimizationService(search).run(  # type: ignore[arg-type]
        _request(),
        OptimizationLoopPolicy(no_deployment_patience=3, max_iterations=10),
    )

    assert result.stop_reason == "no_deployment_patience"
    assert result.iterations_run == 6
    assert result.total_deployment_updates == 1
    assert result.no_deployment_streak == 3
    assert [request.seed for request in search.requests] == list(range(100, 106))


def test_hard_iteration_limit_stops_continuous_deployments() -> None:
    search = _SearchService([(0, (True,))] * 3)

    result = OptimizationService(search).run(  # type: ignore[arg-type]
        _request(),
        OptimizationLoopPolicy(no_deployment_patience=2, max_iterations=3),
    )

    assert result.stop_reason == "max_iterations"
    assert result.iterations_run == 3
    assert result.total_deployment_updates == 3
    assert result.no_deployment_streak == 0


def test_failure_and_interrupt_stop_immediately_without_advancing_patience() -> None:
    for exit_code, expected_reason in ((1, "failed"), (130, "interrupted")):
        search = _SearchService([(exit_code, (False,)), (0, (False,))])

        result = OptimizationService(search).run(  # type: ignore[arg-type]
            _request(),
            OptimizationLoopPolicy(no_deployment_patience=2, max_iterations=4),
        )

        assert result.stop_reason == expected_reason
        assert result.exit_code == exit_code
        assert result.iterations_run == 1
        assert result.no_deployment_streak == 0
        assert len(search.requests) == 1


def test_resident_and_shape14_groups_partition_the_official_workload() -> None:
    project_root = Path(__file__).resolve().parents[1]
    all_shapes = load_shapes(project_root)
    resident = load_resident_shapes(project_root)
    streamed = load_streamed_shapes(project_root)

    assert tuple((*resident, *streamed)) == all_shapes
    assert {shape.case_id for shape in resident}.isdisjoint(
        shape.case_id for shape in streamed
    )
    assert _optimization_case_ids(project_root, "resident") == tuple(
        f"official_{index:02d}" for index in range(1, 14)
    )
    assert _optimization_case_ids(project_root, "shape14") == ("official_14",)
