from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from autotune.optimization_loop import OptimizationLoop, OptimizationLoopPolicy
from autotune.search_sweep import SearchSweepRequest, SearchSweepResult
from benchmarking.protocols import (
    load_resident_shapes,
    load_shapes,
    load_streamed_shapes,
)
from cli import _optimization_case_ids


class _SearchSweep:
    def __init__(
        self,
        outcomes: list[tuple[int, tuple[tuple[bool, bool, bool], ...]]],
    ) -> None:
        self.outcomes = iter(outcomes)
        self.requests: list[SearchSweepRequest] = []

    def run(self, request: SearchSweepRequest) -> object:
        self.requests.append(request)
        exit_code, shapes = next(self.outcomes)
        return SimpleNamespace(
            exit_code=exit_code,
            shape_results=tuple(
                SimpleNamespace(
                    deployment_updated=updated,
                    search_result=SimpleNamespace(
                        made_search_progress=made_search_progress,
                        level1_space_exhausted=level1_space_exhausted,
                    ),
                )
                for updated, made_search_progress, level1_space_exhausted in shapes
            ),
        )


def _request() -> SearchSweepRequest:
    return SearchSweepRequest(
        project_root=Path("."),
        case_ids=("official_01",),
        budget_seconds=1.0,
        seed=100,
    )


def test_partial_search_progress_continues_to_a_later_deployment() -> None:
    search = _SearchSweep(
        [
            (0, ((False, True, False),)),
            (0, ((True, True, False),)),
        ]
    )

    result = OptimizationLoop(search).run(  # type: ignore[arg-type]
        _request(),
        OptimizationLoopPolicy(no_progress_patience=1, max_iterations=2),
    )

    assert result.stop_reason == "max_iterations"
    assert result.iterations_run == 2
    assert result.total_deployment_updates == 1
    assert result.no_progress_streak == 0
    assert [request.seed for request in search.requests] == [100, 101]


def test_no_progress_patience_stops_only_after_empty_iterations() -> None:
    search = _SearchSweep(
        [
            (0, ((False, False, False),)),
            (0, ((False, False, False),)),
            (0, ((True, True, False),)),
        ]
    )

    result = OptimizationLoop(search).run(  # type: ignore[arg-type]
        _request(),
        OptimizationLoopPolicy(no_progress_patience=2, max_iterations=4),
    )

    assert result.stop_reason == "no_progress_patience"
    assert result.iterations_run == 2
    assert result.no_progress_streak == 2
    assert len(search.requests) == 2


def test_hard_iteration_limit_stops_continuous_deployments() -> None:
    search = _SearchSweep([(0, ((True, True, False),))] * 3)

    result = OptimizationLoop(search).run(  # type: ignore[arg-type]
        _request(),
        OptimizationLoopPolicy(no_progress_patience=2, max_iterations=3),
    )

    assert result.stop_reason == "max_iterations"
    assert result.iterations_run == 3
    assert result.total_deployment_updates == 3
    assert result.no_progress_streak == 0


def test_no_new_evidence_stops_an_exhausted_search_immediately() -> None:
    class _ExhaustedSweep(_SearchSweep):
        def run(self, request: SearchSweepRequest) -> object:
            result = super().run(request)
            for item in result.shape_results:
                item.search_result.made_search_progress = False
                item.search_result.level1_space_exhausted = True
            return result

    result = OptimizationLoop(
        _ExhaustedSweep([(0, ((False, False, True),))])
    ).run(
        _request(),
        OptimizationLoopPolicy(no_progress_patience=5, max_iterations=10),
    )

    assert result.stop_reason == "search_space_exhausted"
    assert result.iterations_run == 1


def test_failure_and_interrupt_stop_immediately_without_advancing_patience() -> None:
    for exit_code, expected_reason in ((1, "failed"), (130, "interrupted")):
        search = _SearchSweep(
            [
                (exit_code, ((False, False, False),)),
                (0, ((False, False, False),)),
            ]
        )

        result = OptimizationLoop(search).run(  # type: ignore[arg-type]
            _request(),
            OptimizationLoopPolicy(no_progress_patience=2, max_iterations=4),
        )

        assert result.stop_reason == expected_reason
        assert result.exit_code == exit_code
        assert result.iterations_run == 1
        assert result.no_progress_streak == 0
        assert len(search.requests) == 1


def test_budget_limited_sweep_is_success_but_interrupt_is_not() -> None:
    partial = SimpleNamespace(
        selected_config=None,
        search_result=SimpleNamespace(stop_reason="insufficient_screen_budget"),
    )
    interrupted = SimpleNamespace(
        selected_config=None,
        search_result=SimpleNamespace(stop_reason="interrupted"),
    )

    assert SearchSweepResult((partial,)).exit_code == 0  # type: ignore[arg-type]
    assert SearchSweepResult((interrupted,)).exit_code == 130  # type: ignore[arg-type]


def test_resident_and_shape14_groups_partition_the_official_workload() -> None:
    project_root = Path(__file__).resolve().parents[1]
    all_shapes = load_shapes(project_root)
    resident = load_resident_shapes(project_root)
    streamed = load_streamed_shapes(project_root)

    assert (*resident, *streamed) == all_shapes
    assert {shape.case_id for shape in resident}.isdisjoint(
        shape.case_id for shape in streamed
    )
    assert _optimization_case_ids(project_root, "resident") == tuple(
        f"official_{index:02d}" for index in range(1, 14)
    )
    assert _optimization_case_ids(project_root, "shape14") == ("official_14",)
