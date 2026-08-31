from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from autotune.optimization_loop import OptimizationLoop, OptimizationLoopPolicy
from autotune.search_sweep import SearchSweepRequest, SearchSweepResult
from benchmarking.protocols import (
    ContractError,
    load_resident_shapes,
    load_shapes,
    load_streamed_shapes,
)
from cli import _optimization_case_ids, build_parser


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
                        made_level1_progress=made_level1_progress,
                        level1_space_exhausted=level1_space_exhausted,
                    ),
                )
                for updated, made_level1_progress, level1_space_exhausted in shapes
            ),
        )


def _request() -> SearchSweepRequest:
    return SearchSweepRequest(
        project_root=Path("."),
        case_ids=("official_01",),
        budget_seconds=1.0,
        seed=100,
        structure_seed=777,
    )


def test_cli_exposes_only_deployment_patience() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "optimize",
            "--group",
            "resident",
            "--no-deployment-patience",
            "3",
        ]
    )

    assert args.no_deployment_patience == 3
    seeded = parser.parse_args(
        [
            "optimize",
            "--group",
            "resident",
            "--seed",
            "100",
            "--structure-seed",
            "777",
        ]
    )
    assert seeded.seed == 100
    assert seeded.structure_seed == 777
    selected = parser.parse_args(
        [
            "optimize",
            "--group",
            "resident",
            "--case-id",
            "official_01",
            "--case-id",
            "official_07",
        ]
    )
    assert selected.case_id == ["official_01", "official_07"]
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "optimize",
                "--group",
                "resident",
                "--no-progress-patience",
                "3",
            ]
        )


def test_deployment_patience_allows_a_later_deployment() -> None:
    search = _SearchSweep(
        [
            (0, ((False, True, False),)),
            (0, ((True, True, False),)),
        ]
    )

    result = OptimizationLoop(search).run(  # type: ignore[arg-type]
        _request(),
        OptimizationLoopPolicy(no_deployment_patience=2, max_iterations=2),
    )

    assert result.stop_reason == "max_iterations"
    assert result.iterations_run == 2
    assert result.total_deployment_updates == 1
    assert result.no_deployment_streak == 0
    assert [request.seed for request in search.requests] == [100, 101]
    assert [request.structure_seed for request in search.requests] == [777, 777]


def test_new_screen_evidence_resets_deployment_patience() -> None:
    search = _SearchSweep(
        [
            (0, ((False, True, False),)),
            (0, ((False, True, False),)),
            (0, ((True, True, False),)),
        ]
    )

    result = OptimizationLoop(search).run(  # type: ignore[arg-type]
        _request(),
        OptimizationLoopPolicy(no_deployment_patience=2, max_iterations=3),
    )

    assert result.stop_reason == "max_iterations"
    assert result.iterations_run == 3
    assert result.total_deployment_updates == 1
    assert result.no_deployment_streak == 0
    assert len(search.requests) == 3


def test_failed_historical_formal_does_not_reset_deployment_patience() -> None:
    class _HistoricalFormalSweep:
        def run(self, request: SearchSweepRequest) -> object:
            return SimpleNamespace(
                exit_code=0,
                shape_results=(
                    SimpleNamespace(
                        deployment_updated=False,
                        search_result=SimpleNamespace(
                            made_level1_progress=False,
                            level1_space_exhausted=False,
                        ),
                    ),
                ),
            )

    result = OptimizationLoop(_HistoricalFormalSweep()).run(  # type: ignore[arg-type]
        _request(),
        OptimizationLoopPolicy(no_deployment_patience=1, max_iterations=2),
    )

    assert result.stop_reason == "no_deployment_patience"
    assert result.iterations_run == 1
    assert result.no_deployment_streak == 1


def test_hard_iteration_limit_stops_continuous_deployments() -> None:
    search = _SearchSweep([(0, ((True, False, False),))] * 3)

    result = OptimizationLoop(search).run(  # type: ignore[arg-type]
        _request(),
        OptimizationLoopPolicy(no_deployment_patience=2, max_iterations=3),
    )

    assert result.stop_reason == "max_iterations"
    assert result.iterations_run == 3
    assert result.total_deployment_updates == 3
    assert result.no_deployment_streak == 0


def test_exhausted_screen_does_not_block_a_historical_deployment() -> None:
    class _ExhaustedSweep(_SearchSweep):
        def run(self, request: SearchSweepRequest) -> object:
            result = super().run(request)
            for item in result.shape_results:
                item.search_result.made_level1_progress = False
                item.search_result.level1_space_exhausted = True
            return result

    result = OptimizationLoop(
        _ExhaustedSweep(
            [
                (0, ((False, False, True),)),
                (0, ((True, False, True),)),
            ]
        )
    ).run(
        _request(),
        OptimizationLoopPolicy(no_deployment_patience=2, max_iterations=2),
    )

    assert result.stop_reason == "max_iterations"
    assert result.iterations_run == 2
    assert result.total_deployment_updates == 1
    assert result.no_deployment_streak == 0


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
            OptimizationLoopPolicy(no_deployment_patience=2, max_iterations=4),
        )

        assert result.stop_reason == expected_reason
        assert result.exit_code == exit_code
        assert result.iterations_run == 1
        assert result.no_deployment_streak == 0
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
    assert _optimization_case_ids(
        project_root,
        "resident",
        ("official_01", "official_07"),
    ) == ("official_01", "official_07")


def test_optimization_case_selection_rejects_duplicates_and_wrong_group() -> None:
    project_root = Path(__file__).resolve().parents[1]
    with pytest.raises(ContractError, match="must be unique"):
        _optimization_case_ids(
            project_root,
            "resident",
            ("official_01", "official_01"),
        )
    with pytest.raises(ContractError, match="do not belong"):
        _optimization_case_ids(project_root, "resident", ("official_14",))
