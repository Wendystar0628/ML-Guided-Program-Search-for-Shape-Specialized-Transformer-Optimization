from __future__ import annotations

import json

import pytest

from autotune.evaluation import (
    ConstraintVector,
    EvaluationScope,
    Fidelity,
    PairedMeasurement,
    TrialMeasurement,
)
from autotune.optimization_loop import OptimizationIteration
from autotune.run_log import SearchRunLog
from autotune.search_engine import (
    SearchResult,
    SearchStageTimings,
    _screen_failure_kind,
)
from autotune.search_sweep import SearchSweepResult, ShapeSearchResult
from solution.config import ConfigSpec, RuntimeBackend, ScheduleConfig, portable_config


def _measurement(
    config: ConfigSpec,
    latency_ms: float,
    fidelity: Fidelity,
) -> TrialMeasurement:
    return TrialMeasurement(
        config_id=config.config_id,
        fidelity=fidelity,
        scope=EvaluationScope.RESIDENT,
        objective_ms=latency_ms,
        median_ms=latency_ms,
        p90_ms=latency_ms * 1.02,
        max_tolerance_ratio=0.4,
        constraints=ConstraintVector(),
    )


def _shape_result() -> ShapeSearchResult:
    incumbent = portable_config()
    challenger = ConfigSpec(
        program=incumbent.program,
        schedule=ScheduleConfig(runtime=RuntimeBackend.CUDA_GRAPH),
    )
    incumbent_formal = _measurement(incumbent, 1.2, Fidelity.FORMAL)
    challenger_formal = _measurement(challenger, 1.0, Fidelity.FORMAL)
    comparison = PairedMeasurement(
        incumbent=incumbent_formal,
        challenger=challenger_formal,
        paired_ratios=(1.2,) * 13,
    )
    return ShapeSearchResult(
        case_id="official_01",
        search_result=SearchResult(
            incumbent_config=incumbent,
            selected_config=challenger,
            selected_measurement=challenger_formal,
            branch_count=4,
            completed_level1=12,
            enhanced_measurements=(_measurement(challenger, 1.01, Fidelity.ENHANCED),),
            locked_challenger=challenger,
            formal_challenger_measurement=challenger_formal,
            formal_comparison=comparison,
            stop_reason="completed",
            new_level1_trials=3,
            feasible_level1=9,
            best_screen_config_id=challenger.config_id,
            best_screen_median_ms=0.98,
            screen_failure_counts=(("accuracy_constraint", 2),),
            stage_timings=SearchStageTimings(
                planning=0.1,
                screen=4.0,
                enhanced=2.0,
                formal=4.5,
                total=10.6,
            ),
            budget_seconds=10.0,
            covered_branches=4,
            mandatory_coverage_complete=True,
        ),
        deployment_updated=True,
    )


def _read_events(path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_run_log_keeps_only_replayable_search_decisions(tmp_path) -> None:
    shape = _shape_result()
    sweep = SearchSweepResult((shape,))
    run_log = SearchRunLog(
        root=tmp_path,
        mode="optimize",
        target="resident",
        request={
            "case_ids": ["official_01"],
            "device": "cuda:0",
            "seed": 1234,
            "study_database": "observations/search/resident/search.sqlite3",
        },
    )

    run_log.record_shape(shape)
    run_log.record_iteration(
        OptimizationIteration(
            index=1,
            search_result=sweep,
            deployment_updates=1,
            shapes_with_level1_progress=1,
            no_deployment_streak=0,
        )
    )
    run_log.finish(
        status="finished",
        stop_reason="max_iterations",
        exit_code=0,
        iterations=1,
        total_deployment_updates=1,
    )

    events = _read_events(run_log.path)
    assert [event["event"] for event in events] == [
        "run_started",
        "shape_finished",
        "iteration_finished",
        "run_finished",
    ]
    shape_event = events[1]
    assert shape_event["screen"]["new_trials"] == 3
    assert shape_event["screen"]["failure_counts_total"] == {"accuracy_constraint": 2}
    assert shape_event["screen"]["scheduler"]["algorithm"] == (
        "fixed_budget_survivor_tpe"
    )
    assert shape_event["timing_seconds"] == {
        "planning": 0.1,
        "screen": 4.0,
        "enhanced": 2.0,
        "formal": 4.5,
        "total": 10.6,
    }
    assert shape_event["budget_seconds"] == 10.0
    assert shape_event["overrun_seconds"] == pytest.approx(0.5)
    assert shape_event["screen"]["covered_branches"] == 4
    assert shape_event["screen"]["mandatory_coverage_complete"] is True
    decision = shape_event["decision"]
    assert decision["outcome"] == "published"
    assert decision["selected_config_id"] == shape.selected_config.config_id
    assert decision["incumbent"]["program"] == (
        shape.search_result.incumbent_config.to_dict()["program"]
    )
    assert decision["incumbent"]["config_id"] == (
        shape.search_result.incumbent_config.config_id
    )
    assert decision["challenger"]["program"] == (
        shape.search_result.locked_challenger.to_dict()["program"]
    )
    assert decision["challenger"]["config_id"] == (
        shape.search_result.locked_challenger.config_id
    )
    assert decision["challenger"]["schedule"] == (
        shape.search_result.locked_challenger.to_dict()["schedule"]
    )
    assert shape_event["decision"]["formal"]["promotion_decision"] == "promote"
    assert len(shape_event["decision"]["formal"]["paired_ratios"]) == 13
    assert len(shape_event["enhanced"]) == 1
    assert events[2]["no_deployment_streak"] == 0
    assert events[-1]["status"] == "finished"
    assert all(event["schema_version"] == 1 for event in events)
    assert all(event["run_id"] == run_log.path.stem for event in events)

    forbidden_keys = {
        "config",
        "metrics",
        "round_medians_ms",
        "expected_execution_signature",
        "actual_execution_signature",
        "incumbent_config_id",
        "challenger_config_id",
        "deployment_approved",
        "deployment_updated",
        "rounds_used",
        "promotion_wins",
    }

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value), set())
        return set()

    assert keys(events).isdisjoint(forbidden_keys)
    assert run_log.path.stat().st_size < 10_000


def test_run_log_persists_a_short_failure_without_traceback(tmp_path) -> None:
    run_log = SearchRunLog(
        root=tmp_path,
        mode="search",
        target="official_01",
        request={"case_ids": ["official_01"]},
    )

    run_log.fail(RuntimeError("x" * 800))

    failure = _read_events(run_log.path)[-1]
    assert failure["event"] == "run_failed"
    assert failure["error_type"] == "RuntimeError"
    assert len(failure["error_message"]) == 500
    assert "traceback" not in failure


def test_decision_records_approved_unchanged_and_short_measurement_failure(
    tmp_path,
) -> None:
    approved = _shape_result()
    run_log = SearchRunLog(
        root=tmp_path,
        mode="search",
        target="official_01",
        request={"case_ids": ["official_01"]},
    )
    run_log.record_shape(
        ShapeSearchResult(
            approved.case_id,
            approved.search_result,
            deployment_updated=False,
        )
    )
    assert _read_events(run_log.path)[1]["decision"]["outcome"] == (
        "approved_unchanged"
    )

    incumbent = portable_config()
    challenger = ConfigSpec(
        program=incumbent.program,
        schedule=ScheduleConfig(runtime=RuntimeBackend.CUDA_GRAPH),
    )
    failed_formal = TrialMeasurement.infeasible(
        config_id=challenger.config_id,
        fidelity=Fidelity.FORMAL,
        scope=EvaluationScope.RESIDENT,
        penalty_ms=1_000_000_000.0,
        constraints=ConstraintVector(runtime=1.0),
        failure_kind="runtime_resource_exhausted",
        metrics={"message": "x" * 800},
    )
    rejected = ShapeSearchResult(
        case_id="official_01",
        search_result=SearchResult(
            incumbent_config=incumbent,
            selected_config=incumbent,
            selected_measurement=None,
            branch_count=1,
            completed_level1=1,
            enhanced_measurements=(),
            locked_challenger=challenger,
            formal_challenger_measurement=failed_formal,
            formal_comparison=None,
            stop_reason="no_feasible_formal",
        ),
        deployment_updated=False,
    )
    run_log.record_shape(rejected)

    decision = _read_events(run_log.path)[2]["decision"]
    assert decision["outcome"] == "rejected"
    failure = decision["formal"]["challenger"]
    assert failure["failure_kind"] == "runtime_resource_exhausted"
    assert failure["failure_message"] == "x" * 500


def test_screen_failure_summary_preserves_the_primary_constraint() -> None:
    config = portable_config()

    def infeasible(
        constraints: ConstraintVector,
        failure_kind: str,
    ) -> TrialMeasurement:
        return TrialMeasurement.infeasible(
            config_id=config.config_id,
            fidelity=Fidelity.SCREEN,
            scope=EvaluationScope.RESIDENT,
            penalty_ms=1_000_000_000.0,
            constraints=constraints,
            failure_kind=failure_kind,
        )

    assert (
        _screen_failure_kind(
            infeasible(ConstraintVector(accuracy=0.2), "constraint_violation")
        )
        == "accuracy_constraint"
    )
    assert (
        _screen_failure_kind(
            infeasible(ConstraintVector(execution_path=1.0), "constraint_violation")
        )
        == "execution_path_constraint"
    )
    assert (
        _screen_failure_kind(infeasible(ConstraintVector(runtime=1.0), "out_of_memory"))
        == "out_of_memory"
    )


def test_run_log_keeps_no_candidate_and_interrupted_outcomes(tmp_path) -> None:
    incumbent = portable_config()
    shape = ShapeSearchResult(
        case_id="official_01",
        search_result=SearchResult(
            incumbent_config=incumbent,
            selected_config=incumbent,
            selected_measurement=None,
            branch_count=3,
            completed_level1=3,
            enhanced_measurements=(),
            locked_challenger=None,
            formal_challenger_measurement=None,
            formal_comparison=None,
            stop_reason="no_feasible_screen",
            new_level1_trials=3,
        ),
        deployment_updated=False,
    )
    run_log = SearchRunLog(
        root=tmp_path,
        mode="search",
        target="official_01",
        request={"case_ids": ["official_01"]},
    )

    run_log.record_shape(shape)
    run_log.finish(status="interrupted", exit_code=130)

    events = _read_events(run_log.path)
    decision = events[1]["decision"]
    assert events[1]["stop_reason"] == "no_feasible_screen"
    assert decision["challenger"] is None
    assert decision["outcome"] == "not_run"
    assert events[-1]["status"] == "interrupted"
    assert events[-1]["exit_code"] == 130
