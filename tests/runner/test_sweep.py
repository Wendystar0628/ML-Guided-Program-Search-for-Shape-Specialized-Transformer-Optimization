"""Aggregation tests for ordered workload sweeps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from runner.contracts import ContractError, load_workload_set
from runner.supervisor import CancellationToken
from runner.sweep import (
    BenchmarkSweepRequest,
    BenchmarkSweepService,
    summarize_sweep,
)
from tests.support.runner_fixtures import (
    PROJECT_ROOT,
    WORKLOAD_SET_ID,
    successful_run,
    tiny_protocol,
)


def test_sweep_uses_equal_group_weights() -> None:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    speedups = {
        case.case_id: (8.0 if case.case_id.startswith("mask_s512") else 2.0)
        for case in workload.cases
    }
    runs = [
        successful_run(case.case_id, speedups[case.case_id]) for case in workload.cases
    ]

    summary = summarize_sweep(workload, runs, target="solution")

    assert summary["sweep_outcome"] == "complete"
    assert [group["geomean_speedup"] for group in summary["groups"]] == pytest.approx(
        [2.0, 2.0, 2.0, 8.0, 2.0]
    )
    assert summary["group_balanced_geomean_speedup"] == pytest.approx(
        (2.0**0.8) * (8.0**0.2)
    )
    assert summary["worst_case_speedup"] == 2.0


@pytest.mark.parametrize(
    "replacement",
    [
        None,
        {"outcome": "oom"},
        {"outcome": "success", "speedup": 0.0},
        {"outcome": "success", "speedup": float("nan")},
        {"outcome": "success", "speedup": float("inf")},
    ],
    ids=("missing", "failed", "zero", "nan", "infinity"),
)
def test_incomplete_sweep_never_reports_aggregate(
    replacement: dict[str, Any] | None,
) -> None:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    runs = [successful_run(case.case_id, 2.0) for case in workload.cases]
    if replacement is None:
        runs.pop(3)
    else:
        run = runs[3]
        run["outcome"] = replacement["outcome"]
        if "speedup" in replacement:
            run["performance"]["speedup"] = replacement["speedup"]

    summary = summarize_sweep(workload, runs, target="solution")

    assert summary["sweep_outcome"] == "incomplete"
    assert summary["groups"] == []
    assert summary["group_balanced_geomean_speedup"] is None
    assert summary["worst_case_speedup"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timer", "perf_counter_ns"),
        ("sample_count", 2),
        ("baseline_p90", 0.0),
        ("baseline_p90", 1.0),
        ("target_round_medians", []),
        ("speedup", 3.0),
    ],
    ids=(
        "non-cuda-timer",
        "sample-count",
        "invalid-p90",
        "p90-below-median",
        "round-count",
        "speedup-mismatch",
    ),
)
def test_incomplete_sweep_rejects_invalid_compact_statistics(
    field: str,
    value: Any,
) -> None:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    runs = [successful_run(case.case_id, 2.0) for case in workload.cases]
    performance = runs[3]["performance"]
    if field == "timer":
        performance["timer"] = value
    elif field == "sample_count":
        performance["sample_count"] = value
    elif field == "baseline_p90":
        performance["baseline"]["p90_ms"] = value
    elif field == "target_round_medians":
        performance["target"]["round_medians_ms"] = value
    else:
        performance["speedup"] = value

    summary = summarize_sweep(workload, runs, target="solution")

    assert summary["sweep_outcome"] == "incomplete"
    assert summary["groups"] == []
    assert summary["group_balanced_geomean_speedup"] is None
    assert summary["worst_case_speedup"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trial_count", 0),
        ("failed_elements", 1),
        ("max_abs_error", float("nan")),
    ],
)
def test_incomplete_sweep_rejects_invalid_correctness_summary(
    field: str,
    value: Any,
) -> None:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    runs = [successful_run(case.case_id, 2.0) for case in workload.cases]
    runs[3]["correctness"][field] = value

    summary = summarize_sweep(workload, runs, target="solution")

    assert summary["sweep_outcome"] == "incomplete"
    assert summary["groups"] == []
    assert summary["group_balanced_geomean_speedup"] is None
    assert summary["worst_case_speedup"] is None


def test_sweep_service_owns_one_isolated_directory_and_summary(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, Path]] = []

    def fake_run_case(
        project_root: Path,
        **arguments: Any,
    ) -> tuple[dict[str, Any], Path]:
        case = arguments["case"]
        sweep_id = arguments["sweep_id"]
        result_dir = arguments["result_dir"]
        assert project_root == PROJECT_ROOT.resolve()
        run = successful_run(case.case_id, 2.0, sweep_id=sweep_id)
        run_path = result_dir / f"{case.case_id}.json"
        run_path.parent.mkdir(parents=True, exist_ok=True)
        run_path.write_text(json.dumps(run), encoding="utf-8")
        calls.append((case.case_id, sweep_id, result_dir))
        return run, run_path

    output_root = tmp_path / "sweeps"
    result = BenchmarkSweepService(fake_run_case).run(
        BenchmarkSweepRequest(
            project_root=PROJECT_ROOT,
            workload_set_id=WORKLOAD_SET_ID,
            protocol=tiny_protocol(),
            device="cuda:0",
            output_root=output_root,
            sweep_id="owned-sweep",
        )
    )

    expected_directory = output_root / "owned-sweep"
    assert result.sweep_directory == expected_directory.resolve()
    assert result.summary_path == expected_directory.resolve() / "summary.json"
    assert {sweep_id for _, sweep_id, _ in calls} == {"owned-sweep"}
    assert {directory for _, _, directory in calls} == {
        expected_directory.resolve() / "runs"
    }
    persisted = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert persisted == result.summary
    assert persisted["sweep_id"] == "owned-sweep"
    assert persisted["sweep_outcome"] == "complete"
    assert [item["run_id"] for item in persisted["case_results"]] == [
        f"fixture-{case.case_id}"
        for case in load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID).cases
    ]
    assert all("run_file" not in item for item in persisted["case_results"])


def test_sweep_validator_runs_before_summary_is_persisted(tmp_path: Path) -> None:
    def fake_run_case(
        _project_root: Path,
        **arguments: Any,
    ) -> tuple[dict[str, Any], Path]:
        case = arguments["case"]
        result_dir = arguments["result_dir"]
        run = successful_run(case.case_id, 2.0, sweep_id=arguments["sweep_id"])
        run_path = result_dir / f"{case.case_id}.json"
        run_path.parent.mkdir(parents=True, exist_ok=True)
        run_path.write_text(json.dumps(run), encoding="utf-8")
        return run, run_path

    def reject(*_arguments: Any) -> None:
        raise ContractError("fixture rejection")

    output_root = tmp_path / "sweeps"
    with pytest.raises(ContractError, match="fixture rejection"):
        BenchmarkSweepService(fake_run_case).run(
            BenchmarkSweepRequest(
                project_root=PROJECT_ROOT,
                workload_set_id=WORKLOAD_SET_ID,
                protocol=tiny_protocol(),
                device="cuda:0",
                output_root=output_root,
                sweep_id="rejected-sweep",
            ),
            validate_before_persist=reject,
        )

    assert not (output_root / "rejected-sweep" / "summary.json").exists()


def test_sweep_cancellation_persists_one_incomplete_summary(tmp_path: Path) -> None:
    token = CancellationToken()
    calls: list[str] = []

    def fake_run_case(
        _project_root: Path,
        **arguments: Any,
    ) -> tuple[dict[str, Any], Path]:
        assert arguments["cancellation_token"] is token
        case = arguments["case"]
        calls.append(case.case_id)
        run = successful_run(
            case.case_id,
            2.0,
            sweep_id=arguments["sweep_id"],
        )
        run_path = arguments["result_dir"] / f"{case.case_id}.json"
        run_path.parent.mkdir(parents=True, exist_ok=True)
        run_path.write_text(json.dumps(run), encoding="utf-8")
        token.cancel()
        return run, run_path

    result = BenchmarkSweepService(fake_run_case).run(
        BenchmarkSweepRequest(
            project_root=PROJECT_ROOT,
            workload_set_id=WORKLOAD_SET_ID,
            protocol=tiny_protocol(),
            device="cuda:0",
            output_root=tmp_path / "sweeps",
            sweep_id="cancelled-sweep",
        ),
        cancellation_token=token,
        validate_before_persist=lambda *args: pytest.fail(
            "a cancelled sweep must persist before complete-run validation"
        ),
    )

    assert calls == [load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID).cases[0].case_id]
    assert len(result.runs) == 1
    assert result.summary["sweep_id"] == "cancelled-sweep"
    assert result.summary["sweep_outcome"] == "cancelled"
    assert result.summary["group_balanced_geomean_speedup"] is None
    assert json.loads(result.summary_path.read_text(encoding="utf-8")) == result.summary
