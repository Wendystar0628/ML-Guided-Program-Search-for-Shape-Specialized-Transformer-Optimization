"""Aggregation tests for ordered workload sweeps."""

from __future__ import annotations

from typing import Any

import pytest

from runner.contracts import load_workload_set
from runner.sweep import summarize_sweep
from tests.support.runner_fixtures import (
    PROJECT_ROOT,
    WORKLOAD_SET_ID,
    successful_run,
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
