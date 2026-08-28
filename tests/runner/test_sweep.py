"""Tests for the compact official-shape sweep service."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from runner.contracts import RunVariant, load_workload_set
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


def test_complete_sweep_summarizes_official_01_through_13_without_weights() -> None:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    runs = [successful_run(f"official_{index:02d}", 2.0) for index in range(1, 14)]

    summary = summarize_sweep(workload, runs, target="solution")

    assert summary["sweep_outcome"] == "complete"
    assert [item["case_id"] for item in summary["case_results"]] == [
        f"official_{index:02d}" for index in range(1, 14)
    ]
    assert summary["excluded_case_ids"] == ["official_14"]
    assert summary["geomean_speedup"] == 2.0
    assert "groups" not in summary
    assert "weights" not in summary


def test_incomplete_sweep_never_reports_an_aggregate() -> None:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    runs = [successful_run(f"official_{index:02d}", 1.1) for index in range(1, 13)]

    summary = summarize_sweep(workload, runs, target="solution")

    assert summary["sweep_outcome"] == "incomplete"
    assert summary["geomean_speedup"] is None
    assert summary["failed_cases"] == [
        {"case_id": "official_13", "outcome": "missing"}
    ]


def test_invalid_compact_timing_is_rejected_before_aggregation() -> None:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    runs = [successful_run(f"official_{index:02d}", 1.1) for index in range(1, 14)]
    runs[0]["performance"]["target"]["p90_ms"] = math.nan

    summary = summarize_sweep(workload, runs, target="solution")

    assert summary["sweep_outcome"] == "incomplete"
    assert summary["geomean_speedup"] is None
    assert summary["failed_cases"][0]["case_id"] == "official_01"
    assert summary["failed_cases"][0]["outcome"] == "invalid_target_p90"


def test_sweep_service_runs_only_local_shapes_and_owns_one_directory(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fake_run_shape(
        _project_root: Path,
        *,
        shape: Any,
        sweep_id: str,
        result_dir: Path,
        **_kwargs: Any,
    ) -> tuple[dict[str, Any], Path]:
        calls.append(shape.case_id)
        result = successful_run(shape.case_id, 1.25, sweep_id=sweep_id)
        path = result_dir / f"{shape.case_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result), encoding="utf-8")
        return result, path

    result = BenchmarkSweepService(run_shape=fake_run_shape).run(
        BenchmarkSweepRequest(
            project_root=PROJECT_ROOT,
            workload_set_id=WORKLOAD_SET_ID,
            protocol=tiny_protocol(),
            device="cpu",
            variant=RunVariant(),
            output_root=tmp_path,
            sweep_id="official-fixture",
        )
    )

    assert calls == [f"official_{index:02d}" for index in range(1, 14)]
    assert len(result.runs) == 13
    assert result.summary["sweep_outcome"] == "complete"
    assert result.summary_path == tmp_path / "official-fixture" / "summary.json"
    assert json.loads(result.summary_path.read_text(encoding="utf-8")) == result.summary


def test_sweep_cancellation_persists_one_incomplete_summary(tmp_path: Path) -> None:
    token = CancellationToken()

    def fake_run_shape(
        _project_root: Path,
        *,
        shape: Any,
        sweep_id: str,
        result_dir: Path,
        **_kwargs: Any,
    ) -> tuple[dict[str, Any], Path]:
        result = successful_run(shape.case_id, 1.0, sweep_id=sweep_id)
        path = result_dir / f"{shape.case_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result), encoding="utf-8")
        token.cancel()
        return result, path

    result = BenchmarkSweepService(run_shape=fake_run_shape).run(
        BenchmarkSweepRequest(
            project_root=PROJECT_ROOT,
            workload_set_id=WORKLOAD_SET_ID,
            protocol=tiny_protocol(),
            device="cpu",
            output_root=tmp_path,
            sweep_id="cancelled-fixture",
        ),
        cancellation_token=token,
    )

    assert len(result.runs) == 1
    assert result.summary["sweep_outcome"] == "cancelled"
    assert result.summary_path.exists()
