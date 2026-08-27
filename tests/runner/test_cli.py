"""Tests for the thin command-line adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from runner import __main__ as runner_cli
from runner.contracts import MeasurementProtocol, WorkloadCase
from runner.sweep import BenchmarkSweepService
from tests.support.runner_fixtures import (
    EXPECTED_CASES,
    WORKLOAD_SET_ID,
    successful_run,
)


def test_cli_parses_public_commands() -> None:
    parser = runner_cli.build_parser()

    probe = parser.parse_args(["probe"])
    assert probe.command == "probe"
    assert probe.device == "cuda:0"
    assert probe.mode == "diagnostic"

    benchmark = parser.parse_args(["benchmark"])
    assert benchmark.command == "benchmark"
    assert benchmark.target == "solution"
    assert benchmark.workload_set == WORKLOAD_SET_ID
    assert benchmark.case_id is None
    assert benchmark.preset == "smoke"
    assert benchmark.solution_policy == "dispatch"

    profile = parser.parse_args(["profile", "--case-id", "attention_s2048_fp16"])
    assert profile.command == "profile"
    assert profile.case_id == "attention_s2048_fp16"
    assert profile.solution_policy == "dispatch"

    tune = parser.parse_args(
        [
            "tune",
            "--case-id",
            "launch_s64_fp16",
            "--candidate",
            "eager-auto",
        ]
    )
    assert tune.command == "tune"
    assert tune.case_id == ["launch_s64_fp16"]
    assert tune.candidate == ["eager-auto"]
    assert not hasattr(tune, "candidate_limit")

    calibrate = parser.parse_args(
        [
            "calibrate",
            "--case-id",
            "launch_s64_fp16",
            "--case-id",
            "wide_s256_bf16",
            "--candidate-limit",
            "2",
            "--matmul-precision",
            "highest",
            "--no-allow-tf32",
        ]
    )
    assert calibrate.command == "calibrate"
    assert calibrate.case_id == ["launch_s64_fp16", "wide_s256_bf16"]
    assert calibrate.candidate_limit == 2
    assert calibrate.matmul_precision == "highest"
    assert calibrate.allow_tf32 is False
    assert calibrate.plan_only is False

def test_tune_requires_explicit_candidates() -> None:
    parser = runner_cli.build_parser()

    with pytest.raises(SystemExit) as missing_candidate:
        parser.parse_args(["tune", "--case-id", "balanced_s128_fp16"])

    assert missing_candidate.value.code == 2


def test_single_case_benchmark_cli_dispatches_supervisor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str | None, str | None]] = []
    result_dir = Path("verified_hardware/fixture/results/runs")

    def fake_run_managed_benchmark(
        project_root: Path,
        *,
        workload_set_id: str,
        case: WorkloadCase,
        protocol: MeasurementProtocol,
        device: str,
        target: str,
        workload_sha256: str | None,
        sweep_id: str | None = None,
        result_dir: Path | None = None,
        solution_policy: str | None = None,
    ) -> tuple[dict[str, Any], Path]:
        del project_root, protocol, device
        assert workload_set_id == WORKLOAD_SET_ID
        assert target == "solution"
        assert solution_policy == "dispatch"
        assert result_dir == Path("verified_hardware/fixture/results/runs")
        calls.append((case.case_id, workload_sha256, sweep_id))
        return successful_run(
            case.case_id,
            2.0,
            sweep_id=sweep_id or "single-case",
        ), tmp_path / f"{case.case_id}.json"

    monkeypatch.setattr(runner_cli, "run_managed_benchmark", fake_run_managed_benchmark)

    assert (
        runner_cli.main(
            [
                "benchmark",
                "--case-id",
                "balanced_s128_fp16",
                "--result-dir",
                str(result_dir),
            ]
        )
        == 0
    )
    assert [case_id for case_id, _, _ in calls] == ["balanced_s128_fp16"]
    assert calls[0][2] is None
    assert all(workload_hash for _, workload_hash, _ in calls)


def test_full_benchmark_cli_dispatches_sweep_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, Path]] = []

    def fake_run_managed_benchmark(
        _project_root: Path,
        **arguments: Any,
    ) -> tuple[dict[str, Any], Path]:
        case = arguments["case"]
        sweep_id = arguments["sweep_id"]
        result_dir = arguments["result_dir"]
        calls.append((case.case_id, sweep_id, result_dir))
        return (
            successful_run(case.case_id, 2.0, sweep_id=sweep_id),
            result_dir / f"{case.case_id}.json",
        )

    monkeypatch.setattr(
        runner_cli,
        "BenchmarkSweepService",
        lambda: BenchmarkSweepService(fake_run_managed_benchmark),
    )
    output_root = tmp_path / "sweeps"

    assert (
        runner_cli.main(["benchmark", "--result-dir", str(output_root)])
        == 0
    )
    assert [case_id for case_id, _, _ in calls] == [
        case[0] for case in EXPECTED_CASES
    ]
    sweep_ids = {sweep_id for _, sweep_id, _ in calls}
    assert len(sweep_ids) == 1
    sweep_id = next(iter(sweep_ids))
    assert {directory for _, _, directory in calls} == {
        (output_root / sweep_id / "runs").resolve()
    }
    assert (output_root / sweep_id / "summary.json").is_file()
