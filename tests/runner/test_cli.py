"""Tests for the thin command-line adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from runner import __main__ as runner_cli
from runner.contracts import MeasurementProtocol, WorkloadCase
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

    promote = parser.parse_args(
        [
            "promote",
            "--tuning-id",
            "fixture-tuning",
            "--route-table",
            "verified_hardware/fixture/routes.json",
        ]
    )
    assert promote.command == "promote"
    assert promote.tuning_id == ["fixture-tuning"]
    assert promote.route_table == Path("verified_hardware/fixture/routes.json")


def test_tune_requires_explicit_candidates() -> None:
    parser = runner_cli.build_parser()

    with pytest.raises(SystemExit) as missing_candidate:
        parser.parse_args(["tune", "--case-id", "balanced_s128_fp16"])

    assert missing_candidate.value.code == 2


@pytest.mark.parametrize(
    ("extra_arguments", "expected_ids", "expected_result_dir"),
    [
        (
            [
                "--case-id",
                "balanced_s128_fp16",
                "--result-dir",
                "verified_hardware/fixture/results/runs",
            ],
            ["balanced_s128_fp16"],
            Path("verified_hardware/fixture/results/runs"),
        ),
        ([], [case[0] for case in EXPECTED_CASES], None),
    ],
    ids=("single-case", "ordered-sweep"),
)
def test_benchmark_cli_dispatches_public_runner_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra_arguments: list[str],
    expected_ids: list[str],
    expected_result_dir: Path | None,
) -> None:
    calls: list[tuple[str, str | None, str | None]] = []

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
        assert result_dir == expected_result_dir
        calls.append((case.case_id, workload_sha256, sweep_id))
        return successful_run(
            case.case_id,
            2.0,
            sweep_id=sweep_id or "single-case",
        ), tmp_path / f"{case.case_id}.json"

    monkeypatch.setattr(runner_cli, "run_managed_benchmark", fake_run_managed_benchmark)

    assert runner_cli.main(["benchmark", *extra_arguments]) == 0
    assert [case_id for case_id, _, _ in calls] == expected_ids
    if len(expected_ids) == 1:
        assert calls[0][2] is None
    else:
        sweep_ids = {sweep_id for _, _, sweep_id in calls}
        assert len(sweep_ids) == 1
        assert None not in sweep_ids
    assert all(workload_hash for _, workload_hash, _ in calls)
