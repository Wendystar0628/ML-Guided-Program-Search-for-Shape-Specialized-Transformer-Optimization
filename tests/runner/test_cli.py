from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from runner import __main__ as runner_cli
from runner.calibration import CalibrationEvent
from runner.contracts import RunVariant, TransformerShape
from tests.support.runner_fixtures import WORKLOAD_SET_ID, successful_run


def test_cli_defaults_to_the_official_workload_and_float32_variant() -> None:
    parser = runner_cli.build_parser()

    benchmark = parser.parse_args(["benchmark"])
    profile = parser.parse_args(["profile", "--case-id", "official_13"])
    tune = parser.parse_args(
        [
            "tune",
            "--case-id",
            "official_02",
            "--candidate",
            "graph",
        ]
    )
    streamed = parser.parse_args(["benchmark-streamed"])

    assert benchmark.workload_set == WORKLOAD_SET_ID
    assert benchmark.dtype == "float32"
    assert benchmark.solution_policy == "dispatch"
    assert profile.case_id == "official_13"
    assert tune.case_id == ["official_02"]
    assert tune.candidate == ["graph"]
    assert streamed.case_id is None
    assert streamed.solution_policy == "screen"
    assert not hasattr(streamed, "target")


def test_tune_requires_an_explicit_candidate() -> None:
    parser = runner_cli.build_parser()

    with pytest.raises(SystemExit) as error:
        parser.parse_args(["tune", "--case-id", "official_02"])

    assert error.value.code == 2


def test_calibrate_rejects_official_14_as_a_configuration_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = runner_cli.main(
        ["calibrate", "--case-id", "official_14", "--plan-only"]
    )

    assert exit_code == 2
    assert "configuration error" in capsys.readouterr().out


def test_streamed_benchmark_rejects_a_resident_shape(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = runner_cli.main(["benchmark-streamed", "--case-id", "official_02"])

    assert exit_code == 2
    output = capsys.readouterr().out
    assert "configuration error" in output
    assert "resident=['official_02']" in output


def test_regular_single_case_benchmark_directs_streamed_shapes_to_separate_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = runner_cli.main(["benchmark", "--case-id", "official_14"])

    assert exit_code == 2
    output = capsys.readouterr().out
    assert "configuration error" in output
    assert "run benchmark-streamed instead" in output


def test_streamed_benchmark_prints_target_only_metrics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    shape = TransformerShape(
        case_id="streamed_fixture",
        batch_size=32,
        seq_len=100_000,
        d_model=1024,
        num_heads=16,
        ffn_dim=1024,
        num_layers=2,
        causal=True,
    )
    result = {
        "outcome": "success",
        "correctness": {"passed": True},
        "performance": {
            "comparison_mode": "target_only",
            "target": {"median_ms": 600.0, "p90_ms": 610.0},
            "achieved_tflops": 65.0,
            "peak_device_allocated_bytes": 3 * 1024**3,
        },
    }

    class FakeService:
        def run(
            self,
            request: Any,
            *,
            on_case_started: Any,
            on_case_completed: Any,
        ) -> Any:
            captured["request"] = request
            result_path = tmp_path / "streamed.json"
            on_case_started(1, 1, shape)
            on_case_completed(1, 1, shape, result, result_path)
            return SimpleNamespace(runs=(result,), result_paths=(result_path,))

    monkeypatch.setattr(runner_cli, "StreamedBenchmarkService", FakeService)

    exit_code = runner_cli.main(["benchmark-streamed", "--device", "cuda:0"])

    assert exit_code == 0
    assert captured["request"].case_ids == ()
    output = capsys.readouterr().out
    assert "[1/1] streamed streamed_fixture" in output
    assert "target median: 600.000000 ms" in output
    assert "useful matmul throughput: 65.000 TFLOP/s" in output
    assert "peak device allocation: 3.000 GiB" in output
    assert "baseline median" not in output
    assert "observed speedup" not in output


def test_tuning_summary_prints_target_only_winner_without_speedup(
    capsys: pytest.CaptureFixture[str],
) -> None:
    observation = {
        "candidate_id": "mixed-fp16-cudnn",
        "outcome": "success",
        "policy_applied": True,
        "target_median_ms": 550.0,
        "speedup": None,
        "execution_path": None,
        "correctness_passed": True,
        "failed_elements": 0,
        "max_abs_error": 0.0,
        "result_path": "results/intermediate/streamed/run.json",
    }
    runner_cli._print_tuning_summary(
        {
            "case_id": "official_14",
            "tuning_id": "fixture-tuning",
            "protocol": {"preset": "smoke"},
            "observations": [observation],
            "winner": observation,
        }
    )

    output = capsys.readouterr().out
    assert "mixed-fp16-cudnn: success | 550.000000 ms" in output
    assert "winner: mixed-fp16-cudnn | 550.000000 ms" in output
    assert "None" not in output


def test_single_shape_benchmark_forwards_shape_and_variant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(
        _project_root: Path,
        *,
        shape: TransformerShape,
        variant: RunVariant,
        **kwargs: Any,
    ):
        captured.update(shape=shape, variant=variant, kwargs=kwargs)
        return successful_run(shape.case_id, 1.0), tmp_path / "run.json"

    monkeypatch.setattr(runner_cli, "run_managed_benchmark", fake_run)

    exit_code = runner_cli.main(
        [
            "benchmark",
            "--case-id",
            "official_02",
            "--dtype",
            "float16",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert captured["shape"].case_id == "official_02"
    assert captured["variant"] == RunVariant(dtype="float16")
    assert captured["kwargs"]["solution_policy"] == "dispatch"


def test_full_benchmark_uses_the_shared_sweep_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    class FakeService:
        def run(self, request: Any, **_kwargs: Any) -> Any:
            captured["request"] = request
            return SimpleNamespace(
                summary={
                    "sweep_outcome": "complete",
                    "case_results": [],
                    "failed_cases": [],
                    "geomean_speedup": 1.0,
                },
                summary_path=tmp_path / "summary.json",
                runs=(),
            )

    monkeypatch.setattr(runner_cli, "BenchmarkSweepService", FakeService)

    assert runner_cli.main(["benchmark", "--device", "cpu"]) == 0
    assert captured["request"].workload_set_id == WORKLOAD_SET_ID
    assert captured["request"].variant == RunVariant()


def test_calibration_cli_renders_shape_based_formal_events(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    shape = TransformerShape(
        case_id="official_02",
        batch_size=1,
        seq_len=128,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    runner_cli._print_calibration_event(
        CalibrationEvent(
            kind="formal_plans_ready",
            data={
                "shapes": [shape],
                "plans": [{"candidate_order": ["graph", "eager-sdpa"]}],
            },
        )
    )
    runner_cli._print_calibration_event(
        CalibrationEvent(
            kind="promotion_completed",
            data={
                "shapes": [shape],
                "winners": [
                    {
                        "candidate_id": "graph",
                        "solution_policy": "graph",
                    }
                ],
                "route_path": tmp_path / "routes.json",
                "route_action": "updated verified package",
            },
        )
    )

    output = capsys.readouterr().out
    assert "official_02: graph, eager-sdpa" in output
    assert "official_02: deployed graph -> graph" in output


def test_routing_plan_renders_the_current_cost_model_signals(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner_cli._print_routing_plan(
        "official_02",
        {
            "source": "hardware_cost_prior",
            "routing_signals": {
                "machine_ridge_flops_per_byte": 71.25,
                "workload_intensity_to_ridge": 0.5,
                "dense_attention_to_l2": 1.25,
                "estimated_peak_to_device_memory": 0.125,
                "estimated_blocks_per_sm": 2.0,
            },
            "candidate_order": ["eager-sdpa", "graph"],
            "selection_reasons": {},
            "capability_rejections": {},
        },
    )

    output = capsys.readouterr().out
    assert "attention/L2=1.250" in output
    assert "peak/device-memory=0.125" in output
    assert "candidates: eager-sdpa, graph" in output
