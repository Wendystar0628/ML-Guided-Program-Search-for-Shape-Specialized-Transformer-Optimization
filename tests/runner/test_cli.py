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
            "causal-sdpa",
        ]
    )

    assert benchmark.workload_set == WORKLOAD_SET_ID
    assert benchmark.dtype == "float32"
    assert benchmark.solution_policy == "dispatch"
    assert profile.case_id == "official_13"
    assert tune.case_id == ["official_02"]
    assert tune.candidate == ["causal-sdpa"]


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
                "plans": [{"candidate_order": ["graph", "eager-auto"]}],
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
    assert "official_02: graph, eager-auto" in output
    assert "official_02: deployed graph -> graph" in output
