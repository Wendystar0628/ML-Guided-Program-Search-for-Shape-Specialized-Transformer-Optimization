"""High-value regression tests for the performance development loop."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from official import torch_transformer_benchmark as official
from runner import __main__ as runner_cli
from runner import supervisor
from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    WorkloadCase,
    load_workload_set,
    validate_official_snapshot,
)
from runner.execution import run_performance
from runner.supervisor import run_managed_benchmark
from runner.sweep import summarize_sweep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_OFFICIAL_SHA256 = (
    "1630fe39ebc845beeaef73aaaf2d47e061fc56fd20777706c3ddc961664c266b"
)
WORKLOAD_SET_ID = "rtx4080_core_v1"
EXPECTED_CASES = (
    ("launch_s64_fp16", 1, 64, 256, 8, 1024, 4, "float16", False, 0.0),
    ("balanced_s128_fp32", 8, 128, 512, 8, 2048, 6, "float32", False, 0.0),
    ("balanced_s128_fp16", 8, 128, 512, 8, 2048, 6, "float16", False, 0.0),
    ("attention_s2048_fp16", 1, 2048, 512, 8, 2048, 4, "float16", False, 0.0),
    (
        "attention_s2048_causal_fp16",
        1,
        2048,
        512,
        8,
        2048,
        4,
        "float16",
        True,
        0.0,
    ),
    ("mask_s512_full_fp16", 8, 512, 512, 8, 2048, 4, "float16", False, 0.0),
    (
        "mask_s512_padding_fp16",
        8,
        512,
        512,
        8,
        2048,
        4,
        "float16",
        False,
        0.75,
    ),
    (
        "mask_s512_causal_padding_fp16",
        8,
        512,
        512,
        8,
        2048,
        4,
        "float16",
        True,
        0.75,
    ),
    ("wide_s256_bf16", 16, 256, 1024, 8, 4096, 6, "bfloat16", False, 0.0),
)
EXPECTED_GROUPS = (
    ("launch_graph", 0.2, ("launch_s64_fp16",)),
    (
        "balanced_precision",
        0.2,
        ("balanced_s128_fp32", "balanced_s128_fp16"),
    ),
    (
        "long_attention",
        0.2,
        ("attention_s2048_fp16", "attention_s2048_causal_fp16"),
    ),
    (
        "padding_mask",
        0.2,
        (
            "mask_s512_full_fp16",
            "mask_s512_padding_fp16",
            "mask_s512_causal_padding_fp16",
        ),
    ),
    ("wide_gemm_ffn", 0.2, ("wide_s256_bf16",)),
)


def _tiny_case(*, causal: bool = True, padding_ratio: float = 0.5) -> WorkloadCase:
    return WorkloadCase(
        case_id="tiny_cpu_smoke",
        batch_size=1,
        seq_len=2,
        d_model=4,
        num_heads=1,
        ffn_dim=8,
        num_layers=1,
        dtype="float32",
        causal=causal,
        padding_ratio=padding_ratio,
        input_scale=1.0,
    )


def _tiny_protocol() -> MeasurementProtocol:
    return MeasurementProtocol(
        preset="smoke",
        seed=17,
        accuracy_trials=1,
        warmup=0,
        repeats=1,
        rounds=1,
        matmul_precision="highest",
        allow_tf32=False,
        timeout_seconds=60.0,
    )


def _copy_runtime_project(tmp_path: Path) -> Path:
    project = tmp_path / "managed-project"
    project.mkdir()
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    for directory in ("official", "runner", "solution"):
        shutil.copytree(
            PROJECT_ROOT / directory,
            project / directory,
            ignore=ignored,
        )
    return project


def _successful_run(
    case_id: str,
    speedup: float,
    *,
    sweep_id: str = "fixture-sweep",
) -> dict[str, Any]:
    workload_set = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    case = next(case for case in workload_set["cases"] if case.case_id == case_id)
    target_median = 2.0 / speedup
    return {
        "schema_version": 2,
        "run_kind": "benchmark",
        "target": "solution",
        "sweep_id": sweep_id,
        "outcome": "success",
        "source": {
            "official_sha256": EXPECTED_OFFICIAL_SHA256,
            "solution_sha256": "fixture-solution-hash",
        },
        "workload": {
            "set_id": WORKLOAD_SET_ID,
            "sha256": workload_set["sha256"],
            "case": case.as_dict(),
        },
        "protocol": {"accuracy_trials": 1, "repeats": 1, "rounds": 1},
        "environment": {
            "device": "cuda:0",
            "gpu": "fixture-gpu",
        },
        "correctness": {
            "passed": True,
            "trial_count": 1,
            "failed_elements": 0,
            "max_abs_error": 0.0,
        },
        "performance": {
            "timer": "cuda_event",
            "sample_count": 1,
            "baseline": {
                "median_ms": 2.0,
                "p90_ms": 2.0,
                "round_medians_ms": [2.0],
            },
            "target": {
                "median_ms": target_median,
                "p90_ms": target_median,
                "round_medians_ms": [target_median],
            },
            "speedup": speedup,
        },
    }


def test_official_snapshot_and_core_workload_contract() -> None:
    metadata = validate_official_snapshot(PROJECT_ROOT)
    snapshot_path = PROJECT_ROOT / metadata["snapshot_path"]
    assert metadata["sha256"] == EXPECTED_OFFICIAL_SHA256
    assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == (
        EXPECTED_OFFICIAL_SHA256
    )

    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    actual_cases = tuple(
        (
            case.case_id,
            case.batch_size,
            case.seq_len,
            case.d_model,
            case.num_heads,
            case.ffn_dim,
            case.num_layers,
            case.dtype,
            case.causal,
            case.padding_ratio,
        )
        for case in workload["cases"]
    )
    assert actual_cases == EXPECTED_CASES
    assert all(case.input_scale == 1.0 for case in workload["cases"])
    assert (
        tuple(
            (group.group_id, group.weight, group.case_ids)
            for group in workload["groups"]
        )
        == EXPECTED_GROUPS
    )

    raw_document = json.loads(workload["path"].read_text(encoding="utf-8"))
    canonical = json.dumps(
        raw_document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert workload["sha256"] == hashlib.sha256(canonical).hexdigest()


def test_sweep_uses_equal_group_weights() -> None:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    speedups = {
        case.case_id: (8.0 if case.case_id.startswith("mask_s512") else 2.0)
        for case in workload["cases"]
    }
    runs = [
        _successful_run(case.case_id, speedups[case.case_id])
        for case in workload["cases"]
    ]
    summary = summarize_sweep(workload, runs, target="solution")

    assert summary["sweep_outcome"] == "complete"
    assert [group["geomean_speedup"] for group in summary["groups"]] == pytest.approx(
        [2.0, 2.0, 2.0, 8.0, 2.0]
    )
    expected = (2.0**0.8) * (8.0**0.2)
    assert summary["group_balanced_geomean_speedup"] == pytest.approx(expected)
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
    runs = [_successful_run(case.case_id, 2.0) for case in workload["cases"]]
    if replacement is None:
        runs.pop(3)
    else:
        original = runs[3]
        original["outcome"] = replacement["outcome"]
        if "speedup" in replacement:
            original["performance"]["speedup"] = replacement["speedup"]

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
    runs = [_successful_run(case.case_id, 2.0) for case in workload["cases"]]
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
    runs = [_successful_run(case.case_id, 2.0) for case in workload["cases"]]
    runs[3]["correctness"][field] = value

    summary = summarize_sweep(workload, runs, target="solution")
    assert summary["sweep_outcome"] == "incomplete"
    assert summary["groups"] == []
    assert summary["group_balanced_geomean_speedup"] is None
    assert summary["worst_case_speedup"] is None


def test_cli_parses_probe_benchmark_profile_and_tune() -> None:
    parser = runner_cli.build_parser()

    probe = parser.parse_args(["probe"])
    assert probe.command == "probe"
    assert probe.device == "cuda:0"

    benchmark = parser.parse_args(["benchmark"])
    assert benchmark.command == "benchmark"
    assert benchmark.target == "solution"
    assert benchmark.workload_set == WORKLOAD_SET_ID
    assert benchmark.case_id is None
    assert benchmark.preset == "smoke"
    assert benchmark.solution_policy == "auto"

    profile = parser.parse_args(["profile", "--case-id", "attention_s2048_fp16"])
    assert profile.command == "profile"
    assert profile.target == "solution"
    assert profile.workload_set == WORKLOAD_SET_ID
    assert profile.case_id == "attention_s2048_fp16"
    assert profile.solution_policy == "auto"

    tune = parser.parse_args(["tune", "--case-id", "launch_s64_fp16"])
    assert tune.command == "tune"
    assert tune.case_id == ["launch_s64_fp16"]
    assert tune.candidate is None
    assert tune.preset == "smoke"


@pytest.mark.parametrize(
    ("extra_arguments", "expected_count"),
    [
        (["--case-id", "balanced_s128_fp16"], 1),
        ([], len(EXPECTED_CASES)),
    ],
    ids=("single-case", "ordered-sweep"),
)
def test_cli_dispatches_single_case_or_ordered_sweep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra_arguments: list[str],
    expected_count: int,
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
    ) -> tuple[dict[str, Any], Path]:
        del project_root, protocol, device
        calls.append((case.case_id, workload_sha256, sweep_id))
        assert workload_set_id == WORKLOAD_SET_ID
        assert target == "solution"
        return _successful_run(
            case.case_id,
            2.0,
            sweep_id=sweep_id or "single-case",
        ), tmp_path / f"{case.case_id}.json"

    monkeypatch.setattr(runner_cli, "run_managed_benchmark", fake_run_managed_benchmark)
    exit_code = runner_cli.main(["benchmark", *extra_arguments])

    assert exit_code == 0
    assert len(calls) == expected_count
    expected_ids = [case[0] for case in EXPECTED_CASES]
    if expected_count == 1:
        assert [case_id for case_id, _, _ in calls] == ["balanced_s128_fp16"]
        assert calls[0][2] is None
    else:
        assert [case_id for case_id, _, _ in calls] == expected_ids
        sweep_ids = {sweep_id for _, _, sweep_id in calls}
        assert len(sweep_ids) == 1
        assert None not in sweep_ids
    assert all(workload_hash for _, workload_hash, _ in calls)


def test_performance_alternates_order_and_uses_all_raw_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = WorkloadCase(
        case_id="timing_fixture",
        batch_size=1,
        seq_len=2,
        d_model=4,
        num_heads=1,
        ffn_dim=8,
        num_layers=1,
        dtype="float32",
        causal=False,
        padding_ratio=0.0,
        input_scale=1.5,
    )
    protocol = MeasurementProtocol(
        preset="smoke",
        seed=17,
        accuracy_trials=1,
        warmup=2,
        repeats=2,
        rounds=3,
    )
    config = official.TransformerConfig(
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        d_model=case.d_model,
        num_heads=case.num_heads,
        ffn_dim=case.ffn_dim,
        num_layers=case.num_layers,
        causal=case.causal,
    )
    fixed_inputs = torch.zeros(1, 2, 4)
    fixed_mask = torch.ones(1, 2, dtype=torch.bool)
    events: list[tuple[str, str, int]] = []
    generated_arguments: dict[str, Any] = {}

    class MarkerModel(nn.Module):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

    baseline = MarkerModel("baseline")
    solution = MarkerModel("solution")
    batches = {
        "baseline": iter(([1.0, 100.0], [2.0, 3.0], [4.0, 5.0])),
        "solution": iter(([1.0, 9.0], [2.0, 8.0], [3.0, 7.0])),
    }

    def fake_generate_random_case(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        generated_arguments.update(kwargs)
        return fixed_inputs, fixed_mask

    def fake_warmup(
        model: MarkerModel,
        inputs: torch.Tensor,
        valid_mask: torch.Tensor,
        iterations: int,
        device: torch.device,
    ) -> None:
        assert inputs is fixed_inputs
        assert valid_mask is fixed_mask
        assert device.type == "cpu"
        events.append(("warmup", model.name, iterations))

    def fake_benchmark_once(
        model: MarkerModel,
        inputs: torch.Tensor,
        valid_mask: torch.Tensor,
        iterations: int,
        device: torch.device,
    ) -> list[float]:
        assert inputs is fixed_inputs
        assert valid_mask is fixed_mask
        assert device.type == "cpu"
        events.append(("timing", model.name, iterations))
        return list(next(batches[model.name]))

    monkeypatch.setattr(official, "generate_random_case", fake_generate_random_case)
    monkeypatch.setattr(official, "warmup_model", fake_warmup)
    monkeypatch.setattr(official, "benchmark_once", fake_benchmark_once)

    performance = run_performance(
        baseline,
        solution,
        config,
        case,
        protocol,
        torch.device("cpu"),
        torch.float32,
    )

    assert generated_arguments["seed"] == 100_017
    assert generated_arguments["input_scale"] == 1.5
    assert events == [
        ("warmup", "baseline", 2),
        ("warmup", "solution", 2),
        ("timing", "baseline", 2),
        ("timing", "solution", 2),
        ("timing", "solution", 2),
        ("timing", "baseline", 2),
        ("timing", "baseline", 2),
        ("timing", "solution", 2),
    ]
    assert performance["baseline"]["median_ms"] == 3.5
    assert performance["baseline"]["p90_ms"] == pytest.approx(52.5)
    assert performance["baseline"]["round_medians_ms"] == [50.5, 2.5, 4.5]
    assert performance["baseline"]["sample_count"] == 6
    assert performance["target"]["median_ms"] == 5.0
    assert performance["target"]["p90_ms"] == pytest.approx(8.5)
    assert performance["target"]["round_medians_ms"] == [5.0, 5.0, 5.0]
    assert performance["target"]["sample_count"] == 6
    assert performance["speedup"] == pytest.approx(0.7)
    assert "samples_ms" not in json.dumps(performance)
    assert "solution" not in performance


def test_managed_cpu_solution_smoke_persists_result(tmp_path: Path) -> None:
    project = _copy_runtime_project(tmp_path)
    protocol = _tiny_protocol()

    result, result_path = run_managed_benchmark(
        project,
        workload_set_id="tiny_test_fixture",
        case=_tiny_case(),
        protocol=protocol,
        device="cpu",
        target="solution",
        workload_sha256="fixture-hash",
    )

    assert result["outcome"] == "success"
    assert result["schema_version"] == 2
    assert "sweep_id" not in result
    assert result["correctness"]["passed"] is True
    assert "failure" not in result
    performance = result["performance"]
    assert performance["sample_count"] == 1
    assert performance["baseline"]["median_ms"] > 0
    assert performance["baseline"]["p90_ms"] > 0
    assert len(performance["baseline"]["round_medians_ms"]) == 1
    assert performance["target"]["median_ms"] > 0
    assert performance["target"]["p90_ms"] > 0
    assert len(performance["target"]["round_medians_ms"]) == 1
    assert math.isfinite(performance["speedup"])
    assert performance["speedup"] > 0

    assert result["workload"]["sha256"] == "fixture-hash"
    assert result["workload"]["case"]["case_id"] == "tiny_cpu_smoke"
    assert "status" not in result
    assert "path" not in result
    assert "case" not in result
    assert "workload_set_id" not in result
    assert "trials" not in result["correctness"]
    assert "samples_ms" not in json.dumps(result)
    assert result_path.parent == project / "results" / "runs"
    assert json.loads(result_path.read_text(encoding="utf-8")) == result


def test_managed_cpu_profile_persists_compact_hotspots(tmp_path: Path) -> None:
    project = _copy_runtime_project(tmp_path)

    result, result_path = supervisor.run_managed_profile(
        project,
        workload_set_id="tiny_test_fixture",
        case=_tiny_case(),
        protocol=_tiny_protocol(),
        device="cpu",
        target="solution",
        workload_sha256="fixture-hash",
    )

    assert result["outcome"] == "success"
    assert result["correctness"]["passed"] is True
    profile = result["profile"]
    assert profile["iterations"] == 1
    assert profile["operator_hotspots"]
    for hotspot in profile["operator_hotspots"]:
        assert hotspot["name"].startswith("aten::")
        assert hotspot["calls_per_forward"] > 0
        assert hotspot["self_time_us_per_forward"] > 0
        assert 0 < hotspot["share_pct"] <= 100
    assert "trace" not in json.dumps(result).lower()
    assert json.loads(result_path.read_text(encoding="utf-8")) == result


def test_managed_failure_persists_only_known_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_worker(
        project_root: Path,
        request: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        del project_root, request, timeout_seconds
        return {
            "outcome": "timeout",
            "environment": None,
            "probe": None,
            "failure": {
                "stage": "worker",
                "type": "TimeoutExpired",
                "message": "worker exceeded its time limit",
                "exit_code": None,
            },
        }

    monkeypatch.setattr(supervisor, "_run_worker", fake_run_worker)
    result, result_path = supervisor.run_managed_probe(
        tmp_path,
        device="cuda:0",
        timeout_seconds=1.0,
    )

    assert set(result) == {
        "schema_version",
        "run_id",
        "created_at",
        "run_kind",
        "requested_device",
        "outcome",
        "failure",
    }
    assert result["outcome"] == "timeout"
    assert result["failure"] == {
        "stage": "worker",
        "type": "TimeoutExpired",
        "message": "worker exceeded its time limit",
    }
    assert json.loads(result_path.read_text(encoding="utf-8")) == result


def test_correctness_failure_persists_summary_not_trials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = _copy_runtime_project(tmp_path)

    def fake_run_worker(
        project_root: Path,
        request: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        del project_root, request, timeout_seconds
        return {
            "outcome": "invalid_output",
            "solution_source_sha256": "fixture-solution-hash",
            "environment": {
                "device": "cpu",
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
            },
            "correctness": {
                "passed": False,
                "trial_count": 1,
                "failed_elements": 3,
                "max_abs_error": 0.25,
                "max_relative_error": 2.0,
                "trials": [
                    {
                        "seed": 17,
                        "passed": False,
                        "failed_elements": 3,
                        "max_abs_error": 0.25,
                        "max_relative_error": 2.0,
                        "error": "ContractError: output shape mismatch",
                    }
                ],
            },
            "execution_path": {"qkv_projection": "packed"},
            "failure": {
                "stage": "correctness",
                "type": "CorrectnessError",
                "message": "Solution failed the correctness contract",
                "exit_code": None,
            },
        }

    monkeypatch.setattr(supervisor, "_run_worker", fake_run_worker)
    result, result_path = supervisor.run_managed_benchmark(
        project,
        workload_set_id="tiny_test_fixture",
        case=_tiny_case(),
        protocol=_tiny_protocol(),
        device="cpu",
        target="solution",
        workload_sha256="fixture-hash",
    )

    assert result["outcome"] == "invalid_output"
    assert result["correctness"] == {
        "passed": False,
        "trial_count": 1,
        "failed_elements": 3,
        "max_abs_error": 0.25,
        "max_relative_error": 2.0,
        "diagnostic": "ContractError: output shape mismatch",
    }
    assert "performance" not in result
    assert "trials" not in result["correctness"]
    assert "samples_ms" not in json.dumps(result)
    assert json.loads(result_path.read_text(encoding="utf-8")) == result


def test_probe_rejects_failed_device_operation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_worker(
        project_root: Path,
        request: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        del project_root, request, timeout_seconds
        return {
            "outcome": "success",
            "environment": {"device": "cpu", "torch": torch.__version__},
            "probe": {
                "device_operation_passed": False,
                "sdpa": {"available": False, "reason": "cuda_required"},
            },
            "failure": None,
        }

    monkeypatch.setattr(supervisor, "_run_worker", fake_run_worker)
    result, _ = supervisor.run_managed_probe(tmp_path, device="cpu")

    assert result["outcome"] == "runtime_error"
    assert result["probe"]["device_operation_passed"] is False
    assert result["failure"] == {
        "stage": "result_compaction",
        "type": "InvalidWorkerResponse",
        "message": "device operation failed",
    }


@pytest.mark.parametrize("timeout", [0.0, float("nan"), float("inf")])
def test_probe_rejects_invalid_timeout_before_start(
    tmp_path: Path,
    timeout: float,
) -> None:
    with pytest.raises(ContractError, match="timeout_seconds must be finite"):
        supervisor.run_managed_probe(
            tmp_path,
            device="cpu",
            timeout_seconds=timeout,
        )
    assert not (tmp_path / "results").exists()
