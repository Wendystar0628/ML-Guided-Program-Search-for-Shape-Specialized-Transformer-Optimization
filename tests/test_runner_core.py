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
    solution_implementation_hash,
    solution_source_hash,
    validate_official_snapshot,
)
from runner.execution import (
    _validate_cuda_graph_composition,
    _validate_profile_execution_path,
    run_performance,
)
from runner.supervisor import run_managed_benchmark
from runner.sweep import summarize_sweep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_OFFICIAL_SHA256 = (
    "1630fe39ebc845beeaef73aaaf2d47e061fc56fd20777706c3ddc961664c266b"
)
WORKLOAD_SET_ID = "transformer_core_v1"
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


def test_solution_hash_excludes_external_route_tables(tmp_path: Path) -> None:
    solution_root = tmp_path / "solution"
    solution_root.mkdir()
    (solution_root / "transformer.py").write_text("VALUE = 1\n", encoding="utf-8")
    route_path = solution_root / "dispatch_routes.json"
    route_path.write_text(
        '{"schema_version":1,"default_policy":"auto","routes":[]}\n',
        encoding="utf-8",
    )
    original = solution_source_hash(solution_root)
    implementation = solution_implementation_hash(solution_root)

    route_path.write_text(
        '{"schema_version":1,"default_policy":"reference","routes":[]}\n',
        encoding="utf-8",
    )

    assert solution_source_hash(solution_root) == original
    assert solution_implementation_hash(solution_root) == implementation


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
    assert profile.target == "solution"
    assert profile.workload_set == WORKLOAD_SET_ID
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
    assert tune.preset == "smoke"

    default_calibrate = parser.parse_args(["calibrate"])
    assert default_calibrate.candidate_limit == 3

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


def _routing_probe_result() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "run_id": "fixture-routing-probe",
        "created_at": "2026-08-27T00:00:00+00:00",
        "requested_device": "cuda:0",
        "outcome": "success",
        "environment": {
            "device": "cuda:0",
            "gpu": "fixture-gpu",
            "compute_capability": "8.9",
            "total_memory_bytes": 16_000_000_000,
            "driver": "fixture-driver",
            "platform": "Windows-fixture",
            "torch": "fixture-torch",
            "cuda_runtime": "13.2",
        },
        "probe": {
            "mode": "routing",
            "device_operation_passed": True,
            "runtime_policy": {
                "matmul_precision": "high",
                "allow_tf32": True,
            },
            "hardware_profile": {
                "available": True,
                "device_type": "cuda",
                "platform": {"system": "Windows", "machine": "AMD64"},
                "software": {
                    "driver": "fixture-driver",
                    "torch": "fixture-torch",
                    "cuda_runtime": "13.2",
                    "triton": "fixture-triton",
                    "triton_available": True,
                },
                "gpu": {
                    "available": True,
                    "name": "fixture-gpu",
                    "compute_capability": "8.9",
                    "architecture_family": "ada",
                    "bf16_supported": True,
                    "cuda_graph_available": True,
                    "total_memory_bytes": 16_000_000_000,
                    "sm_count": 76,
                    "l2_cache_bytes": 64_000_000,
                    "shared_memory_per_sm_bytes": 102_400,
                    "registers_per_sm": 65_536,
                    "memory_bus_width_bits": 256,
                    "memory_clock_rate_khz": 1_000,
                    "theoretical_memory_bandwidth_gbps": 716.8,
                },
            },
            "performance_anchors": {
                "eager_launch": {"effective_latency_us": 4.0},
                "cuda_graph_replay": {
                    "replay_latency_us": 8.0,
                    "effective_latency_per_node_us": 2.0,
                },
                "device_copy": {"effective_bandwidth_gbps": 600.0},
                "gemm_float16": {"tflops": 80.0},
                "gemm_bfloat16": {"tflops": 75.0},
                "gemm_float32": {"tflops": 40.0},
                "softmax_fp32": {"throughput_gigaelements_per_second": 250.0},
            },
            "sdpa": {"available": True, "scenarios": []},
        },
    }


def test_routing_probe_is_flattened_to_compact_model_inputs() -> None:
    profile = runner_cli._hardware_profile_from_probe(_routing_probe_result())

    assert profile["device_type"] == "cuda"
    assert profile["device_name"] == "fixture-gpu"
    assert profile["compute_capability"] == "8.9"
    assert profile["architecture_family"] == "ada"
    assert profile["triton"] == "fixture-triton"
    assert profile["triton_available"] is True
    assert profile["bf16_supported"] is True
    assert profile["cuda_graph_available"] is True
    assert profile["platform_system"] == "Windows"
    assert profile["sm_count"] == 76
    assert profile["memory_clock_khz"] == 1_000
    assert profile["performance_anchors"] == {
        "launch_latency_us": 4.0,
        "graph_replay_per_node_us": 2.0,
        "memory_bandwidth_gbps": 600.0,
        "gemm_tflops": {
            "float16": 80.0,
            "bfloat16": 75.0,
            "float32": 40.0,
        },
        "softmax_giga_elements_per_s": 250.0,
    }
    assert "gpu" not in profile
    assert "sdpa" not in profile


def _successful_tuning_summary(
    case_id: str,
    candidate_order: list[str],
    tmp_path: Path,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "tuning_id": f"tuning-{case_id}",
        "summary_path": str(tmp_path / f"tuning-{case_id}.json"),
        "observations": [],
        "winner": {"candidate_id": candidate_order[0]},
    }


def _staged_tuning_summary(
    case: WorkloadCase,
    candidate_order: list[str],
    preset: str,
    tmp_path: Path,
    *,
    speedups: dict[str, float] | None = None,
) -> dict[str, Any]:
    candidates = {
        candidate.candidate_id: candidate
        for candidate in runner_cli.candidates_for_case(case)
    }
    resolved_speedups = speedups or {
        candidate_id: 1.0 + (index * 0.1)
        for index, candidate_id in enumerate(candidate_order)
    }
    observations: list[dict[str, Any]] = []
    for candidate_id in candidate_order:
        candidate = candidates[candidate_id]
        speedup = resolved_speedups[candidate_id]
        target = 2.0 / speedup
        observations.append(
            {
                "candidate_id": candidate_id,
                "solution_policy": candidate.solution_policy,
                "compile_solution": candidate.compile_solution,
                "cuda_graph_solution": candidate.cuda_graph_solution,
                "outcome": "success",
                "correctness_passed": True,
                "failed_elements": 0,
                "policy_applied": True,
                "speedup": speedup,
                "conservative_speedup": speedup,
                "baseline_round_medians_ms": [2.0, 2.0],
                "target_round_medians_ms": [target, target],
                "target_median_ms": target,
                "target_p90_ms": target * 1.01,
                "execution_path": {"shape_route": candidate.solution_policy},
            }
        )
    winner = max(observations, key=lambda item: item["conservative_speedup"])
    deployable = [
        item
        for item in observations
        if item["compile_solution"] is False
        and item["cuda_graph_solution"] is False
    ]
    return {
        "case_id": case.case_id,
        "tuning_id": f"{preset}-{case.case_id}",
        "summary_path": str(tmp_path / f"{preset}-{case.case_id}.json"),
        "complete": True,
        "protocol": {"preset": preset},
        "source_consistent": True,
        "implementation_consistent": True,
        "source_solution_sha256": solution_source_hash(PROJECT_ROOT / "solution"),
        "source_implementation_sha256": solution_implementation_hash(
            PROJECT_ROOT / "solution"
        ),
        "observations": observations,
        "winner": winner,
        "deployable_winner": (
            max(deployable, key=lambda item: item["conservative_speedup"])
            if deployable
            else None
        ),
    }


def test_tune_requires_explicit_candidates_and_has_no_candidate_limit() -> None:
    parser = runner_cli.build_parser()

    with pytest.raises(SystemExit) as missing_candidate:
        parser.parse_args(["tune", "--case-id", "balanced_s128_fp16"])
    assert missing_candidate.value.code == 2

    with pytest.raises(SystemExit) as removed_candidate_limit:
        parser.parse_args(
            [
                "tune",
                "--case-id",
                "balanced_s128_fp16",
                "--candidate",
                "eager-auto",
                "--candidate-limit",
                "2",
            ]
        )
    assert removed_candidate_limit.value.code == 2


def test_explicit_tune_preserves_order_without_a_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tuning_calls: list[dict[str, Any]] = []

    def unexpected_probe(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], Path]:
        del args, kwargs
        raise AssertionError("explicit tuning must not run a hardware probe")

    def unexpected_plan(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise AssertionError("explicit tuning must not build a routing plan")

    def fake_tuning(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        tuning_calls.append(kwargs)
        return _successful_tuning_summary(
            kwargs["case"].case_id,
            kwargs["requested_candidates"],
            tmp_path,
        )

    monkeypatch.setattr(runner_cli, "run_managed_probe", unexpected_probe)
    monkeypatch.setattr(runner_cli, "build_routing_plan", unexpected_plan)
    monkeypatch.setattr(runner_cli, "run_tuning_case", fake_tuning)
    monkeypatch.setattr(runner_cli, "_print_tuning_summary", lambda *args: None)

    exit_code = runner_cli.main(
        [
            "tune",
            "--case-id",
            "balanced_s128_fp16",
            "--candidate",
            "eager-torch",
            "--candidate",
            "eager-auto",
            "--matmul-precision",
            "highest",
            "--no-allow-tf32",
            "--timeout",
            "42",
        ]
    )

    assert exit_code == 0
    assert tuning_calls[0]["requested_candidates"] == [
        "eager-torch",
        "eager-auto",
    ]
    assert tuning_calls[0]["device"] == "cuda:0"
    assert tuning_calls[0]["base_protocol"].matmul_precision == "highest"
    assert tuning_calls[0]["base_protocol"].allow_tf32 is False
    assert tuning_calls[0]["base_protocol"].timeout_seconds == 42.0
    assert tuning_calls[0]["routing_plan"] == {
        "source": "explicit_candidates",
        "decision_scope": "candidate_order_only",
        "requires_full_workload_measurement": True,
        "candidate_order": ["eager-torch", "eager-auto"],
    }
    assert tuning_calls[0]["device_profile"] is None


def test_calibrate_plan_only_covers_the_full_workload_without_tuning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    def fake_probe(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], Path]:
        del args
        assert kwargs["probe_mode"] == "routing"
        events.append("probe")
        return _routing_probe_result(), tmp_path / "probe.json"

    def fake_plan(
        case: WorkloadCase,
        hardware_profile: dict[str, Any],
        candidate_ids: tuple[str, ...],
        *,
        limit: int,
    ) -> dict[str, Any]:
        del hardware_profile, candidate_ids
        assert limit == 1
        events.append(f"plan:{case.case_id}")
        return {
            "source": "hardware_cost_model",
            "bottleneck_class": "balanced",
            "workload_analysis": {"case_id": case.case_id},
            "candidate_order": ["eager-auto"],
            "selection_reasons": {},
            "capability_rejections": {},
        }

    monkeypatch.setattr(runner_cli, "run_managed_probe", fake_probe)
    monkeypatch.setattr(runner_cli, "build_routing_plan", fake_plan)
    monkeypatch.setattr(
        runner_cli,
        "run_tuning_case",
        lambda *args, **kwargs: pytest.fail("plan-only must not run tuning"),
    )
    monkeypatch.setattr(runner_cli, "_print_run_summary", lambda *args: None)
    monkeypatch.setattr(runner_cli, "_print_routing_plan", lambda *args: None)

    exit_code = runner_cli.main(["calibrate", "--plan-only", "--candidate-limit", "1"])

    assert exit_code == 0
    assert events == ["probe", *(f"plan:{case[0]}" for case in EXPECTED_CASES)]


def test_formal_plan_only_allows_one_shared_case_on_a_new_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runner_cli,
        "run_managed_probe",
        lambda *args, **kwargs: (_routing_probe_result(), tmp_path / "probe.json"),
    )
    monkeypatch.setattr(
        runner_cli,
        "find_matching_verified_route",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        runner_cli,
        "build_routing_plan",
        lambda *args, **kwargs: {
            "source": "hardware_cost_model",
            "candidate_order": ["eager-auto"],
        },
    )
    monkeypatch.setattr(
        runner_cli,
        "run_tuning_case",
        lambda *args, **kwargs: pytest.fail("plan-only must not run tuning"),
    )
    monkeypatch.setattr(runner_cli, "_print_run_summary", lambda *args: None)
    monkeypatch.setattr(runner_cli, "_print_routing_plan", lambda *args: None)

    exit_code = runner_cli.main(
        [
            "calibrate",
            "--preset",
            "formal",
            "--plan-only",
            "--case-id",
            "mask_s512_padding_fp16",
            "--candidate-limit",
            "1",
        ]
    )

    assert exit_code == 0


def test_cpu_plan_only_does_not_require_a_verified_gpu_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runner_cli,
        "run_managed_probe",
        lambda *args, **kwargs: (_routing_probe_result(), tmp_path / "probe.json"),
    )
    monkeypatch.setattr(
        runner_cli,
        "_hardware_profile_from_probe",
        lambda result: {"device_type": "cpu", "platform_system": "Windows"},
    )
    monkeypatch.setattr(
        runner_cli,
        "verified_profile_from_probe_result",
        lambda result: pytest.fail("CPU planning must not build a GPU package"),
    )
    monkeypatch.setattr(
        runner_cli,
        "build_routing_plan",
        lambda *args, **kwargs: {
            "source": "hardware_cost_model",
            "candidate_order": ["eager-auto"],
        },
    )
    monkeypatch.setattr(runner_cli, "_print_run_summary", lambda *args: None)
    monkeypatch.setattr(runner_cli, "_print_routing_plan", lambda *args: None)

    exit_code = runner_cli.main(
        [
            "calibrate",
            "--device",
            "cpu",
            "--plan-only",
            "--candidate-limit",
            "1",
        ]
    )

    assert exit_code == 0


@pytest.mark.parametrize(
    "arguments",
    [
        ["--matmul-precision", "highest"],
        ["--matmul-precision", "medium"],
        ["--no-allow-tf32"],
    ],
)
def test_formal_calibration_rejects_non_deployable_precision_before_probe(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    monkeypatch.setattr(
        runner_cli,
        "run_managed_probe",
        lambda *args, **kwargs: pytest.fail(
            "invalid Formal deployment settings must fail before probing"
        ),
    )

    exit_code = runner_cli.main(["calibrate", "--preset", "formal", *arguments])

    assert exit_code == 2


def test_calibrate_probes_once_then_plans_then_runs_full_workloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    profiles: list[dict[str, Any]] = []

    def fake_probe(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], Path]:
        del args
        assert kwargs["probe_mode"] == "routing"
        events.append("probe")
        return _routing_probe_result(), tmp_path / "probe.json"

    def fake_plan(
        case: WorkloadCase,
        hardware_profile: dict[str, Any],
        candidate_ids: tuple[str, ...],
        *,
        limit: int,
    ) -> dict[str, Any]:
        del candidate_ids
        assert limit == 1
        events.append(f"plan:{case.case_id}")
        profiles.append(hardware_profile)
        return {
            "source": "hardware_cost_model",
            "candidate_order": ["eager-auto"],
        }

    def fake_tuning(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        case_id = kwargs["case"].case_id
        events.append(f"tune:{case_id}")
        assert kwargs["requested_candidates"] == ["eager-auto"]
        assert kwargs["device_profile"] is profiles[0]
        return {
            **_successful_tuning_summary(case_id, ["eager-auto"], tmp_path),
            "complete": True,
        }

    monkeypatch.setattr(runner_cli, "run_managed_probe", fake_probe)
    monkeypatch.setattr(runner_cli, "build_routing_plan", fake_plan)
    monkeypatch.setattr(runner_cli, "run_tuning_case", fake_tuning)
    monkeypatch.setattr(runner_cli, "_print_run_summary", lambda *args: None)
    monkeypatch.setattr(runner_cli, "_print_tuning_summary", lambda *args: None)
    monkeypatch.setattr(runner_cli, "_print_routing_plan", lambda *args: None)
    monkeypatch.setattr(
        runner_cli,
        "auto_promote_calibration",
        lambda *args, **kwargs: pytest.fail("smoke calibration must not publish routes"),
    )

    exit_code = runner_cli.main(
        [
            "calibrate",
            "--case-id",
            "launch_s64_fp16",
            "--case-id",
            "wide_s256_bf16",
            "--candidate-limit",
            "1",
        ]
    )

    assert exit_code == 0
    assert events == [
        "probe",
        "plan:launch_s64_fp16",
        "plan:wide_s256_bf16",
        "tune:launch_s64_fp16",
        "tune:wide_s256_bf16",
    ]
    assert len(profiles) == 2
    assert profiles[0] is profiles[1]


def test_formal_calibration_runs_all_smoke_before_formal_and_promotes_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    route_path = tmp_path / "verified_hardware" / "fixture" / "routes.json"
    routing_probe_result = _routing_probe_result()

    def fake_probe(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], Path]:
        del args
        assert kwargs["probe_mode"] == "routing"
        events.append("probe")
        return routing_probe_result, tmp_path / "probe.json"

    def fake_plan(
        case: WorkloadCase,
        hardware_profile: dict[str, Any],
        candidate_ids: tuple[str, ...],
        *,
        limit: int,
    ) -> dict[str, Any]:
        del hardware_profile, candidate_ids
        assert limit == 1
        events.append(f"plan:{case.case_id}")
        return {
            "source": "hardware_cost_model",
            "candidate_order": ["eager-auto"],
        }

    def fake_tuning(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        case = kwargs["case"]
        preset = kwargs["base_protocol"].preset
        events.append(f"{preset}:{case.case_id}")
        assert kwargs["requested_candidates"] == ["eager-auto"]
        return _staged_tuning_summary(
            case,
            kwargs["requested_candidates"],
            preset,
            tmp_path,
        )

    def fake_auto_promote(
        project_root: Path,
        summaries: list[dict[str, Any]],
        *,
        probe_result: dict[str, Any],
        full_workload_case_ids: list[str],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], Path, bool]:
        del project_root
        events.append("auto-promote")
        assert probe_result is routing_probe_result
        assert full_workload_case_ids == [case[0] for case in EXPECTED_CASES]
        assert [summary["case_id"] for summary in summaries] == [
            "launch_s64_fp16",
            "wide_s256_bf16",
        ]
        assert all(summary["protocol"]["preset"] == "formal" for summary in summaries)
        winners = [dict(summary["deployable_winner"]) for summary in summaries]
        return {"schema_version": 2}, winners, route_path, False

    monkeypatch.setattr(runner_cli, "run_managed_probe", fake_probe)
    monkeypatch.setattr(runner_cli, "build_routing_plan", fake_plan)
    monkeypatch.setattr(runner_cli, "run_tuning_case", fake_tuning)
    monkeypatch.setattr(
        runner_cli,
        "find_matching_verified_route",
        lambda *args, **kwargs: route_path,
    )
    monkeypatch.setattr(runner_cli, "auto_promote_calibration", fake_auto_promote)
    monkeypatch.setattr(runner_cli, "_print_run_summary", lambda *args: None)
    monkeypatch.setattr(runner_cli, "_print_tuning_summary", lambda *args: None)
    monkeypatch.setattr(runner_cli, "_print_routing_plan", lambda *args: None)

    exit_code = runner_cli.main(
        [
            "calibrate",
            "--preset",
            "formal",
            "--case-id",
            "launch_s64_fp16",
            "--case-id",
            "wide_s256_bf16",
            "--candidate-limit",
            "1",
        ]
    )

    assert exit_code == 0
    assert events == [
        "probe",
        "plan:launch_s64_fp16",
        "plan:wide_s256_bf16",
        "smoke:launch_s64_fp16",
        "smoke:wide_s256_bf16",
        "formal:launch_s64_fp16",
        "formal:wide_s256_bf16",
        "auto-promote",
    ]


def test_formal_calibration_does_not_publish_without_deployable_winner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    route_path = tmp_path / "verified_hardware" / "fixture" / "routes.json"
    probe_result = _routing_probe_result()

    monkeypatch.setattr(
        runner_cli,
        "run_managed_probe",
        lambda *args, **kwargs: (probe_result, tmp_path / "probe.json"),
    )
    monkeypatch.setattr(
        runner_cli,
        "build_routing_plan",
        lambda *args, **kwargs: {
            "source": "hardware_cost_model",
            "candidate_order": ["eager-auto"],
        },
    )
    def fake_tuning(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        summary = _staged_tuning_summary(
            kwargs["case"],
            ["eager-auto"],
            kwargs["base_protocol"].preset,
            tmp_path,
        )
        if kwargs["base_protocol"].preset == "formal":
            summary["winner"] = None
            summary["deployable_winner"] = None
        return summary

    monkeypatch.setattr(runner_cli, "run_tuning_case", fake_tuning)
    monkeypatch.setattr(
        runner_cli,
        "find_matching_verified_route",
        lambda *args, **kwargs: route_path,
    )
    monkeypatch.setattr(
        runner_cli,
        "auto_promote_calibration",
        lambda *args, **kwargs: pytest.fail(
            "an incomplete Formal result must not publish routes"
        ),
    )
    monkeypatch.setattr(runner_cli, "_print_run_summary", lambda *args: None)
    monkeypatch.setattr(runner_cli, "_print_tuning_summary", lambda *args: None)
    monkeypatch.setattr(runner_cli, "_print_routing_plan", lambda *args: None)

    exit_code = runner_cli.main(
        [
            "calibrate",
            "--preset",
            "formal",
            "--case-id",
            "launch_s64_fp16",
            "--candidate-limit",
            "1",
        ]
    )

    assert exit_code == 1


def test_calibrate_stops_before_planning_when_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    def fake_probe(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], Path]:
        del args, kwargs
        events.append("probe")
        return {"outcome": "runtime_error"}, tmp_path / "probe.json"

    monkeypatch.setattr(runner_cli, "run_managed_probe", fake_probe)
    monkeypatch.setattr(
        runner_cli,
        "build_routing_plan",
        lambda *args, **kwargs: pytest.fail("failed probe must not build a plan"),
    )
    monkeypatch.setattr(
        runner_cli,
        "run_tuning_case",
        lambda *args, **kwargs: pytest.fail("failed probe must not run workloads"),
    )
    monkeypatch.setattr(runner_cli, "_print_run_summary", lambda *args: None)

    exit_code = runner_cli.main(
        ["calibrate", "--case-id", "launch_s64_fp16"]
    )

    assert exit_code == 1
    assert events == ["probe"]


def test_compile_and_cuda_graph_candidates_are_mutually_exclusive() -> None:
    protocol = MeasurementProtocol(
        preset="smoke",
        compile_solution=True,
        cuda_graph_solution=True,
    )

    with pytest.raises(ContractError, match="cannot combine"):
        protocol.validate()


@pytest.mark.parametrize(
    "protocol",
    [
        MeasurementProtocol(preset="smoke", compile_solution=True),
        MeasurementProtocol(preset="smoke", cuda_graph_solution=True),
    ],
)
def test_solution_graph_rejects_compile_or_outer_graph(
    protocol: MeasurementProtocol,
) -> None:
    with pytest.raises(ContractError, match="Solution CUDA Graph"):
        _validate_cuda_graph_composition(
            {"runtime_wrapper": "solution_eager_cuda_graph"},
            protocol,
        )


def test_operator_profile_rejects_the_solution_graph_wrapper() -> None:
    with pytest.raises(ContractError, match="hides per-operator"):
        _validate_profile_execution_path(
            {"runtime_wrapper": "solution_eager_cuda_graph"}
        )


@pytest.mark.parametrize(
    ("extra_arguments", "expected_count", "expected_result_dir"),
    [
        (
            [
                "--case-id",
                "balanced_s128_fp16",
                "--result-dir",
                "verified_hardware/fixture/results/runs",
            ],
            1,
            Path("verified_hardware/fixture/results/runs"),
        ),
        ([], len(EXPECTED_CASES), None),
    ],
    ids=("single-case", "ordered-sweep"),
)
def test_cli_dispatches_single_case_or_ordered_sweep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra_arguments: list[str],
    expected_count: int,
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
    ) -> tuple[dict[str, Any], Path]:
        del project_root, protocol, device
        assert result_dir == expected_result_dir
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


def test_probe_rejects_invalid_mode_before_start(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="unsupported probe mode"):
        supervisor.run_managed_probe(
            tmp_path,
            device="cpu",
            probe_mode="unknown",
        )
    assert not (tmp_path / "results").exists()
