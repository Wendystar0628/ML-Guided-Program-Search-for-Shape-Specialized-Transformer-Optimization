"""Shared fixtures for focused runner tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from runner.contracts import (
    MeasurementProtocol,
    WorkloadCase,
    load_workload_set,
    solution_implementation_hash,
    solution_source_hash,
)
from runner.tuning import candidates_for_case

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def tiny_case(*, causal: bool = True, padding_ratio: float = 0.5) -> WorkloadCase:
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


def tiny_protocol() -> MeasurementProtocol:
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


def successful_run(
    case_id: str,
    speedup: float,
    *,
    sweep_id: str = "fixture-sweep",
) -> dict[str, Any]:
    workload_set = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    case = next(case for case in workload_set.cases if case.case_id == case_id)
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
            "sha256": workload_set.sha256,
            "case": case.as_dict(),
        },
        "protocol": {"accuracy_trials": 1, "repeats": 1, "rounds": 1},
        "environment": {"device": "cuda:0", "gpu": "fixture-gpu"},
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


def routing_probe_result() -> dict[str, Any]:
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
            "runtime_policy": {"matmul_precision": "high", "allow_tf32": True},
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


def staged_tuning_summary(
    case: WorkloadCase,
    candidate_order: list[str],
    preset: str,
    tmp_path: Path,
) -> dict[str, Any]:
    candidates = {
        candidate.candidate_id: candidate for candidate in candidates_for_case(case)
    }
    observations: list[dict[str, Any]] = []
    for index, candidate_id in enumerate(candidate_order):
        candidate = candidates[candidate_id]
        speedup = 1.0 + index * 0.1
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
                "execution_path": {
                    "requested_policy": candidate.solution_policy,
                    "selected_policy": candidate.solution_policy,
                },
            }
        )
    deployable = [
        item
        for item in observations
        if item["compile_solution"] is False and item["cuda_graph_solution"] is False
    ]
    implementation_hash = solution_implementation_hash(PROJECT_ROOT / "solution")
    return {
        "case_id": case.case_id,
        "tuning_id": f"{preset}-{case.case_id}",
        "summary_path": str(tmp_path / f"{preset}-{case.case_id}.json"),
        "complete": True,
        "protocol": {"preset": preset},
        "source_consistent": True,
        "implementation_consistent": True,
        "source_solution_sha256": solution_source_hash(PROJECT_ROOT / "solution"),
        "source_implementation_sha256": implementation_hash,
        "observations": observations,
        "winner": max(observations, key=lambda item: item["conservative_speedup"]),
        "deployable_winner": max(
            deployable,
            key=lambda item: item["conservative_speedup"],
            default=None,
        ),
    }


def canonical_workload_hash() -> str:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    raw_document = json.loads(workload.path.read_text(encoding="utf-8"))
    canonical = json.dumps(
        raw_document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
