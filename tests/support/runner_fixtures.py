"""Small shared builders for runner tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from project_identity import official_snapshot_hash
from runner.contracts import (
    OFFICIAL_WORKLOAD_SET_ID,
    MeasurementProtocol,
    RunVariant,
    TransformerShape,
    load_workload_set,
    select_transformer_shape,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKLOAD_SET_ID = OFFICIAL_WORKLOAD_SET_ID


def official_shape(case_id: str = "official_02") -> TransformerShape:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    return select_transformer_shape(workload, case_id)


def tiny_shape(*, causal: bool = True) -> TransformerShape:
    return TransformerShape(
        case_id="tiny_cpu_smoke",
        batch_size=1,
        seq_len=2,
        d_model=4,
        num_heads=1,
        ffn_dim=4,
        num_layers=1,
        causal=causal,
    )


def official_variant(**changes: Any) -> RunVariant:
    values: dict[str, Any] = {
        "dtype": "float32",
        "padding_ratio": 0.0,
        "input_scale": 1.0,
    }
    values.update(changes)
    return RunVariant(**values)


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
    policy: str = "causal-sdpa",
) -> dict[str, Any]:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    shape = select_transformer_shape(workload, case_id)
    protocol = tiny_protocol()
    variant = official_variant()
    target_median = 2.0 / speedup
    return {
        "schema_version": 3,
        "run_id": f"fixture-{case_id}",
        "run_kind": "benchmark",
        "target": "solution",
        "sweep_id": sweep_id,
        "created_at": "2026-08-28T00:00:00+00:00",
        "outcome": "success",
        "source": {
            "official_sha256": official_snapshot_hash(PROJECT_ROOT),
            "solution_sha256": "fixture-solution-hash",
        },
        "workload": {
            "set_id": WORKLOAD_SET_ID,
            "sha256": workload.sha256,
            "shape": shape.as_dict(),
            "variant": variant.as_dict(),
        },
        "protocol": protocol.as_dict(),
        "environment": {"device": "cuda:0", "gpu": "fixture-gpu"},
        "correctness": {
            "passed": True,
            "trial_count": protocol.accuracy_trials,
            "failed_elements": 0,
            "max_abs_error": 0.0,
            "max_relative_error": 0.0,
        },
        "performance": {
            "timer": "cuda_event",
            "sample_count": protocol.repeats * protocol.rounds,
            "baseline": {"median_ms": 2.0, "p90_ms": 2.0},
            "target": {"median_ms": target_median, "p90_ms": target_median},
            "speedup": speedup,
        },
        "execution_path": {
            "requested_policy": policy,
            "selected_policy": policy,
            "attention_backend": "causal_sdpa",
            "runtime_wrapper": "eager",
            "batch_strategy": "full",
            "block_backend": "torch",
        },
    }


def routing_probe_result() -> dict[str, Any]:
    return {
        "schema_version": 3,
        "run_id": "fixture-routing-probe",
        "created_at": "2026-08-28T00:00:00+00:00",
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
                "softmax_fp32": {
                    "throughput_gigaelements_per_second": 250.0
                },
            },
            "sdpa": {"available": True, "scenarios": []},
        },
    }
