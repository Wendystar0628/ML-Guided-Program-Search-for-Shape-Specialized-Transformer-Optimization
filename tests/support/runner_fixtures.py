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
from runner.performance_metrics import LOGICAL_OPERATOR_TRAFFIC_SCOPE

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
    policy: str = "eager-sdpa",
) -> dict[str, Any]:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    shape = select_transformer_shape(workload, case_id)
    protocol = tiny_protocol()
    variant = official_variant()
    target_median = 2.0 / speedup
    useful_flops = 1_000_000_000
    logical_operator_bytes = 100_000_000
    return {
        "schema_version": 6,
        "run_id": f"fixture-{case_id}",
        "run_kind": "benchmark",
        "target": "solution",
        "comparison_mode": "paired",
        "sweep_id": sweep_id,
        "created_at": "2026-08-28T00:00:00+00:00",
        "outcome": "success",
        "selected_policy": policy,
        "policy_applied": True,
        "actual_policy": policy,
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
            "comparison_mode": "paired",
            "timer": "cuda_event",
            "sample_count": protocol.repeats * protocol.rounds,
            "baseline": {"median_ms": 2.0, "p90_ms": 2.0},
            "target": {"median_ms": target_median, "p90_ms": target_median},
            "speedup": speedup,
            "useful_matmul_flops": useful_flops,
            "attention_flops_fraction": 0.25,
            "achieved_tflops": useful_flops / (target_median * 1_000_000_000.0),
            "estimated_logical_operator_bytes": logical_operator_bytes,
            "logical_operator_arithmetic_intensity_flops_per_byte": (
                useful_flops / logical_operator_bytes
            ),
            "estimated_logical_operator_traffic_gbps": (
                logical_operator_bytes / (target_median * 1_000_000.0)
            ),
            "logical_operator_traffic_scope": LOGICAL_OPERATOR_TRAFFIC_SCOPE,
        },
        "execution_path": {
            "requested_policy": policy,
            "selected_policy": policy,
            "attention_backend": "causal_sdpa",
            "runtime_wrapper": "eager",
            "residual_norm_backend": "torch",
            "execution_mode": "eager",
            "fallback_reasons": [],
            "observed_execution": {
                "complete": True,
                "attention_backends": ["causal_sdpa"],
                "residual_norm_backends": ["torch"],
            },
        },
    }
