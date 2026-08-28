"""Compact builders shared by exact-route tests."""

from __future__ import annotations

import copy
import platform
from types import SimpleNamespace
from typing import Any

import torch

from route_contracts import ROUTE_FIELDS, SCHEMA_VERSION
from runner.candidates import CANDIDATE_SPECS
from runner.contracts import RunVariant
from runner.tuning_contracts import TUNING_SCHEMA_VERSION
from tests.support.runner_fixtures import official_shape


def transformer_config(*, case_id: str = "official_02") -> SimpleNamespace:
    shape = official_shape(case_id)
    return SimpleNamespace(
        batch_size=shape.batch_size,
        seq_len=shape.seq_len,
        d_model=shape.d_model,
        num_heads=shape.num_heads,
        ffn_dim=shape.ffn_dim,
        num_layers=shape.num_layers,
        causal=shape.causal,
    )


def route_runtime_identity(
    *,
    device_name: str = "Fixture GPU",
    compute_capability: str = "8.9",
) -> dict[str, object]:
    return {
        "device_type": "cuda",
        "device_name": device_name,
        "compute_capability": compute_capability,
        "platform_system": platform.system(),
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "driver": "fixture-driver",
        "matmul_precision": "high",
        "allow_tf32": True,
    }


def exact_match(
    *,
    case_id: str = "official_02",
    device_name: str = "Fixture GPU",
    dtype: str = "float32",
) -> dict[str, object]:
    shape = official_shape(case_id)
    match: dict[str, object] = {
        **route_runtime_identity(device_name=device_name),
        "dtype": dtype,
        "B": shape.batch_size,
        "S": shape.seq_len,
        "D": shape.d_model,
        "heads": shape.num_heads,
        "ffn": shape.ffn_dim,
        "layers": shape.num_layers,
        "causal": shape.causal,
    }
    assert set(match) == ROUTE_FIELDS
    return match


def exact_route_document(
    policy: str = "graph",
    *,
    case_id: str = "official_02",
    device_name: str = "Fixture GPU",
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "default_policy": "eager-sdpa",
        "routes": [
            {
                "match": exact_match(
                    case_id=case_id,
                    device_name=device_name,
                ),
                "policy": policy,
            }
        ],
    }


def candidate_execution_path(policy: str) -> dict[str, object]:
    candidate = next(
        spec
        for spec in CANDIDATE_SPECS.values()
        if spec.solution_policy == policy and spec.deployable
    )
    path: dict[str, object] = {
        "requested_policy": policy,
        "selected_policy": policy,
    }
    for expectation in candidate.evidence.path_expectations:
        path[expectation.field] = min(expectation.accepted_values, key=repr)
    observed: dict[str, object] = {"complete": True}
    for expectation in candidate.evidence.observed_expectations:
        values = expectation.required_values or frozenset(
            {min(expectation.accepted_values)}
        )
        observed[expectation.field] = sorted(values)
    path["observed_execution"] = observed
    return path


def candidate_observation(
    policy: str,
    speedup: float,
    *,
    target_median_ms: float | None = None,
    target_p90_ms: float | None = None,
    implementation_hash: str = "fixture-implementation-hash",
    official_hash: str = "0" * 64,
) -> dict[str, object]:
    candidate = next(
        spec
        for spec in CANDIDATE_SPECS.values()
        if spec.solution_policy == policy and spec.deployable
    )
    target = 2.0 / speedup if target_median_ms is None else target_median_ms
    p90 = target if target_p90_ms is None else target_p90_ms
    return {
        "candidate_id": candidate.candidate_id,
        "solution_policy": policy,
        "outcome": "success",
        "correctness_passed": True,
        "failed_elements": 0,
        "max_abs_error": 0.0,
        "target_median_ms": target,
        "target_p90_ms": p90,
        "speedup": speedup,
        "solution_sha256": implementation_hash,
        "official_snapshot_sha256": official_hash,
        "execution_path": candidate_execution_path(policy),
    }


def formal_summary(
    *,
    case_id: str = "official_02",
    challenger_policy: str = "graph",
    challenger_speedup: float = 1.20,
    control_speedup: float = 1.0,
    challenger_target_median_ms: float | None = None,
    challenger_target_p90_ms: float | None = None,
    control_target_median_ms: float | None = None,
    control_target_p90_ms: float | None = None,
    implementation_hash: str = "fixture-implementation-hash",
    official_hash: str = "0" * 64,
) -> dict[str, Any]:
    observations = [
        candidate_observation(
            "eager-sdpa",
            control_speedup,
            target_median_ms=control_target_median_ms,
            target_p90_ms=control_target_p90_ms,
            implementation_hash=implementation_hash,
            official_hash=official_hash,
        ),
        candidate_observation(
            challenger_policy,
            challenger_speedup,
            target_median_ms=challenger_target_median_ms,
            target_p90_ms=challenger_target_p90_ms,
            implementation_hash=implementation_hash,
            official_hash=official_hash,
        ),
    ]
    return {
        "schema_version": TUNING_SCHEMA_VERSION,
        "tuning_id": f"formal-{case_id}",
        "complete": True,
        "protocol": {
            "preset": "formal",
            "seed": 1234,
            "accuracy_trials": 5,
            "rtol": 0.02,
            "atol": 0.002,
            "warmup": 20,
            "repeats": 100,
            "rounds": 3,
            "compile_baseline": False,
            "compile_solution": False,
            "compile_mode": "default",
            "matmul_precision": "high",
            "allow_tf32": True,
        },
        "source_consistent": True,
        "source_solution_sha256": implementation_hash,
        "implementation_consistent": True,
        "source_implementation_sha256": implementation_hash,
        "official_consistent": True,
        "official_snapshot_sha256": official_hash,
        "device_profile": route_runtime_identity(),
        "case_id": case_id,
        "workload": {
            "set_id": "official_transformer_v1",
            "sha256": "1" * 64,
            "shape": official_shape(case_id).as_dict(),
            "variant": RunVariant().as_dict(),
        },
        "observations": observations,
        "winner": copy.deepcopy(observations[-1]),
        "deployable_winner": copy.deepcopy(observations[-1]),
    }


def routing_probe_result() -> dict[str, object]:
    return {
        "schema_version": 3,
        "run_id": "fixture-probe",
        "created_at": "2026-08-28T00:00:00+00:00",
        "requested_device": "cuda:0",
        "outcome": "success",
        "environment": {
            "device": "cuda:0",
            "gpu": "Fixture GPU",
            "compute_capability": "8.9",
            "torch": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "driver": "fixture-driver",
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
                "platform": {
                    "system": platform.system(),
                    "machine": platform.machine(),
                },
                "software": {
                    "torch": str(torch.__version__),
                    "cuda_runtime": str(torch.version.cuda),
                    "driver": "fixture-driver",
                    "efficient_sdpa_enabled": True,
                    "cudnn_sdpa_enabled": True,
                },
                "gpu": {
                    "available": True,
                    "name": "Fixture GPU",
                    "compute_capability": "8.9",
                    "free_memory_bytes": 12 * 1024**3,
                },
            },
            "performance_anchors": {
                "eager_launch": {
                    "available": True,
                    "effective_latency_us": 4.0,
                }
            },
        },
    }
