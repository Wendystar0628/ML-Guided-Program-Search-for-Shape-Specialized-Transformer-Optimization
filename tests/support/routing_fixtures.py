"""Compact builders shared by exact-route tests."""

from __future__ import annotations

import copy
import platform
from types import SimpleNamespace
from typing import Any

import torch
import triton

from runner.candidates import CANDIDATE_SPECS
from runner.contracts import RunVariant
from runner.route_promotion import TUNING_SCHEMA_VERSION
from solution.dispatch import ROUTE_FIELDS, SCHEMA_VERSION
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
) -> dict[str, str]:
    return {
        "device_type": "cuda",
        "device_name": device_name,
        "compute_capability": compute_capability,
        "platform_system": platform.system(),
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "triton": str(triton.__version__),
        "driver": "fixture-driver",
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
    policy: str = "causal-sdpa",
    *,
    case_id: str = "official_02",
    device_name: str = "Fixture GPU",
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "default_policy": "auto",
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
    if candidate.evidence.requires_observed_execution:
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
    implementation_hash: str = "fixture-implementation-hash",
    official_hash: str = "0" * 64,
) -> dict[str, object]:
    candidate = next(
        spec
        for spec in CANDIDATE_SPECS.values()
        if spec.solution_policy == policy and spec.deployable
    )
    target = 2.0 / speedup
    return {
        "candidate_id": candidate.candidate_id,
        "solution_policy": policy,
        "outcome": "success",
        "correctness_passed": True,
        "failed_elements": 0,
        "max_abs_error": 0.0,
        "target_median_ms": target,
        "target_p90_ms": target,
        "speedup": speedup,
        "solution_sha256": implementation_hash,
        "official_snapshot_sha256": official_hash,
        "execution_path": candidate_execution_path(policy),
    }


def formal_summary(
    *,
    case_id: str = "official_02",
    challenger_policy: str = "causal-sdpa",
    challenger_speedup: float = 1.20,
    auto_speedup: float = 1.0,
    implementation_hash: str = "fixture-implementation-hash",
    official_hash: str = "0" * 64,
) -> dict[str, Any]:
    observations = [
        candidate_observation(
            "auto",
            auto_speedup,
            implementation_hash=implementation_hash,
            official_hash=official_hash,
        ),
        candidate_observation(
            challenger_policy,
            challenger_speedup,
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
                    "triton": str(triton.__version__),
                    "driver": "fixture-driver",
                },
                "gpu": {
                    "available": True,
                    "name": "Fixture GPU",
                    "compute_capability": "8.9",
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
