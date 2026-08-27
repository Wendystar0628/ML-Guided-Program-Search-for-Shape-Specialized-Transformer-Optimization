"""Small builders shared by the routing test modules."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import triton

from project_identity import official_snapshot_hash, solution_implementation_hash
from runner.candidates import CANDIDATE_SPECS
from runner.contracts import load_workload_set
from runner.route_promotion import verified_profile_from_probe_result
from solution.dispatch import ROUTE_FIELDS, SCHEMA_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_WORKLOAD_SET_ID = "fixture_transformer_v1"


def transformer_config(*, causal: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=1,
        seq_len=2048,
        d_model=512,
        num_heads=8,
        ffn_dim=2048,
        num_layers=4,
        causal=causal,
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
    device_name: str = "Fixture GPU",
    causal: bool = False,
    batch_size: int = 1,
    seq_len: int = 2048,
    dtype: str = "float16",
) -> dict[str, object]:
    match: dict[str, object] = {
        **route_runtime_identity(device_name=device_name),
        "dtype": dtype,
        "B": batch_size,
        "S": seq_len,
        "D": 512,
        "heads": 8,
        "ffn": 2048,
        "layers": 4,
        "causal": causal,
    }
    assert set(match) == ROUTE_FIELDS
    return match


def exact_route_document(
    policy: str,
    *,
    device_name: str = "Fixture GPU",
    causal: bool = False,
    batch_size: int = 1,
    seq_len: int = 2048,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "default_policy": "auto",
        "routes": [
            {
                "match": exact_match(
                    device_name=device_name,
                    causal=causal,
                    batch_size=batch_size,
                    seq_len=seq_len,
                ),
                "policy": policy,
            }
        ],
    }


def routing_probe_result() -> dict[str, object]:
    return {
        "schema_version": 2,
        "run_id": "fixture-probe",
        "created_at": "2026-08-27T00:00:00+00:00",
        "requested_device": "cuda:0",
        "outcome": "success",
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


def candidate_execution_path(policy: str) -> dict[str, object]:
    candidate = next(
        spec
        for spec in CANDIDATE_SPECS.values()
        if spec.solution_policy == policy and spec.deployable
    )
    path: dict[str, object] = {
        "requested_policy": policy,
        "selected_policy": policy,
        "shape_route": policy,
    }
    for expectation in candidate.evidence.path_expectations:
        path[expectation.field] = min(expectation.accepted_values, key=repr)
    if (
        candidate.evidence.requires_observed_execution
        or candidate.evidence.observed_expectations
    ):
        observed: dict[str, object] = {"complete": True}
        for expectation in candidate.evidence.observed_expectations:
            values = expectation.required_values or frozenset(
                {min(expectation.accepted_values)}
            )
            observed[expectation.field] = sorted(values)
        path["observed_execution"] = observed
    return path


def candidate_observation(
    *,
    candidate_id: str,
    policy: str,
    speedup: float,
) -> dict[str, object]:
    observation = copy.deepcopy(formal_summary()["observations"][2])
    assert isinstance(observation, dict)
    observation.update(
        {
            "candidate_id": candidate_id,
            "solution_policy": policy,
            "conservative_speedup": speedup,
            "baseline_round_medians_ms": [speedup, speedup, speedup],
            "execution_path": candidate_execution_path(policy),
        }
    )
    return observation


def formal_summary() -> dict[str, Any]:
    implementation_hash = "fixture-implementation-hash"
    return {
        "schema_version": 2,
        "tuning_id": "fixture-tuning",
        "complete": True,
        "protocol": {
            "preset": "formal",
            "seed": 1234,
            "accuracy_trials": 5,
            "rtol": 0.01,
            "atol": 0.001,
            "warmup": 20,
            "repeats": 100,
            "rounds": 3,
            "matmul_precision": "high",
            "allow_tf32": True,
        },
        "source_consistent": True,
        "source_solution_sha256": implementation_hash,
        "implementation_consistent": True,
        "source_implementation_sha256": implementation_hash,
        "official_consistent": True,
        "official_snapshot_sha256": "0" * 64,
        "device_profile": {
            "device_type": "cuda",
            "device_name": "Fixture GPU",
            "compute_capability": "8.9",
            "platform_system": platform.system(),
            "torch": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "triton": str(triton.__version__),
            "driver": "fixture-driver",
        },
        "workload": {
            "set_id": FIXTURE_WORKLOAD_SET_ID,
            "sha256": "0" * 64,
            "case": {
                "case_id": "attention_fixture",
                "batch_size": 1,
                "seq_len": 2048,
                "d_model": 512,
                "num_heads": 8,
                "ffn_dim": 2048,
                "num_layers": 4,
                "dtype": "float16",
                "causal": False,
                "padding_ratio": 0.0,
                "input_scale": 1.0,
            },
        },
        "observations": [
            {
                "candidate_id": "compile-default",
                "solution_policy": "auto",
                "compile_solution": True,
                "cuda_graph_solution": False,
                "outcome": "success",
                "correctness_passed": True,
                "failed_elements": 0,
                "policy_applied": True,
                "conservative_speedup": 3.0,
                "baseline_round_medians_ms": [3.0, 3.0, 3.0],
                "target_round_medians_ms": [1.0, 1.0, 1.0],
                "target_median_ms": 0.9,
                "target_p90_ms": 1.0,
                "solution_sha256": implementation_hash,
                "official_snapshot_sha256": "0" * 64,
                "execution_path": {
                    "requested_policy": "auto",
                    "selected_policy": "auto",
                    "shape_route": "compile-control",
                },
            },
            {
                "candidate_id": "eager-auto",
                "solution_policy": "auto",
                "compile_solution": False,
                "cuda_graph_solution": False,
                "outcome": "success",
                "correctness_passed": True,
                "failed_elements": 0,
                "policy_applied": True,
                "conservative_speedup": 1.4,
                "baseline_round_medians_ms": [1.4, 1.4, 1.4],
                "target_round_medians_ms": [1.0, 1.0, 1.0],
                "target_median_ms": 6.5,
                "target_p90_ms": 6.8,
                "solution_sha256": implementation_hash,
                "official_snapshot_sha256": "0" * 64,
                "execution_path": {
                    "requested_policy": "auto",
                    "selected_policy": "auto",
                    "shape_route": "safe-auto",
                },
            },
            {
                "candidate_id": "long-tail-online",
                "solution_policy": "long-tail-online",
                "compile_solution": False,
                "cuda_graph_solution": False,
                "outcome": "success",
                "correctness_passed": True,
                "failed_elements": 0,
                "policy_applied": True,
                "conservative_speedup": 1.5,
                "baseline_round_medians_ms": [1.5, 1.5, 1.5],
                "target_round_medians_ms": [1.0, 1.0, 1.0],
                "target_median_ms": 6.1,
                "target_p90_ms": 6.4,
                "solution_sha256": implementation_hash,
                "official_snapshot_sha256": "0" * 64,
                "execution_path": candidate_execution_path("long-tail-online"),
            },
        ],
    }


def promotion_project(tmp_path: Path) -> Path:
    solution_root = tmp_path / "solution"
    solution_root.mkdir()
    (solution_root / "transformer.py").write_text("VALUE = 1\n", encoding="utf-8")
    _write_official_snapshot(tmp_path)
    return tmp_path


def _write_official_snapshot(project_root: Path) -> str:
    official_root = project_root / "official"
    official_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = official_root / "fixture_benchmark.py"
    if not snapshot_path.exists():
        snapshot_path.write_text("# official fixture\n", encoding="utf-8")
    digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    (official_root / "snapshot.json").write_text(
        json.dumps(
            {
                "snapshot_path": "official/fixture_benchmark.py",
                "byte_count": snapshot_path.stat().st_size,
                "sha256": digest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return digest


def bind_workload_summaries(
    project_root: Path,
    summaries: list[dict[str, Any]],
) -> str:
    official_hash = _write_official_snapshot(project_root)
    cases: list[dict[str, Any]] = []
    for summary in summaries:
        case = summary["workload"]["case"]
        assert isinstance(case, dict)
        cases.append(copy.deepcopy(case))
    case_ids = [str(case["case_id"]) for case in cases]
    document = {
        "schema_version": 1,
        "workload_set_id": FIXTURE_WORKLOAD_SET_ID,
        "groups": [
            {
                "group_id": "fixture",
                "display_name": "Fixture",
                "weight": 1.0,
                "case_ids": case_ids,
            }
        ],
        "ordered_cases": cases,
    }
    workload_path = (
        project_root / "runner" / "workloads" / f"{FIXTURE_WORKLOAD_SET_ID}.json"
    )
    workload_path.parent.mkdir(parents=True, exist_ok=True)
    workload_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    workload_set = load_workload_set(project_root, FIXTURE_WORKLOAD_SET_ID)
    implementation_hash = solution_implementation_hash(project_root / "solution")
    for summary in summaries:
        summary["workload"]["set_id"] = FIXTURE_WORKLOAD_SET_ID
        summary["workload"]["sha256"] = workload_set.sha256
        summary["source_solution_sha256"] = implementation_hash
        summary["source_implementation_sha256"] = implementation_hash
        summary["official_snapshot_sha256"] = official_hash
        summary["official_consistent"] = True
        for observation in summary["observations"]:
            observation["solution_sha256"] = implementation_hash
            observation["official_snapshot_sha256"] = official_hash
    return workload_set.sha256


def promotable_summary(
    project_root: Path,
    *,
    case_id: str = "attention_fixture",
    causal: bool = False,
) -> dict[str, Any]:
    summary = formal_summary()
    case = summary["workload"]["case"]
    case["case_id"] = case_id
    case["causal"] = causal
    summary["tuning_id"] = f"fixture-tuning-{case_id}"
    bind_workload_summaries(project_root, [summary])
    return summary


def s512_summary(case_id: str, *, padding_ratio: float) -> dict[str, Any]:
    summary = formal_summary()
    summary["tuning_id"] = f"fixture-tuning-{case_id}"
    case = summary["workload"]["case"]
    case.update(
        {
            "case_id": case_id,
            "batch_size": 8,
            "seq_len": 512,
            "padding_ratio": padding_ratio,
        }
    )
    winner = summary["observations"][2]
    winner["candidate_id"] = "s512-native-softmax"
    winner["solution_policy"] = "s512-native-softmax"
    winner["execution_path"] = candidate_execution_path("s512-native-softmax")
    return summary


def write_bundle_manifest(
    project_root: Path,
    route_path: Path,
    *,
    workload_set_id: str,
) -> Path:
    workload_set = load_workload_set(project_root, workload_set_id)
    implementation_hash = solution_implementation_hash(project_root / "solution")
    official_hash = official_snapshot_hash(project_root)
    route_document = json.loads(route_path.read_text(encoding="utf-8"))
    route_matches = [entry["match"] for entry in route_document["routes"]]
    route_hashes = [
        hashlib.sha256(
            json.dumps(
                match,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        for match in route_matches
    ]
    manifest_path = route_path.with_name("manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "workload_set": {
                    "set_id": workload_set_id,
                    "sha256": workload_set.sha256,
                },
                "official": {"snapshot_sha256": official_hash},
                "solution": {"implementation_sha256": implementation_hash},
                "route_table": {
                    "sha256": hashlib.sha256(route_path.read_bytes()).hexdigest()
                },
                "formal": {
                    "protocol": {"preset": "formal"},
                    "source_summaries": [
                        {
                            "summary_id": f"fixture-summary-{case.case_id}",
                            "case_id": case.case_id,
                            "route_sha256": route_hashes[
                                min(index, len(route_hashes) - 1)
                            ],
                        }
                        for index, case in enumerate(workload_set.cases)
                    ],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def write_catalog_bundle(
    directory: Path,
    document: dict[str, object],
    *,
    project_root: Path = PROJECT_ROOT,
    workload_set_id: str = "transformer_core_v1",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    route_path = directory / "routes.json"
    route_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    write_bundle_manifest(
        project_root,
        route_path,
        workload_set_id=workload_set_id,
    )
    return route_path


def write_discovery_bundle(
    project_root: Path,
    package: Path,
    profile: dict[str, object] | None = None,
) -> Path:
    solution_root = project_root / "solution"
    solution_root.mkdir(parents=True, exist_ok=True)
    transformer = solution_root / "transformer.py"
    if not transformer.exists():
        transformer.write_text("VALUE = 1\n", encoding="utf-8")
    summary = formal_summary()
    bind_workload_summaries(project_root, [summary])
    package.mkdir(parents=True, exist_ok=True)
    verified_profile = profile or verified_profile_from_probe_result(
        routing_probe_result()
    )
    (package / "profile.json").write_text(
        json.dumps(verified_profile),
        encoding="utf-8",
    )
    route_path = package / "routes.json"
    route_path.write_text(
        json.dumps(exact_route_document("auto")),
        encoding="utf-8",
    )
    write_bundle_manifest(
        project_root,
        route_path,
        workload_set_id=FIXTURE_WORKLOAD_SET_ID,
    )
    (package / "README.md").write_text("fixture\n", encoding="utf-8")
    (package / "run_verified.py").write_text("# fixture\n", encoding="utf-8")
    results = package / "results"
    results.mkdir()
    (results / ".gitignore").write_text("*\n", encoding="utf-8")
    return route_path


__all__ = [
    "FIXTURE_WORKLOAD_SET_ID",
    "PROJECT_ROOT",
    "bind_workload_summaries",
    "candidate_execution_path",
    "candidate_observation",
    "exact_match",
    "exact_route_document",
    "formal_summary",
    "promotable_summary",
    "promotion_project",
    "route_runtime_identity",
    "routing_probe_result",
    "s512_summary",
    "transformer_config",
    "write_bundle_manifest",
    "write_catalog_bundle",
    "write_discovery_bundle",
]
