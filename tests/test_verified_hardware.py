from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from project_identity import official_snapshot_hash, solution_implementation_hash
from runner import verified_hardware as verified
from runner.supervisor import CancellationToken
from runner.sweep import BenchmarkSweepService
from solution.dispatch import SCHEMA_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_SET_ID = "transformer_core_v1"

IDENTITY = {
    "gpu": {
        "name": "NVIDIA GeForce RTX 4080",
        "compute_capability": "8.9",
    },
    "platform": {"system": "Windows", "machine": "AMD64"},
    "software": {
        "torch": "2.12.1+cu132",
        "cuda_runtime": "13.2",
        "triton": "3.7.1",
        "driver": "610.88",
    },
}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_official_snapshot(project_root: Path) -> str:
    snapshot_path = project_root / "official" / "torch_transformer_benchmark.py"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    content = b"VALUE = 1\n"
    snapshot_path.write_bytes(content)
    _write_json(
        project_root / "official" / "snapshot.json",
        {
            "snapshot_path": "official/torch_transformer_benchmark.py",
            "byte_count": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        },
    )
    return official_snapshot_hash(project_root)


def _profile() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "device_operation_passed": True,
        "hardware_profile": {"device_type": "cuda", **IDENTITY},
    }


def _case(case_id: str, *, sequence_length: int) -> dict[str, Any]:
    launch_shape = sequence_length == 64
    return {
        "case_id": case_id,
        "batch_size": 1,
        "seq_len": sequence_length,
        "d_model": 256 if launch_shape else 64,
        "num_heads": 8 if launch_shape else 1,
        "ffn_dim": 1024 if launch_shape else 128,
        "num_layers": 4 if launch_shape else 1,
        "dtype": "float16",
        "causal": False,
        "padding_ratio": 0.0,
        "input_scale": 1.0,
    }


def _workload_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workload_set_id": WORKLOAD_SET_ID,
        "groups": [
            {
                "group_id": "fixture",
                "display_name": "Fixture",
                "weight": 1.0,
                "case_ids": ["calibrated", "fallback"],
            }
        ],
        "ordered_cases": [
            _case("calibrated", sequence_length=64),
            _case("fallback", sequence_length=128),
        ],
    }


def _routes() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "default_policy": "auto",
        "routes": [
            {
                "match": {
                    "device_type": "cuda",
                    "device_name": IDENTITY["gpu"]["name"],
                    "compute_capability": IDENTITY["gpu"]["compute_capability"],
                    "platform_system": IDENTITY["platform"]["system"],
                    "torch": IDENTITY["software"]["torch"],
                    "cuda_runtime": IDENTITY["software"]["cuda_runtime"],
                    "triton": IDENTITY["software"]["triton"],
                    "driver": IDENTITY["software"]["driver"],
                    "dtype": "float16",
                    "B": 1,
                    "S": 64,
                    "D": 256,
                    "heads": 8,
                    "ffn": 1024,
                    "layers": 4,
                    "causal": False,
                },
                "policy": "cuda-graph",
            },
            {
                "match": {
                    "device_type": "cuda",
                    "device_name": IDENTITY["gpu"]["name"],
                    "compute_capability": IDENTITY["gpu"]["compute_capability"],
                    "platform_system": IDENTITY["platform"]["system"],
                    "torch": IDENTITY["software"]["torch"],
                    "cuda_runtime": IDENTITY["software"]["cuda_runtime"],
                    "triton": IDENTITY["software"]["triton"],
                    "driver": IDENTITY["software"]["driver"],
                    "dtype": "float16",
                    "B": 1,
                    "S": 128,
                    "D": 64,
                    "heads": 1,
                    "ffn": 128,
                    "layers": 1,
                    "causal": False,
                },
                "policy": "auto",
            },
        ],
    }


def _bundle_paths(tmp_path: Path) -> verified.BundlePaths:
    project_root = tmp_path / "project"
    bundle_root = project_root / "verified_hardware" / "nvidia_geforce_rtx_4080"
    return verified.BundlePaths(
        project_root=project_root,
        bundle_root=bundle_root,
        profile=bundle_root / "profile.json",
        routes=bundle_root / "routes.json",
        sweeps=bundle_root / "results" / "sweeps",
    )


def _write_manifest(
    paths: verified.BundlePaths,
    *,
    workload_document: dict[str, Any] | None = None,
    routes: dict[str, Any] | None = None,
) -> None:
    workload_document = workload_document or _workload_document()
    routes = routes or _routes()
    solution_root = paths.project_root / "solution"
    solution_root.mkdir(parents=True, exist_ok=True)
    transformer = solution_root / "transformer.py"
    if not transformer.exists():
        transformer.write_text("VALUE = 1\n", encoding="utf-8")
    workload_path = (
        paths.project_root / "runner" / "workloads" / f"{WORKLOAD_SET_ID}.json"
    )
    _write_json(workload_path, workload_document)
    _write_json(paths.routes, routes)
    workload_set = verified.load_workload_set(
        paths.project_root,
        WORKLOAD_SET_ID,
    )
    official_hash = _write_official_snapshot(paths.project_root)
    implementation_hash = solution_implementation_hash(solution_root)
    protocol = {"preset": "formal"}
    source_summaries = [
        {
            "summary_id": f"fixture-summary-{case['case_id']}",
            "case_id": case["case_id"],
            "route_sha256": hashlib.sha256(
                json.dumps(
                    route["match"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
        }
        for case, route in zip(
            workload_document["ordered_cases"],
            routes["routes"],
            strict=True,
        )
    ]
    _write_json(
        paths.manifest,
        {
            "schema_version": 2,
            "workload_set": {
                "set_id": WORKLOAD_SET_ID,
                "sha256": workload_set.sha256,
            },
            "official": {"snapshot_sha256": official_hash},
            "solution": {"implementation_sha256": implementation_hash},
            "route_table": {
                "sha256": hashlib.sha256(paths.routes.read_bytes()).hexdigest()
            },
            "formal": {
                "protocol": protocol,
                "source_summaries": source_summaries,
            },
        },
    )


def _benchmark_result(
    *,
    case: dict[str, Any],
    workload_sha256: str,
    route_source: str,
    route_sha256: str,
    policy: str,
    route_origin: str,
    baseline_ms: float,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    protocol = protocol or {"accuracy_trials": 1, "repeats": 1, "rounds": 1}
    rounds = int(protocol["rounds"])
    repeats = int(protocol["repeats"])
    return {
        "schema_version": 2,
        "run_id": f"run-{case['case_id']}",
        "sweep_id": "fixture-sweep",
        "created_at": "2026-08-27T00:00:00+00:00",
        "run_kind": "benchmark",
        "target": "solution",
        "outcome": "success",
        "workload": {
            "set_id": WORKLOAD_SET_ID,
            "sha256": workload_sha256,
            "case": case,
        },
        "source": {
            "official_sha256": "official-hash",
            "solution_sha256": "solution-hash",
        },
        "protocol": protocol,
        "environment": {"device": "cuda:0", "gpu": IDENTITY["gpu"]["name"]},
        "correctness": {
            "passed": True,
            "trial_count": protocol["accuracy_trials"],
            "failed_elements": 0,
            "max_abs_error": 0.0,
        },
        "performance": {
            "timer": "cuda_event",
            "sample_count": repeats * rounds,
            "baseline": {
                "median_ms": baseline_ms,
                "p90_ms": baseline_ms,
                "round_medians_ms": [baseline_ms] * rounds,
            },
            "target": {
                "median_ms": 1.0,
                "p90_ms": 1.0,
                "round_medians_ms": [1.0] * rounds,
            },
            "speedup": baseline_ms,
        },
        "execution_path": {
            "requested_policy": "dispatch",
            "dispatch_source": route_source,
            "dispatch_table_sha256": route_sha256,
            "dispatch_policy": policy,
            "selected_policy": policy,
            "route_origin": route_origin,
            **(
                {
                    "runtime_wrapper": "solution_eager_cuda_graph",
                    "observed_execution": {"complete": True},
                }
                if policy == "cuda-graph"
                else {}
            ),
        },
    }


def test_profile_validation_only_requires_the_same_gpu() -> None:
    expected = verified.expected_hardware_identity(_profile())
    verified.validate_hardware_identity(expected, IDENTITY)

    software_drift = {
        **IDENTITY,
        "software": {**IDENTITY["software"], "triton": "3.8.0"},
    }
    verified.validate_hardware_identity(expected, software_drift)

    platform_drift = {
        **IDENTITY,
        "platform": {"system": "Linux", "machine": "aarch64"},
    }
    verified.validate_hardware_identity(expected, platform_drift)

    different_gpu = {
        **IDENTITY,
        "gpu": {**IDENTITY["gpu"], "name": "Different GPU"},
    }
    with pytest.raises(verified.VerifiedHardwareError, match="gpu.name"):
        verified.validate_hardware_identity(expected, different_gpu)


def test_runtime_identity_reports_an_unavailable_cuda_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fail_properties(_index: int) -> None:
        raise RuntimeError("invalid device ordinal")

    monkeypatch.setattr(torch.cuda, "get_device_properties", fail_properties)

    with pytest.raises(verified.VerifiedHardwareError, match="cannot inspect"):
        verified.collect_runtime_identity("cuda:99")


def test_run_verified_attributes_routes_and_writes_compact_summary(
    tmp_path: Path,
) -> None:
    paths = _bundle_paths(tmp_path)
    workload_document = _workload_document()
    _write_json(
        paths.project_root
        / "runner"
        / "workloads"
        / f"{WORKLOAD_SET_ID}.json",
        workload_document,
    )
    _write_json(paths.profile, _profile())
    _write_manifest(paths, workload_document=workload_document)

    def fake_run_case(project_root: Path, **arguments: Any) -> tuple[dict[str, Any], Path]:
        assert project_root == paths.project_root.resolve()
        assert arguments["device"] == "cuda:0"
        assert arguments["target"] == "solution"
        assert arguments["solution_policy"] == "dispatch"
        workload_set = verified.load_workload_set(
            paths.project_root,
            WORKLOAD_SET_ID,
        )
        digest = hashlib.sha256(paths.routes.read_bytes()).hexdigest()
        source = verified._portable_source(paths.routes, paths.project_root)
        case = arguments["case"].as_dict()
        index = [
            item["case_id"] for item in workload_document["ordered_cases"]
        ].index(case["case_id"])
        policy, origin = verified._expected_route(_routes(), case, IDENTITY)
        run = _benchmark_result(
            case=case,
            workload_sha256=workload_set.sha256,
            route_source=source,
            route_sha256=digest,
            policy=policy,
            route_origin=origin,
            baseline_ms=2.0 + index,
            protocol=arguments["protocol"].as_dict(),
        )
        run["sweep_id"] = arguments["sweep_id"]
        result_path = arguments["result_dir"] / f"{case['case_id']}.json"
        _write_json(result_path, run)
        return run, result_path

    first_summary_path = verified.run_verified(
        verified.LaunchConfig(device="cuda:0", preset="formal"),
        paths=paths,
        identity_collector=lambda _device: IDENTITY,
        sweep_service=BenchmarkSweepService(fake_run_case),
    )
    first_reference = json.loads(paths.reference_formal.read_text(encoding="utf-8"))

    summary_path = verified.run_verified(
        verified.LaunchConfig(device="cuda:0", preset="formal"),
        paths=paths,
        identity_collector=lambda _device: IDENTITY,
        sweep_service=BenchmarkSweepService(fake_run_case),
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert first_summary_path != summary_path
    assert first_reference["sweep_id"] != summary["sweep_id"]
    assert summary_path.parent.parent == paths.sweeps.resolve()
    assert summary_path.name == "summary.json"
    assert json.loads(paths.reference_formal.read_text(encoding="utf-8")) == summary
    assert summary["dispatch"]["source"] == (
        "verified_hardware/nvidia_geforce_rtx_4080/routes.json"
    )
    assert [item["dispatch_policy"] for item in summary["case_results"]] == [
        "cuda-graph",
        "auto",
    ]
    assert [item["route_origin"] for item in summary["case_results"]] == [
        "calibrated",
        "calibrated",
    ]
    assert [item["selected_policy"] for item in summary["case_results"]] == [
        "cuda-graph",
        "auto",
    ]
    assert all(item["policy_applied"] is True for item in summary["case_results"])
    assert [item["run_id"] for item in summary["case_results"]] == [
        "run-calibrated",
        "run-fallback",
    ]
    assert all("run_file" not in item for item in summary["case_results"])
    assert math.isclose(summary["group_balanced_geomean_speedup"], math.sqrt(6.0))


def test_run_verified_propagates_cancellation_without_replacing_reference(
    tmp_path: Path,
) -> None:
    paths = _bundle_paths(tmp_path)
    workload_document = _workload_document()
    _write_json(paths.profile, _profile())
    _write_manifest(paths, workload_document=workload_document)
    previous_reference = {"sweep_outcome": "complete", "sweep_id": "previous"}
    _write_json(paths.reference_formal, previous_reference)
    token = CancellationToken()
    token.cancel()

    summary_path = verified.run_verified(
        verified.LaunchConfig(device="cuda", preset="formal"),
        paths=paths,
        identity_collector=lambda _device: IDENTITY,
        sweep_service=BenchmarkSweepService(
            lambda *args, **kwargs: pytest.fail(
                "a pre-cancelled verified sweep must not start a case"
            )
        ),
        cancellation_token=token,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["sweep_outcome"] == "cancelled"
    assert json.loads(paths.reference_formal.read_text(encoding="utf-8")) == (
        previous_reference
    )


def test_validate_run_routes_rejects_an_unexpected_table_hash(tmp_path: Path) -> None:
    paths = _bundle_paths(tmp_path)
    case = _case("calibrated", sequence_length=64)
    run = _benchmark_result(
        case=case,
        workload_sha256="workload-hash",
        route_source=verified._portable_source(paths.routes, paths.project_root),
        route_sha256="wrong-hash",
        policy="cuda-graph",
        route_origin="calibrated",
        baseline_ms=2.0,
    )

    with pytest.raises(verified.VerifiedHardwareError, match="route-table hash"):
        verified.validate_run_routes(
            [run],
            routes=_routes(),
            identity=IDENTITY,
            route_path=paths.routes,
            route_sha256="expected-hash",
            project_root=paths.project_root,
        )


def test_validate_run_routes_rejects_an_internal_policy_fallback(
    tmp_path: Path,
) -> None:
    paths = _bundle_paths(tmp_path)
    case = _case("calibrated", sequence_length=64)
    route_sha256 = "expected-hash"
    run = _benchmark_result(
        case=case,
        workload_sha256="workload-hash",
        route_source=verified._portable_source(paths.routes, paths.project_root),
        route_sha256=route_sha256,
        policy="cuda-graph",
        route_origin="calibrated",
        baseline_ms=2.0,
    )
    # The route labels still claim CUDA Graph, but the reported wrapper proves
    # the specialized execution path did not run.
    run["execution_path"]["runtime_wrapper"] = "torch_eager"

    with pytest.raises(verified.VerifiedHardwareError, match="without fallback"):
        verified.validate_run_routes(
            [run],
            routes=_routes(),
            identity=IDENTITY,
            route_path=paths.routes,
            route_sha256=route_sha256,
            project_root=paths.project_root,
        )


def test_verified_workload_rejects_a_missing_exact_route(tmp_path: Path) -> None:
    paths = _bundle_paths(tmp_path)
    _write_json(
        paths.project_root
        / "runner"
        / "workloads"
        / f"{WORKLOAD_SET_ID}.json",
        _workload_document(),
    )
    workload_set = verified.load_workload_set(
        paths.project_root,
        WORKLOAD_SET_ID,
    )
    routes = _routes()
    routes["routes"] = routes["routes"][:1]

    with pytest.raises(verified.VerifiedHardwareError, match="no exact decision"):
        verified.validate_workload_route_coverage(
            workload_set,
            routes=routes,
            identity=IDENTITY,
        )


def test_verified_bundle_can_return_from_runtime_b_to_exact_runtime_a(
    tmp_path: Path,
) -> None:
    paths = _bundle_paths(tmp_path)
    _write_json(
        paths.project_root
        / "runner"
        / "workloads"
        / f"{WORKLOAD_SET_ID}.json",
        _workload_document(),
    )
    workload_set = verified.load_workload_set(paths.project_root, WORKLOAD_SET_ID)
    routes = _routes()
    runtime_b = copy.deepcopy(IDENTITY)
    runtime_b["platform"]["system"] = "Linux"
    runtime_b["software"].update(
        {
            "torch": "different-torch",
            "cuda_runtime": "different-cuda",
            "triton": "different-triton",
            "driver": "different-driver",
        }
    )
    for route in copy.deepcopy(routes["routes"]):
        route["match"].update(
            {
                "platform_system": runtime_b["platform"]["system"],
                "torch": runtime_b["software"]["torch"],
                "cuda_runtime": runtime_b["software"]["cuda_runtime"],
                "triton": runtime_b["software"]["triton"],
                "driver": runtime_b["software"]["driver"],
            }
        )
        routes["routes"].append(route)

    profile_a = _profile()
    profile_b = copy.deepcopy(profile_a)
    profile_b["hardware_profile"]["platform"] = runtime_b["platform"]
    profile_b["hardware_profile"]["software"] = runtime_b["software"]

    for profile, runtime in (
        (profile_a, IDENTITY),
        (profile_b, runtime_b),
        (profile_b, IDENTITY),
    ):
        expected = verified.expected_hardware_identity(profile)
        verified.validate_hardware_identity(expected, runtime)
        verified.validate_workload_route_coverage(
            workload_set,
            routes=routes,
            identity=runtime,
        )


def test_verified_workload_rejects_a_broad_route(tmp_path: Path) -> None:
    paths = _bundle_paths(tmp_path)
    _write_json(
        paths.project_root
        / "runner"
        / "workloads"
        / f"{WORKLOAD_SET_ID}.json",
        _workload_document(),
    )
    workload_set = verified.load_workload_set(
        paths.project_root,
        WORKLOAD_SET_ID,
    )
    routes = {
        "schema_version": SCHEMA_VERSION,
        "default_policy": "auto",
        "routes": [{"match": {"device_type": "cuda"}, "policy": "auto"}],
    }

    with pytest.raises(verified.VerifiedHardwareError, match="must be exact"):
        verified.validate_workload_route_coverage(
            workload_set,
            routes=routes,
            identity=IDENTITY,
        )


def test_reference_formal_is_the_current_unified_sweep_summary() -> None:
    bundle = PROJECT_ROOT / "verified_hardware" / "nvidia_geforce_rtx_4080"
    reference = json.loads(
        (bundle / "results" / "reference_formal.json").read_text(encoding="utf-8")
    )
    routes_path = bundle / "routes.json"
    workload_set = verified.load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)

    assert reference["schema_version"] == 1
    assert reference["target"] == "solution"
    assert reference["sweep_outcome"] == "complete"
    assert reference["failed_cases"] == []
    assert reference["workload_set_id"] == workload_set.workload_set_id
    assert reference["workload_sha256"] == workload_set.sha256
    assert reference["source"] == {
        "official_sha256": official_snapshot_hash(PROJECT_ROOT),
        "solution_sha256": solution_implementation_hash(PROJECT_ROOT / "solution"),
    }
    assert reference["dispatch"] == {
        "source": "verified_hardware/nvidia_geforce_rtx_4080/routes.json",
        "sha256": hashlib.sha256(routes_path.read_bytes()).hexdigest(),
    }
    case_results = reference["case_results"]
    assert [item["case_id"] for item in case_results] == [
        case.case_id for case in workload_set.cases
    ]
    assert all(item["route_origin"] == "calibrated" for item in case_results)
    assert all(item["policy_applied"] is True for item in case_results)
    assert all(isinstance(item["run_id"], str) for item in case_results)
    assert all("run_file" not in item for item in case_results)
