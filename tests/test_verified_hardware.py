from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pytest
import torch

from runner import verified_hardware as verified

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
    },
}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _profile() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "device_operation_passed": True,
        "hardware_profile": IDENTITY,
    }


def _case(case_id: str, *, sequence_length: int) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "batch_size": 1,
        "seq_len": sequence_length,
        "d_model": 64,
        "num_heads": 1,
        "ffn_dim": 128,
        "num_layers": 1,
        "dtype": "float16",
        "causal": False,
        "padding_ratio": 0.0,
        "input_scale": 1.0,
    }


def _workload_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workload_set_id": verified.WORKLOAD_SET_ID,
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
        "schema_version": 2,
        "default_policy": "auto",
        "routes": [
            {
                "match": {
                    "device_type": "cuda",
                    "device_name": IDENTITY["gpu"]["name"],
                    "compute_capability": IDENTITY["gpu"][
                        "compute_capability"
                    ],
                    "platform_system": IDENTITY["platform"]["system"],
                    "torch": IDENTITY["software"]["torch"],
                    "cuda_runtime": IDENTITY["software"]["cuda_runtime"],
                    "triton": IDENTITY["software"]["triton"],
                    "dtype": "float16",
                    "B": 1,
                    "S": 64,
                    "D": 64,
                    "heads": 1,
                    "ffn": 128,
                    "layers": 1,
                    "causal": False,
                },
                "policy": "cuda-graph",
            },
            {
                "match": {
                    "device_type": "cuda",
                    "device_name": IDENTITY["gpu"]["name"],
                    "compute_capability": IDENTITY["gpu"][
                        "compute_capability"
                    ],
                    "platform_system": IDENTITY["platform"]["system"],
                    "torch": IDENTITY["software"]["torch"],
                    "cuda_runtime": IDENTITY["software"]["cuda_runtime"],
                    "triton": IDENTITY["software"]["triton"],
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
    bundle_root = (
        project_root / "verified_hardware" / "nvidia_geforce_rtx_4080"
    )
    return verified.BundlePaths(
        project_root=project_root,
        bundle_root=bundle_root,
        profile=bundle_root / "profile.json",
        routes=bundle_root / "routes.json",
        runs=bundle_root / "results" / "runs",
        summaries=bundle_root / "results" / "summaries",
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
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "run_id": f"run-{case['case_id']}",
        "sweep_id": "fixture-sweep",
        "created_at": "2026-08-27T00:00:00+00:00",
        "run_kind": "benchmark",
        "target": "solution",
        "outcome": "success",
        "workload": {
            "set_id": verified.WORKLOAD_SET_ID,
            "sha256": workload_sha256,
            "case": case,
        },
        "source": {
            "official_sha256": "official-hash",
            "solution_sha256": "solution-hash",
        },
        "protocol": {"accuracy_trials": 1, "repeats": 1, "rounds": 1},
        "environment": {"device": "cuda:0", "gpu": IDENTITY["gpu"]["name"]},
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
                "median_ms": baseline_ms,
                "p90_ms": baseline_ms,
                "round_medians_ms": [baseline_ms],
            },
            "target": {
                "median_ms": 1.0,
                "p90_ms": 1.0,
                "round_medians_ms": [1.0],
            },
            "speedup": baseline_ms,
        },
        "execution_path": {
            "dispatch_source": route_source,
            "dispatch_table_sha256": route_sha256,
            "dispatch_policy": policy,
            "route_origin": route_origin,
        },
    }


def test_runtime_identity_requires_an_exact_route_stack() -> None:
    expected = verified.expected_runtime_identity(_profile())
    verified.validate_runtime_identity(expected, IDENTITY)

    changed = {
        **IDENTITY,
        "software": {**IDENTITY["software"], "triton": "3.8.0"},
    }
    with pytest.raises(verified.VerifiedHardwareError, match="software.triton"):
        verified.validate_runtime_identity(expected, changed)


def test_runtime_identity_reports_an_unavailable_cuda_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fail_properties(_index: int) -> None:
        raise RuntimeError("invalid device ordinal")

    monkeypatch.setattr(torch.cuda, "get_device_properties", fail_properties)

    with pytest.raises(verified.VerifiedHardwareError, match="cannot inspect"):
        verified.collect_runtime_identity("cuda:99")


def test_build_benchmark_command_uses_the_shared_runner_and_local_results(
    tmp_path: Path,
) -> None:
    paths = _bundle_paths(tmp_path)
    command = verified.build_benchmark_command(
        verified.LaunchConfig(device="cuda:1", preset="smoke", timeout=45.0),
        paths,
    )

    assert command[:4] == [verified.sys.executable, "-m", "runner", "benchmark"]
    assert command[command.index("--workload-set") + 1] == verified.WORKLOAD_SET_ID
    assert command[command.index("--device") + 1] == "cuda:1"
    assert command[command.index("--preset") + 1] == "smoke"
    assert command[command.index("--timeout") + 1] == "45"
    assert Path(command[command.index("--result-dir") + 1]) == paths.runs.resolve()


def test_run_verified_attributes_routes_and_writes_compact_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _bundle_paths(tmp_path)
    workload_document = _workload_document()
    _write_json(
        paths.project_root
        / "runner"
        / "workloads"
        / f"{verified.WORKLOAD_SET_ID}.json",
        workload_document,
    )
    _write_json(paths.profile, _profile())
    _write_json(paths.routes, _routes())
    monkeypatch.delenv(verified.ROUTE_TABLE_ENV, raising=False)

    def fake_command_runner(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert command == verified.build_benchmark_command(
            verified.LaunchConfig(device="cuda:0", preset="smoke"),
            paths,
        )
        assert cwd == paths.project_root
        assert check is False
        assert env[verified.ROUTE_TABLE_ENV] == str(paths.routes.resolve())
        assert verified.ROUTE_TABLE_ENV not in verified.os.environ

        workload_set = verified.load_workload_set(
            paths.project_root,
            verified.WORKLOAD_SET_ID,
        )
        digest = verified.route_table_sha256(paths.routes)
        source = verified._portable_source(paths.routes, paths.project_root)
        for index, case in enumerate(workload_document["ordered_cases"]):
            policy, origin = verified._expected_route(_routes(), case, IDENTITY)
            run = _benchmark_result(
                case=case,
                workload_sha256=workload_set["sha256"],
                route_source=source,
                route_sha256=digest,
                policy=policy,
                route_origin=origin,
                baseline_ms=2.0 + index,
            )
            _write_json(paths.runs / f"{index}.json", run)
        return subprocess.CompletedProcess(command, 0)

    summary_path = verified.run_verified(
        verified.LaunchConfig(device="cuda:0", preset="smoke"),
        paths=paths,
        identity_collector=lambda _device: IDENTITY,
        command_runner=fake_command_runner,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_path == paths.summaries / "fixture-sweep.json"
    assert summary["route_table"]["source"] == (
        "verified_hardware/nvidia_geforce_rtx_4080/routes.json"
    )
    assert [item["policy"] for item in summary["case_results"]] == [
        "cuda-graph",
        "auto",
    ]
    assert [item["route_origin"] for item in summary["case_results"]] == [
        "calibrated",
        "calibrated",
    ]
    assert math.isclose(summary["group_balanced_geomean_speedup"], math.sqrt(6.0))


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


def test_verified_workload_rejects_a_missing_exact_route(tmp_path: Path) -> None:
    paths = _bundle_paths(tmp_path)
    _write_json(
        paths.project_root
        / "runner"
        / "workloads"
        / f"{verified.WORKLOAD_SET_ID}.json",
        _workload_document(),
    )
    workload_set = verified.load_workload_set(
        paths.project_root,
        verified.WORKLOAD_SET_ID,
    )
    routes = _routes()
    routes["routes"] = routes["routes"][:1]

    with pytest.raises(verified.VerifiedHardwareError, match="no exact decision"):
        verified.validate_workload_route_coverage(
            workload_set,
            routes=routes,
            identity=IDENTITY,
        )


def test_verified_workload_rejects_a_broad_route(tmp_path: Path) -> None:
    paths = _bundle_paths(tmp_path)
    _write_json(
        paths.project_root
        / "runner"
        / "workloads"
        / f"{verified.WORKLOAD_SET_ID}.json",
        _workload_document(),
    )
    workload_set = verified.load_workload_set(
        paths.project_root,
        verified.WORKLOAD_SET_ID,
    )
    routes = {
        "schema_version": 2,
        "default_policy": "auto",
        "routes": [{"match": {"device_type": "cuda"}, "policy": "auto"}],
    }

    with pytest.raises(verified.VerifiedHardwareError, match="not an exact"):
        verified.validate_workload_route_coverage(
            workload_set,
            routes=routes,
            identity=IDENTITY,
        )


def test_checked_formal_reference_matches_the_current_bundle() -> None:
    bundle = PROJECT_ROOT / "verified_hardware" / "nvidia_geforce_rtx_4080"
    reference = json.loads(
        (bundle / "results" / "reference_formal.json").read_text(encoding="utf-8")
    )
    routes = json.loads((bundle / "routes.json").read_text(encoding="utf-8"))
    profile = json.loads((bundle / "profile.json").read_text(encoding="utf-8"))
    workload_set = verified.load_workload_set(
        PROJECT_ROOT,
        verified.WORKLOAD_SET_ID,
    )

    assert reference["schema_version"] == 2
    assert reference["workload_set"]["set_id"] == verified.WORKLOAD_SET_ID
    assert reference["workload_set"]["sha256"] == workload_set["sha256"]
    assert reference["route_table"]["source"] == (
        "verified_hardware/nvidia_geforce_rtx_4080/routes.json"
    )
    assert reference["route_table"]["sha256"] == hashlib.sha256(
        (bundle / "routes.json").read_bytes()
    ).hexdigest()
    assert reference["correctness"] == {
        "all_cases_passed": True,
        "passed_cases": 9,
        "case_count": 9,
        "total_trials": 45,
        "failed_elements": 0,
        "max_abs_error": 0.00390625,
        "all_routes_calibrated": True,
    }

    cases = {item["workload"]["case_id"]: item for item in reference["cases"]}
    assert list(cases) == [case.case_id for case in workload_set["cases"]]
    identity = verified.expected_runtime_identity(profile)
    for case in workload_set["cases"]:
        record = cases[case.case_id]
        expected_policy, expected_origin = verified._expected_route(
            routes,
            case.as_dict(),
            identity,
        )
        assert record["workload"] == case.as_dict()
        assert record["actual_policy"] == expected_policy
        assert record["route_origin"] == expected_origin == "calibrated"
        assert record["performance"]["speedup"] > 0

    group_speedups: list[float] = []
    for group in workload_set["groups"]:
        values = [cases[case_id]["performance"]["speedup"] for case_id in group.case_ids]
        geomean = math.exp(sum(math.log(value) for value in values) / len(values))
        assert reference["aggregate"]["group_geomean_speedups"][
            group.group_id
        ] == pytest.approx(geomean)
        group_speedups.append(geomean)
    overall = math.exp(
        sum(math.log(value) for value in group_speedups) / len(group_speedups)
    )
    assert reference["aggregate"][
        "group_balanced_geomean_speedup"
    ] == pytest.approx(overall)
