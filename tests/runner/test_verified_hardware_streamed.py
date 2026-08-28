from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import pytest

from route_contracts import (
    MANIFEST_SCHEMA_VERSION,
    RouteTable,
    validate_bundle_manifest,
)
from runner import verified_hardware
from runner.contracts import RunVariant, load_workload_set
from runner.streamed_service import StreamedBenchmarkResult
from runner.verified_hardware import (
    BundlePaths,
    LaunchConfig,
    VerifiedHardwareError,
    build_streamed_reference_summary,
    run_verified_streamed,
)
from tests.support.runner_fixtures import PROJECT_ROOT, WORKLOAD_SET_ID, official_shape

_COVERED_CASE_IDS = tuple(f"official_{index:02d}" for index in range(1, 14))


def _manifest(workload_sha256: str):
    return validate_bundle_manifest(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "workload_set": {
                "set_id": WORKLOAD_SET_ID,
                "sha256": workload_sha256,
            },
            "official": {"snapshot_sha256": "2" * 64},
            "solution": {"implementation_sha256": "3" * 64},
            "route_table": {"sha256": "4" * 64},
            "formal": {
                "protocol": {
                    "preset": "formal",
                    "compile_solution": False,
                    "matmul_precision": "high",
                    "allow_tf32": True,
                },
                "variant": RunVariant().as_dict(),
                "covered_case_ids": list(_COVERED_CASE_IDS),
                "provisional_case_ids": ["official_14"],
                "excluded_case_ids": [],
            },
        }
    )


def _profile() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_probe": {"run_id": "probe-saturated"},
        "device_operation_passed": True,
        "runtime_policy": {"matmul_precision": "high", "allow_tf32": True},
        "hardware_profile": {
            "device_type": "cuda",
            "platform": {"system": platform.system(), "machine": platform.machine()},
            "software": {
                "torch": "fixture-torch",
                "cuda_runtime": "fixture-cuda",
                "driver": "fixture-driver",
            },
            "gpu": {
                "name": "Fixture GPU",
                "compute_capability": "8.9",
            },
        },
        "performance_anchors": {
            "gemm_float16": {
                "available": True,
                "method": "saturated_square_torch_mm",
                "dtype": "float16",
                "tflops": 20.0,
            }
        },
    }


def _runtime_identity() -> dict[str, Any]:
    return {
        "gpu": {"name": "Fixture GPU", "compute_capability": "8.9"},
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "software": {
            "torch": "fixture-torch",
            "cuda_runtime": "fixture-cuda",
            "driver": "fixture-driver",
        },
        "runtime_policy": {"matmul_precision": "high", "allow_tf32": True},
    }


def _streamed_run(workload_sha256: str) -> dict[str, Any]:
    policy = "mixed-fp16-core-cudnn"
    selection_digest = "5" * 64
    return {
        "schema_version": 6,
        "run_id": "streamed-run",
        "created_at": "2026-08-28T00:00:00+00:00",
        "run_kind": "benchmark",
        "target": "solution",
        "comparison_mode": "target_only",
        "outcome": "success",
        "workload": {
            "set_id": WORKLOAD_SET_ID,
            "sha256": workload_sha256,
            "shape": official_shape("official_14").as_dict(),
            "variant": RunVariant().as_dict(),
        },
        "source": {
            "official_sha256": "2" * 64,
            "solution_sha256": "3" * 64,
        },
        "protocol": {
            "preset": "formal",
            "accuracy_trials": 1,
            "repeats": 1,
            "rounds": 3,
        },
        "environment": {
            "device": "cuda:0",
            "gpu": "Fixture GPU",
            "torch": "fixture-torch",
        },
        "correctness": {
            "passed": True,
            "trial_count": 1,
            "failed_elements": 0,
            "max_abs_error": 0.001,
            "max_relative_error": 0.1,
            "compared_elements": 1024,
            "reference_kind": "internal_query_block",
            "reference_scope": "validation_microbatch",
            "validation_level": "provisional",
        },
        "performance": {
            "comparison_mode": "target_only",
            "timer": "cuda_event",
            "sample_count": 3,
            "target": {"median_ms": 100.0, "p90_ms": 110.0},
            "peak_device_allocated_bytes": 1024,
            "end_to_end_ms": 120.0,
            "useful_matmul_flops": 1_000_000_000_000,
            "attention_flops_fraction": 0.9,
            "achieved_tflops": 10.0,
        },
        "execution_path": {
            "requested_policy": policy,
            "selected_policy": policy,
            "dispatch_source": None,
            "dispatch_table_sha256": None,
            "dispatch_policy": None,
            "route_origin": None,
            "attention_backend": "mixed_fp16_cudnn",
            "linear_backend": "autocast_fp16",
            "attention_compute_dtype": "float16",
            "linear_compute_dtype": "float16",
            "residual_norm_backend": "torch",
        },
        "selected_policy": policy,
        "policy_applied": True,
        "actual_policy": policy,
        "workload_execution": {
            "mode": "batch_streamed",
            "validation_microbatch_size": 1,
            "timing_microbatch_candidates": [1, 2],
            "timing_microbatch_size": 2,
            "microbatch_count": 16,
            "reference_kind": "internal_query_block",
            "reference_scope": "validation_microbatch",
            "validation_level": "provisional",
            "selection": {
                "method": "runtime_policy_and_microbatch_screen",
                "policy": policy,
                "timing_microbatch_size": 2,
                "microbatch_count": 16,
                "estimated_logical_batch_ms": 100.0,
                "evidence_sha256": selection_digest,
            },
            "candidate_screening": [
                {
                    "policy": policy,
                    "comparator_passed": True,
                    "timing_schedules": [
                        {
                            "timing_microbatch_size": 2,
                            "microbatch_count": 16,
                            "passed": True,
                        }
                    ],
                }
            ],
        },
    }


def test_streamed_reference_is_compact_provisional_and_uses_probe_mfu() -> None:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    summary = build_streamed_reference_summary(
        _profile(),
        [_streamed_run(workload.sha256)],
        workload_set=workload,
        manifest=_manifest(workload.sha256),
        manifest_sha256="6" * 64,
        profile_sha256="7" * 64,
    )

    case = summary["case_results"][0]
    assert summary["validation_level"] == "provisional"
    assert summary["comparison_mode"] == "target_only"
    assert summary["bundle_identity"] == {
        "manifest_sha256": "6" * 64,
        "profile_sha256": "7" * 64,
        "source_probe_run_id": "probe-saturated",
    }
    assert case["project_estimated_mfu"] == pytest.approx(0.5)
    assert case["measured_compute_roof_tflops"] == pytest.approx(20.0)
    assert case["schedule"]["timing_microbatch_size"] == 2
    assert "baseline" not in case
    assert "speedup" not in case
    assert "geomean_speedup" not in summary


def test_streamed_reference_rejects_source_or_exact_route_claims() -> None:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    run = _streamed_run(workload.sha256)
    run["source"]["solution_sha256"] = "9" * 64

    with pytest.raises(VerifiedHardwareError, match="Solution source identity"):
        build_streamed_reference_summary(
            _profile(),
            [run],
            workload_set=workload,
            manifest=_manifest(workload.sha256),
            manifest_sha256="6" * 64,
            profile_sha256="7" * 64,
        )

    run = _streamed_run(workload.sha256)
    run["execution_path"]["route_origin"] = "calibrated"
    with pytest.raises(VerifiedHardwareError, match="cannot claim an exact"):
        build_streamed_reference_summary(
            _profile(),
            [run],
            workload_set=workload,
            manifest=_manifest(workload.sha256),
            manifest_sha256="6" * 64,
            profile_sha256="7" * 64,
        )


def test_streamed_reference_requires_saturated_verified_probe_anchor() -> None:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    profile = _profile()
    profile["performance_anchors"]["gemm_float16"]["method"] = "square_torch_mm"

    with pytest.raises(VerifiedHardwareError, match="cannot publish streamed MFU"):
        build_streamed_reference_summary(
            profile,
            [_streamed_run(workload.sha256)],
            workload_set=workload,
            manifest=_manifest(workload.sha256),
            manifest_sha256="6" * 64,
            profile_sha256="7" * 64,
        )


@pytest.mark.parametrize("preset", ["smoke", "formal"])
def test_verified_streamed_only_formal_replaces_bundle_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    preset: str,
) -> None:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    manifest = _manifest(workload.sha256)
    run = _streamed_run(workload.sha256)
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(run), encoding="utf-8")
    paths = BundlePaths(
        project_root=PROJECT_ROOT,
        bundle_root=tmp_path,
        profile=tmp_path / "profile.json",
        routes=tmp_path / "routes.json",
        sweeps=tmp_path / "results" / "sweeps",
    )
    paths.reference_streamed.parent.mkdir(parents=True)
    paths.reference_streamed.write_text('{"sentinel": true}', encoding="utf-8")

    class FakeService:
        def run(self, request):
            assert request.case_ids == ("official_14",)
            assert request.solution_policy == "screen"
            return StreamedBenchmarkResult((run,), (raw_path,))

    monkeypatch.setattr(verified_hardware, "load_json", lambda _path: _profile())
    monkeypatch.setattr(
        verified_hardware,
        "load_verified_bundle",
        lambda *_args, **_kwargs: (
            RouteTable(
                default_policy="eager-sdpa",
                routes=tuple(({}, "eager-sdpa") for _ in _COVERED_CASE_IDS),
            ),
            "4" * 64,
            manifest,
        ),
    )
    monkeypatch.setattr(
        verified_hardware,
        "canonical_json_sha256",
        lambda path: "6" * 64 if Path(path).name == "manifest.json" else "7" * 64,
    )

    result_path = run_verified_streamed(
        LaunchConfig(preset=preset),
        paths=paths,
        identity_collector=lambda *_args, **_kwargs: _runtime_identity(),
        streamed_service=FakeService(),
    )

    if preset == "formal":
        assert result_path == paths.reference_streamed
        assert json.loads(result_path.read_text(encoding="utf-8"))["artifact_kind"] == (
            "verified_streamed_reference"
        )
    else:
        assert result_path == raw_path
        assert json.loads(paths.reference_streamed.read_text(encoding="utf-8")) == {
            "sentinel": True
        }


def test_verified_streamed_rejects_a_mixed_generation_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    manifest = _manifest(workload.sha256)
    run = _streamed_run(workload.sha256)
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(run), encoding="utf-8")
    paths = BundlePaths(
        project_root=PROJECT_ROOT,
        bundle_root=tmp_path,
        profile=tmp_path / "profile.json",
        routes=tmp_path / "routes.json",
        sweeps=tmp_path / "results" / "sweeps",
    )
    paths.reference_streamed.parent.mkdir(parents=True)
    paths.reference_streamed.write_text('{"sentinel": true}', encoding="utf-8")

    class FakeService:
        def run(self, _request):
            return StreamedBenchmarkResult((run,), (raw_path,))

    monkeypatch.setattr(verified_hardware, "load_json", lambda _path: _profile())
    monkeypatch.setattr(
        verified_hardware,
        "load_verified_bundle",
        lambda *_args, **_kwargs: (
            RouteTable(
                default_policy="eager-sdpa",
                routes=tuple(({}, "eager-sdpa") for _case_id in _COVERED_CASE_IDS),
            ),
            "4" * 64,
            manifest,
        ),
    )
    manifest_digest_calls = 0

    def changing_digest(path: Path) -> str:
        nonlocal manifest_digest_calls
        if Path(path).name == "manifest.json":
            manifest_digest_calls += 1
            return ("6" if manifest_digest_calls == 1 else "8") * 64
        return "7" * 64

    monkeypatch.setattr(
        verified_hardware,
        "canonical_json_sha256",
        changing_digest,
    )

    with pytest.raises(VerifiedHardwareError, match="mixed-generation"):
        run_verified_streamed(
            LaunchConfig(preset="formal"),
            paths=paths,
            identity_collector=lambda *_args, **_kwargs: _runtime_identity(),
            streamed_service=FakeService(),
        )

    assert json.loads(paths.reference_streamed.read_text(encoding="utf-8")) == {
        "sentinel": True
    }


def test_verified_launcher_defaults_to_resident_and_can_run_all(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "verified_hardware" / "fixture_gpu"
    bundle.mkdir(parents=True)
    resident_path = tmp_path / "resident.json"
    streamed_path = tmp_path / "streamed.json"
    resident_path.write_text('{"sweep_outcome": "complete"}', encoding="utf-8")
    streamed_path.write_text(
        '{"artifact_kind": "verified_streamed_reference"}', encoding="utf-8"
    )
    calls: list[str] = []
    monkeypatch.setattr(
        verified_hardware,
        "run_verified",
        lambda *_args, **_kwargs: calls.append("resident") or resident_path,
    )
    monkeypatch.setattr(
        verified_hardware,
        "run_verified_streamed",
        lambda *_args, **_kwargs: calls.append("streamed") or streamed_path,
    )

    assert verified_hardware.main_for_bundle(bundle, ["--preset", "smoke"]) == 0
    assert calls == ["resident"]

    calls.clear()
    assert (
        verified_hardware.main_for_bundle(
            bundle, ["--preset", "smoke", "--scope", "all"]
        )
        == 0
    )
    assert calls == ["resident", "streamed"]
