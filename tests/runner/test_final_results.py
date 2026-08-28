from __future__ import annotations

import json
from pathlib import Path

from runner.final_results import update_final_performance
from runner.performance_metrics import LOGICAL_OPERATOR_TRAFFIC_SCOPE


def _identity(*, generation: str = "a") -> dict[str, object]:
    return {
        "workload_set_id": "official_transformer_v1",
        "workload_sha256": generation * 64,
        "variant": {"dtype": "float32", "causal": True},
        "source": {
            "official_sha256": "b" * 64,
            "solution_sha256": generation * 64,
        },
        "environment": {
            "device": "cuda:0",
            "gpu": "Fixture GPU",
            "compute_capability": "8.9",
        },
        "bundle_identity": {
            "manifest_sha256": "d" * 64,
            "profile_sha256": "e" * 64,
            "source_probe_run_id": "probe-fixture",
        },
        "created_at": "2026-08-28T12:00:00+00:00",
    }


def _metrics() -> dict[str, object]:
    return {
        "achieved_tflops": 40.0,
        "measured_compute_roof_tflops": 80.0,
        "project_estimated_mfu": 0.5,
        "peak_device_allocated_bytes": 1024,
        "estimated_logical_operator_bytes": 2_000_000,
        "logical_operator_arithmetic_intensity_flops_per_byte": 20.0,
        "estimated_logical_operator_traffic_gbps": 10.0,
        "logical_operator_traffic_scope": LOGICAL_OPERATOR_TRAFFIC_SCOPE,
    }


def _accuracy(*, streamed: bool = False) -> dict[str, object]:
    accuracy: dict[str, object] = {
        "passed": True,
        "trial_count": 1 if streamed else 5,
        "failed_elements": 0,
        "max_abs_error": 0.001,
        "max_relative_error": 0.1,
    }
    if streamed:
        accuracy["compared_elements"] = 1024
    return accuracy


def _resident_summary(*, generation: str = "a") -> dict[str, object]:
    return {
        **_identity(generation=generation),
        "target": "solution",
        "protocol": {
            "preset": "formal",
            "accuracy_trials": 5,
            "warmup": 20,
            "repeats": 100,
            "rounds": 3,
        },
        "sweep_outcome": "complete",
        "failed_cases": [],
        "dispatch": {"source": "verified/routes.json", "sha256": "f" * 64},
        "geomean_speedup": 4.0,
        "case_results": [
            {
                "case_id": "official_01",
                "outcome": "success",
                "baseline": {"median_ms": 4.0, "p90_ms": 4.5},
                "solution": {"median_ms": 1.0, "p90_ms": 1.1},
                "speedup": 4.0,
                "accuracy": _accuracy(),
                "selected_policy": "graph",
                "policy_applied": True,
                "actual_policy": "graph",
                **_metrics(),
            }
        ],
    }


def _streamed_summary(*, generation: str = "a") -> dict[str, object]:
    return {
        **_identity(generation=generation),
        "artifact_kind": "verified_streamed_reference",
        "validation_level": "provisional",
        "comparison_mode": "target_only",
        "protocol": {
            "preset": "formal",
            "accuracy_trials": 1,
            "warmup": 2,
            "repeats": 1,
            "rounds": 3,
        },
        "case_results": [
            {
                "case_id": "official_14",
                "outcome": "success",
                "solution": {"median_ms": 100.0, "p90_ms": 110.0},
                "end_to_end_ms": 120.0,
                "accuracy": _accuracy(streamed=True),
                "selected_policy": "mixed-fp16-core-cudnn",
                "policy_applied": True,
                "actual_policy": "mixed-fp16-core-cudnn",
                "schedule": {
                    "timing_microbatch_size": 2,
                    "microbatch_count": 16,
                    "reference_scope": "validation_microbatch",
                },
                **_metrics(),
            }
        ],
    }


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_resident_only_publishes_compact_paired_result(tmp_path: Path) -> None:
    path = tmp_path / "final_performance.json"

    update_final_performance(
        path,
        hardware_id="fixture_gpu",
        resident_summary=_resident_summary(),
    )

    document = _read(path)
    assert set(document["scopes"]) == {"resident"}
    assert document["scopes"]["resident"]["geomean_speedup"] == 4.0
    case = document["cases"][0]
    assert case["latency_ms"] == {
        "baseline_median": 4.0,
        "baseline_p90": 4.5,
        "target_median": 1.0,
        "target_p90": 1.1,
    }
    assert case["speedup"] == 4.0
    assert case["performance"]["logical_operator"]["scope"] == (
        LOGICAL_OPERATOR_TRAFFIC_SCOPE
    )


def test_streamed_only_omits_fake_baseline_and_speedup(tmp_path: Path) -> None:
    path = tmp_path / "final_performance.json"

    update_final_performance(
        path,
        hardware_id="fixture_gpu",
        streamed_summary=_streamed_summary(),
    )

    document = _read(path)
    assert set(document["scopes"]) == {"streamed"}
    case = document["cases"][0]
    assert case["validation_level"] == "provisional"
    assert case["latency_ms"] == {
        "target_median": 100.0,
        "target_p90": 110.0,
        "host_end_to_end": 120.0,
    }
    assert "baseline" not in json.dumps(case)
    assert "speedup" not in case
    assert case["schedule"] == {
        "timing_microbatch_size": 2,
        "microbatch_count": 16,
        "reference_scope": "validation_microbatch",
    }


def test_same_generation_single_side_update_preserves_other_side(
    tmp_path: Path,
) -> None:
    path = tmp_path / "final_performance.json"
    update_final_performance(
        path,
        hardware_id="fixture_gpu",
        resident_summary=_resident_summary(),
    )

    update_final_performance(
        path,
        hardware_id="fixture_gpu",
        streamed_summary=_streamed_summary(),
    )

    document = _read(path)
    assert set(document["scopes"]) == {"resident", "streamed"}
    assert [case["case_id"] for case in document["cases"]] == [
        "official_01",
        "official_14",
    ]


def test_new_generation_drops_stale_unupdated_side(tmp_path: Path) -> None:
    path = tmp_path / "final_performance.json"
    update_final_performance(
        path,
        hardware_id="fixture_gpu",
        resident_summary=_resident_summary(generation="a"),
        streamed_summary=_streamed_summary(generation="a"),
    )

    update_final_performance(
        path,
        hardware_id="fixture_gpu",
        streamed_summary=_streamed_summary(generation="c"),
    )

    document = _read(path)
    assert document["workload_identity"]["sha256"] == "c" * 64
    assert set(document["scopes"]) == {"streamed"}
    assert [case["case_id"] for case in document["cases"]] == ["official_14"]
