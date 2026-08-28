from __future__ import annotations

import pytest

from runner.contracts import RunVariant
from runner.performance_metrics import (
    LOGICAL_OPERATOR_TRAFFIC_SCOPE,
    derive_performance_metrics,
    derive_project_compute_efficiency,
)
from runner.verified_hardware import enrich_verified_summary_compute_efficiency
from tests.support.runner_fixtures import official_shape


def _anchors() -> dict[str, object]:
    return {
        "gemm_float16": {
            "available": True,
            "method": "saturated_square_torch_mm",
            "dtype": "float16",
            "tflops": 100.0,
        },
        "gemm_bfloat16": {
            "available": True,
            "method": "saturated_square_torch_mm",
            "dtype": "bfloat16",
            "tflops": 80.0,
        },
        "gemm_float32": {
            "available": True,
            "method": "saturated_square_torch_mm",
            "dtype": "float32",
            "tflops": 50.0,
        },
    }


def _case_result(*, achieved_tflops: float = 40.0) -> dict[str, object]:
    return {
        "case_id": "official_01",
        "useful_matmul_flops": 1_000_000_000_000,
        "attention_flops_fraction": 0.25,
        "achieved_tflops": achieved_tflops,
    }


def _execution_path() -> dict[str, object]:
    return {
        "linear_compute_dtype": "float32",
        "attention_compute_dtype": "float16",
    }


def test_workload_metrics_label_logical_traffic_as_an_estimate_not_dram() -> None:
    metrics = derive_performance_metrics(
        official_shape("official_01"),
        RunVariant(),
        {"median_ms": 2.0},
    )

    logical_bytes = metrics["estimated_logical_operator_bytes"]
    assert metrics["logical_operator_arithmetic_intensity_flops_per_byte"] == (
        pytest.approx(metrics["useful_matmul_flops"] / logical_bytes, abs=1e-6)
    )
    assert metrics["estimated_logical_operator_traffic_gbps"] == pytest.approx(
        logical_bytes / 2_000_000.0
    )
    assert metrics["logical_operator_traffic_scope"] == (LOGICAL_OPERATOR_TRAFFIC_SCOPE)
    assert "not_measured_dram" in metrics["logical_operator_traffic_scope"]


def test_project_mfu_uses_segmented_ideal_compute_time() -> None:
    metrics, reason = derive_project_compute_efficiency(
        _case_result(),
        _execution_path(),
        _anchors(),
    )

    assert reason is None
    assert metrics["measured_compute_roof_tflops"] == pytest.approx(400.0 / 7.0)
    assert metrics["project_estimated_mfu"] == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("case_result", "execution_path", "anchors", "expected_reason"),
    [
        (
            {"attention_flops_fraction": 0.25, "achieved_tflops": 40.0},
            _execution_path(),
            _anchors(),
            "missing_or_invalid_useful_matmul_flops",
        ),
        (
            _case_result(),
            {"attention_compute_dtype": "float16"},
            _anchors(),
            "missing_or_invalid_linear_compute_dtype",
        ),
        (
            _case_result(),
            _execution_path(),
            {"gemm_float16": _anchors()["gemm_float16"]},
            "missing_saturated_gemm_anchor_for_linear_float32",
        ),
    ],
)
def test_project_mfu_omits_metrics_when_inputs_are_unavailable(
    case_result: dict[str, object],
    execution_path: dict[str, object],
    anchors: dict[str, object],
    expected_reason: str,
) -> None:
    metrics, reason = derive_project_compute_efficiency(
        case_result,
        execution_path,
        anchors,
    )

    assert metrics == {}
    assert reason == expected_reason


def test_project_mfu_rejects_an_old_unsaturated_anchor() -> None:
    anchors = _anchors()
    anchor = anchors["gemm_float32"]
    assert isinstance(anchor, dict)
    anchor["method"] = "square_torch_mm"

    metrics, reason = derive_project_compute_efficiency(
        _case_result(),
        _execution_path(),
        anchors,
    )

    assert metrics == {}
    assert reason == "missing_saturated_gemm_anchor_for_linear_float32"


def test_project_mfu_above_noise_allowance_is_omitted_not_clamped() -> None:
    metrics, reason = derive_project_compute_efficiency(
        _case_result(achieved_tflops=61.0),
        _execution_path(),
        _anchors(),
    )

    assert metrics == {}
    assert reason == "project_estimated_mfu_exceeds_1.05"


def test_project_mfu_at_noise_allowance_is_published() -> None:
    metrics, reason = derive_project_compute_efficiency(
        _case_result(achieved_tflops=60.0),
        _execution_path(),
        _anchors(),
    )

    assert reason is None
    assert metrics["project_estimated_mfu"] == pytest.approx(1.05)


def test_verified_summary_enrichment_uses_profile_without_running_probe() -> None:
    profile = {"performance_anchors": _anchors()}
    runs = [
        {
            "workload": {"shape": {"case_id": "official_01"}},
            "execution_path": _execution_path(),
        }
    ]
    summary = {"case_results": [_case_result()]}

    enrich_verified_summary_compute_efficiency(profile, runs, summary)

    case_result = summary["case_results"][0]
    assert case_result["measured_compute_roof_tflops"] == pytest.approx(400.0 / 7.0)
    assert case_result["project_estimated_mfu"] == pytest.approx(0.7)
    assert "project_estimated_mfu_unavailable_reason" not in case_result
    definition = summary["metric_definition"]
    assert definition["id"] == "project_estimated_mfu_v1"
    assert "not an official score" in definition["project_estimated_mfu"]
    assert "omit instead of clamp" in definition["publication_rule"]


def test_verified_summary_records_why_project_mfu_was_not_published() -> None:
    runs = [
        {
            "workload": {"shape": {"case_id": "official_01"}},
            "execution_path": {"attention_compute_dtype": "float16"},
        }
    ]
    summary = {"case_results": [_case_result()]}

    enrich_verified_summary_compute_efficiency(
        {"performance_anchors": _anchors()},
        runs,
        summary,
    )

    case_result = summary["case_results"][0]
    assert "measured_compute_roof_tflops" not in case_result
    assert "project_estimated_mfu" not in case_result
    assert (
        case_result["project_estimated_mfu_unavailable_reason"]
        == "missing_or_invalid_linear_compute_dtype"
    )
