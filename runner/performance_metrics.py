"""Derive comparable, hardware-independent metrics from measured latency."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from runner.contracts import ContractError, RunVariant, TransformerShape
from runner.hardware_router import analyze_workload

FLOPS_CONVENTION = (
    "useful_matmul_flops; FMA=2; causal QK/PV count the lower triangle; "
    "normalization, activation, masking, and conversion FLOPs are excluded"
)
LOGICAL_OPERATOR_TRAFFIC_SCOPE = (
    "current_operator_boundary_logical_estimate_not_measured_dram_traffic"
)

PROJECT_MFU_PUBLICATION_LIMIT = 1.05
_SUPPORTED_COMPUTE_DTYPES = frozenset({"float16", "bfloat16", "float32"})


def derive_performance_metrics(
    shape: TransformerShape,
    variant: RunVariant,
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    """Return useful FLOPs and achieved throughput for one full workload."""

    median = timing.get("median_ms")
    if (
        isinstance(median, bool)
        or not isinstance(median, (int, float))
        or not math.isfinite(float(median))
        or float(median) <= 0
    ):
        raise ContractError("target timing requires a finite positive median_ms")
    analysis = analyze_workload(shape, variant)
    achieved_tflops = analysis.total_flops / (float(median) * 1_000_000_000.0)
    logical_traffic_gbps = analysis.separate_operator_bytes / (
        float(median) * 1_000_000.0
    )
    return {
        "useful_matmul_flops": analysis.total_flops,
        "attention_matmul_flops": analysis.attention_flops,
        "projection_ffn_matmul_flops": analysis.projection_ffn_flops,
        "attention_flops_fraction": analysis.attention_fraction,
        "achieved_tflops": achieved_tflops,
        "flops_convention": FLOPS_CONVENTION,
        "estimated_logical_operator_bytes": analysis.separate_operator_bytes,
        "logical_operator_arithmetic_intensity_flops_per_byte": (
            analysis.separate_operator_arithmetic_intensity_flops_per_byte
        ),
        "estimated_logical_operator_traffic_gbps": logical_traffic_gbps,
        "logical_operator_traffic_scope": LOGICAL_OPERATOR_TRAFFIC_SCOPE,
    }


def _finite_positive(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def _gemm_roof_tflops(
    performance_anchors: Mapping[str, Any],
    dtype: str,
) -> float | None:
    anchor = performance_anchors.get(f"gemm_{dtype}")
    if not isinstance(anchor, Mapping):
        return None
    if (
        anchor.get("available") is not True
        or anchor.get("method") != "saturated_square_torch_mm"
        or anchor.get("dtype") != dtype
    ):
        return None
    return _finite_positive(anchor.get("tflops"))


def derive_project_compute_efficiency(
    case_result: Mapping[str, Any],
    execution_path: Mapping[str, Any],
    performance_anchors: Mapping[str, Any],
) -> tuple[dict[str, float], str | None]:
    """Estimate MFU from segmented FLOPs and measured saturated GEMM roofs.

    The function is deliberately independent of Probe execution and persistence.
    It returns no publishable metrics when any required input is unavailable or
    when the estimate exceeds the small measurement-noise allowance. Values are
    never clamped into the publishable range.
    """

    total_flops = _finite_positive(case_result.get("useful_matmul_flops"))
    if total_flops is None:
        return {}, "missing_or_invalid_useful_matmul_flops"
    attention_fraction_value = case_result.get("attention_flops_fraction")
    if isinstance(attention_fraction_value, bool) or not isinstance(
        attention_fraction_value, (int, float)
    ):
        return {}, "missing_or_invalid_attention_flops_fraction"
    attention_fraction = float(attention_fraction_value)
    if not math.isfinite(attention_fraction) or not 0.0 <= attention_fraction <= 1.0:
        return {}, "missing_or_invalid_attention_flops_fraction"
    achieved_tflops = _finite_positive(case_result.get("achieved_tflops"))
    if achieved_tflops is None:
        return {}, "missing_or_invalid_achieved_tflops"

    compute_dtypes: dict[str, str] = {}
    dtype_fields = {
        "linear": "linear_compute_dtype",
        "attention": "attention_compute_dtype",
    }
    for segment, field in dtype_fields.items():
        value = execution_path.get(field)
        if not isinstance(value, str) or value not in _SUPPORTED_COMPUTE_DTYPES:
            return {}, f"missing_or_invalid_{field}"
        compute_dtypes[segment] = value

    roofs: dict[str, float] = {}
    for segment, dtype in compute_dtypes.items():
        roof = _gemm_roof_tflops(performance_anchors, dtype)
        if roof is None:
            return {}, f"missing_saturated_gemm_anchor_for_{segment}_{dtype}"
        roofs[segment] = roof

    attention_flops = total_flops * attention_fraction
    projection_flops = total_flops - attention_flops
    ideal_seconds = math.fsum(
        (
            projection_flops / (roofs["linear"] * 1_000_000_000_000.0),
            attention_flops / (roofs["attention"] * 1_000_000_000_000.0),
        )
    )
    actual_seconds = total_flops / (achieved_tflops * 1_000_000_000_000.0)
    if (
        not math.isfinite(ideal_seconds)
        or ideal_seconds <= 0
        or not math.isfinite(actual_seconds)
        or actual_seconds <= 0
    ):
        return {}, "invalid_segmented_compute_time"

    measured_roof = total_flops / (ideal_seconds * 1_000_000_000_000.0)
    project_mfu = ideal_seconds / actual_seconds
    if not math.isfinite(measured_roof) or measured_roof <= 0:
        return {}, "invalid_measured_compute_roof"
    if not math.isfinite(project_mfu) or project_mfu <= 0:
        return {}, "invalid_project_estimated_mfu"
    if project_mfu > PROJECT_MFU_PUBLICATION_LIMIT:
        return {}, "project_estimated_mfu_exceeds_1.05"
    return {
        "measured_compute_roof_tflops": measured_roof,
        "project_estimated_mfu": project_mfu,
    }, None


def project_mfu_metric_definition() -> dict[str, str]:
    """Return the compact definition persisted with a verified summary."""

    return {
        "id": "project_estimated_mfu_v1",
        "measured_compute_roof_tflops": (
            "FLOP-weighted harmonic roof from saturated GEMM anchors matched "
            "to observed linear and attention compute dtypes"
        ),
        "project_estimated_mfu": (
            "segmented ideal compute time divided by measured Transformer "
            "target time; project estimate, not an official score"
        ),
        "publication_rule": (
            "omit instead of clamp when inputs or anchors are unavailable or "
            "the estimate exceeds 1.05"
        ),
    }


__all__ = [
    "FLOPS_CONVENTION",
    "LOGICAL_OPERATOR_TRAFFIC_SCOPE",
    "PROJECT_MFU_PUBLICATION_LIMIT",
    "derive_performance_metrics",
    "derive_project_compute_efficiency",
    "project_mfu_metric_definition",
]
