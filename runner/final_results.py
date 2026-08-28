"""Publish one concise performance view from verified benchmark references."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from runner.contracts import (
    ContractError,
    atomic_replace_json,
    load_json,
    utc_now,
)
from runner.performance_metrics import LOGICAL_OPERATOR_TRAFFIC_SCOPE

FINAL_PERFORMANCE_SCHEMA_VERSION = 1
FINAL_PERFORMANCE_ARTIFACT_KIND = "final_performance"

METRIC_DEFINITIONS = {
    "latency_ms": (
        "CUDA-event latency. Median and p90 describe the target; paired resident "
        "results also include the official baseline. Streamed host end-to-end time "
        "includes host scheduling around the full logical batch."
    ),
    "speedup": (
        "Official baseline median divided by target median. Published only for "
        "paired resident measurements."
    ),
    "achieved_tflops": (
        "Useful matrix-multiplication FLOPs divided by target median latency. FMA "
        "counts as two FLOPs; causal attention counts the lower triangle; "
        "normalization, activation, masking, and conversion FLOPs are excluded."
    ),
    "project_estimated_mfu": (
        "Segmented ideal compute time divided by measured target time, using "
        "dtype-matched saturated GEMM anchors. This is a project estimate, not an "
        "official competition score."
    ),
    "peak_device_allocated_bytes": (
        "Peak bytes reported by the PyTorch CUDA allocator for the deployed target "
        "path; this is not total process or physical VRAM usage."
    ),
    "logical_operator": (
        "Logical bytes and traffic at the current operator boundaries. The GB/s "
        "value is an analytical estimate, not measured DRAM bandwidth, and may "
        "therefore exceed the device's physical memory bandwidth."
    ),
    "validation_level": (
        "Resident cases use complete paired Formal validation. Batch-streamed "
        "cases retain their explicitly provisional target-only validation scope."
    ),
}


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{field} must be an object")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{field} must be a non-empty string")
    return value


def _positive_float(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ContractError(f"{field} must be a finite positive number")
    return float(value)


def _nonnegative_float(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ContractError(f"{field} must be a finite non-negative number")
    return float(value)


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{field} must be a non-negative integer")
    return value


def _timing(value: object, *, field: str) -> dict[str, float]:
    timing = _mapping(value, field=field)
    median = _positive_float(timing.get("median_ms"), field=f"{field}.median_ms")
    p90 = _positive_float(timing.get("p90_ms"), field=f"{field}.p90_ms")
    if p90 < median:
        raise ContractError(f"{field}.p90_ms cannot be lower than median_ms")
    return {"median": median, "p90": p90}


def _compact_protocol(value: object, *, field: str) -> dict[str, Any]:
    protocol = _mapping(value, field=field)
    compact: dict[str, Any] = {
        "preset": _string(protocol.get("preset"), field=f"{field}.preset")
    }
    for key in ("accuracy_trials", "warmup", "repeats", "rounds"):
        compact[key] = _positive_int(protocol.get(key), field=f"{field}.{key}")
    return compact


def _generation_identity(
    summary: Mapping[str, Any], hardware_id: str
) -> dict[str, Any]:
    bundle = _mapping(summary.get("bundle_identity"), field="bundle_identity")
    return {
        "hardware_identity": {
            "id": hardware_id,
            "environment": dict(
                _mapping(summary.get("environment"), field="environment")
            ),
            "profile_sha256": _string(
                bundle.get("profile_sha256"),
                field="bundle_identity.profile_sha256",
            ),
            "source_probe_run_id": _string(
                bundle.get("source_probe_run_id"),
                field="bundle_identity.source_probe_run_id",
            ),
        },
        "workload_identity": {
            "set_id": _string(summary.get("workload_set_id"), field="workload_set_id"),
            "sha256": _string(summary.get("workload_sha256"), field="workload_sha256"),
            "variant": dict(_mapping(summary.get("variant"), field="variant")),
        },
        "source": dict(_mapping(summary.get("source"), field="source")),
    }


def _scope_evidence(
    summary: Mapping[str, Any], *, require_route: bool
) -> dict[str, str]:
    bundle = _mapping(summary.get("bundle_identity"), field="bundle_identity")
    evidence = {
        "manifest_sha256": _string(
            bundle.get("manifest_sha256"),
            field="bundle_identity.manifest_sha256",
        ),
        "profile_sha256": _string(
            bundle.get("profile_sha256"),
            field="bundle_identity.profile_sha256",
        ),
        "source_probe_run_id": _string(
            bundle.get("source_probe_run_id"),
            field="bundle_identity.source_probe_run_id",
        ),
    }
    if require_route:
        dispatch = _mapping(summary.get("dispatch"), field="resident.dispatch")
        evidence["route_table_sha256"] = _string(
            dispatch.get("sha256"), field="resident.dispatch.sha256"
        )
    return evidence


def _compact_correctness(value: object, *, field: str) -> dict[str, Any]:
    correctness = _mapping(value, field=field)
    if correctness.get("passed") is not True:
        raise ContractError(f"{field}.passed must be true")
    failed_elements = _nonnegative_int(
        correctness.get("failed_elements"), field=f"{field}.failed_elements"
    )
    if failed_elements != 0:
        raise ContractError(f"{field}.failed_elements must be zero")
    compact: dict[str, Any] = {
        "passed": True,
        "trial_count": _positive_int(
            correctness.get("trial_count"), field=f"{field}.trial_count"
        ),
        "failed_elements": failed_elements,
        "max_abs_error": _nonnegative_float(
            correctness.get("max_abs_error"), field=f"{field}.max_abs_error"
        ),
        "max_relative_error": _nonnegative_float(
            correctness.get("max_relative_error"),
            field=f"{field}.max_relative_error",
        ),
    }
    compared = correctness.get("compared_elements")
    if compared is not None:
        compact["compared_elements"] = _positive_int(
            compared, field=f"{field}.compared_elements"
        )
    return compact


def _compact_performance(case: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    mfu = _positive_float(
        case.get("project_estimated_mfu"),
        field=f"{field}.project_estimated_mfu",
    )
    if mfu > 1.05:
        raise ContractError(f"{field}.project_estimated_mfu cannot exceed 1.05")
    scope = _string(
        case.get("logical_operator_traffic_scope"),
        field=f"{field}.logical_operator_traffic_scope",
    )
    if scope != LOGICAL_OPERATOR_TRAFFIC_SCOPE:
        raise ContractError(f"{field} has an unsupported logical traffic scope")
    return {
        "achieved_tflops": _positive_float(
            case.get("achieved_tflops"), field=f"{field}.achieved_tflops"
        ),
        "measured_compute_roof_tflops": _positive_float(
            case.get("measured_compute_roof_tflops"),
            field=f"{field}.measured_compute_roof_tflops",
        ),
        "project_estimated_mfu": mfu,
        "peak_device_allocated_bytes": _positive_int(
            case.get("peak_device_allocated_bytes"),
            field=f"{field}.peak_device_allocated_bytes",
        ),
        "logical_operator": {
            "estimated_bytes": _positive_int(
                case.get("estimated_logical_operator_bytes"),
                field=f"{field}.estimated_logical_operator_bytes",
            ),
            "arithmetic_intensity_flops_per_byte": _positive_float(
                case.get("logical_operator_arithmetic_intensity_flops_per_byte"),
                field=(f"{field}.logical_operator_arithmetic_intensity_flops_per_byte"),
            ),
            "estimated_traffic_gbps": _positive_float(
                case.get("estimated_logical_operator_traffic_gbps"),
                field=f"{field}.estimated_logical_operator_traffic_gbps",
            ),
            "scope": scope,
        },
    }


def _policy(case: Mapping[str, Any], *, field: str) -> str:
    selected = _string(case.get("selected_policy"), field=f"{field}.selected_policy")
    actual = _string(case.get("actual_policy"), field=f"{field}.actual_policy")
    if case.get("policy_applied") is not True or actual != selected:
        raise ContractError(f"{field} lacks matching observed policy evidence")
    return actual


def _case_list(summary: Mapping[str, Any], *, field: str) -> list[Mapping[str, Any]]:
    value = summary.get("case_results")
    if not isinstance(value, list) or not value:
        raise ContractError(f"{field}.case_results must be a non-empty list")
    cases = [
        _mapping(case, field=f"{field}.case_results[{index}]")
        for index, case in enumerate(value)
    ]
    case_ids = [
        _string(case.get("case_id"), field=f"{field}.case_results.case_id")
        for case in cases
    ]
    if len(case_ids) != len(set(case_ids)):
        raise ContractError(f"{field}.case_results contains duplicate case IDs")
    return cases


def _compact_resident(
    summary: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if summary.get("target") != "solution":
        raise ContractError("resident summary must measure the solution target")
    if summary.get("sweep_outcome") != "complete" or summary.get("failed_cases"):
        raise ContractError("resident summary must be a complete successful sweep")
    protocol = _compact_protocol(summary.get("protocol"), field="resident.protocol")
    if protocol["preset"] != "formal":
        raise ContractError("resident summary must use the Formal protocol")

    compact_cases: list[dict[str, Any]] = []
    for index, case in enumerate(_case_list(summary, field="resident")):
        field = f"resident.case_results[{index}]"
        if case.get("outcome") != "success":
            raise ContractError(f"{field}.outcome must be success")
        baseline = _timing(case.get("baseline"), field=f"{field}.baseline")
        target = _timing(case.get("solution"), field=f"{field}.solution")
        speedup = _positive_float(case.get("speedup"), field=f"{field}.speedup")
        expected_speedup = baseline["median"] / target["median"]
        if not math.isclose(speedup, expected_speedup, rel_tol=1e-9, abs_tol=1e-12):
            raise ContractError(f"{field}.speedup does not match median latency")
        compact_cases.append(
            {
                "case_id": _string(case.get("case_id"), field=f"{field}.case_id"),
                "scope": "resident",
                "execution_mode": "resident",
                "validation_level": "formal",
                "policy": _policy(case, field=field),
                "latency_ms": {
                    "baseline_median": baseline["median"],
                    "baseline_p90": baseline["p90"],
                    "target_median": target["median"],
                    "target_p90": target["p90"],
                },
                "speedup": speedup,
                "performance": _compact_performance(case, field=field),
                "correctness": _compact_correctness(
                    case.get("accuracy"), field=f"{field}.accuracy"
                ),
            }
        )

    scope = {
        "validation_level": "formal",
        "comparison_mode": "paired",
        "measured_at": _string(summary.get("created_at"), field="resident.created_at"),
        "protocol": protocol,
        "case_count": len(compact_cases),
        "geomean_speedup": _positive_float(
            summary.get("geomean_speedup"), field="resident.geomean_speedup"
        ),
        "evidence": _scope_evidence(summary, require_route=True),
    }
    return scope, compact_cases


def _compact_streamed(
    summary: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if summary.get("artifact_kind") != "verified_streamed_reference":
        raise ContractError("streamed summary has an unexpected artifact kind")
    if (
        summary.get("validation_level") != "provisional"
        or summary.get("comparison_mode") != "target_only"
    ):
        raise ContractError("streamed summary must remain provisional and target-only")
    protocol = _compact_protocol(summary.get("protocol"), field="streamed.protocol")
    if protocol["preset"] != "formal":
        raise ContractError("streamed summary must use the Formal protocol")

    compact_cases: list[dict[str, Any]] = []
    for index, case in enumerate(_case_list(summary, field="streamed")):
        field = f"streamed.case_results[{index}]"
        if case.get("outcome") != "success":
            raise ContractError(f"{field}.outcome must be success")
        if "baseline" in case or "speedup" in case:
            raise ContractError(f"{field} cannot claim paired baseline or speedup")
        target = _timing(case.get("solution"), field=f"{field}.solution")
        schedule = _mapping(case.get("schedule"), field=f"{field}.schedule")
        compact_cases.append(
            {
                "case_id": _string(case.get("case_id"), field=f"{field}.case_id"),
                "scope": "streamed",
                "execution_mode": "batch_streamed",
                "validation_level": "provisional",
                "policy": _policy(case, field=field),
                "latency_ms": {
                    "target_median": target["median"],
                    "target_p90": target["p90"],
                    "host_end_to_end": _positive_float(
                        case.get("end_to_end_ms"), field=f"{field}.end_to_end_ms"
                    ),
                },
                "performance": _compact_performance(case, field=field),
                "correctness": _compact_correctness(
                    case.get("accuracy"), field=f"{field}.accuracy"
                ),
                "schedule": {
                    "timing_microbatch_size": _positive_int(
                        schedule.get("timing_microbatch_size"),
                        field=f"{field}.schedule.timing_microbatch_size",
                    ),
                    "microbatch_count": _positive_int(
                        schedule.get("microbatch_count"),
                        field=f"{field}.schedule.microbatch_count",
                    ),
                    "reference_scope": _string(
                        schedule.get("reference_scope"),
                        field=f"{field}.schedule.reference_scope",
                    ),
                },
            }
        )

    scope = {
        "validation_level": "provisional",
        "comparison_mode": "target_only",
        "measured_at": _string(summary.get("created_at"), field="streamed.created_at"),
        "protocol": protocol,
        "case_count": len(compact_cases),
        "evidence": _scope_evidence(summary, require_route=False),
    }
    return scope, compact_cases


def _existing_generation(document: Mapping[str, Any]) -> dict[str, Any]:
    if (
        document.get("schema_version") != FINAL_PERFORMANCE_SCHEMA_VERSION
        or document.get("artifact_kind") != FINAL_PERFORMANCE_ARTIFACT_KIND
    ):
        raise ContractError("existing final performance artifact has an unknown schema")
    return {
        "hardware_identity": dict(
            _mapping(document.get("hardware_identity"), field="hardware_identity")
        ),
        "workload_identity": dict(
            _mapping(document.get("workload_identity"), field="workload_identity")
        ),
        "source": dict(_mapping(document.get("source"), field="source")),
    }


def _preserved_side(
    document: Mapping[str, Any], side: str
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    scopes = _mapping(document.get("scopes"), field="scopes")
    cases = document.get("cases")
    if not isinstance(cases, list):
        raise ContractError("existing final performance cases must be a list")
    invalid_scopes = [
        case.get("scope")
        for case in cases
        if not isinstance(case, Mapping)
        or case.get("scope") not in {"resident", "streamed"}
    ]
    if invalid_scopes:
        raise ContractError("existing final performance contains an invalid case scope")
    side_cases = [dict(case) for case in cases if case.get("scope") == side]
    raw_scope = scopes.get(side)
    if raw_scope is None:
        if side_cases:
            raise ContractError(f"existing {side} cases lack a scope summary")
        return None, []
    scope = dict(_mapping(raw_scope, field=f"scopes.{side}"))
    if scope.get("case_count") != len(side_cases):
        raise ContractError(f"existing {side} scope has an inconsistent case count")
    return scope, side_cases


def update_final_performance(
    path: Path,
    *,
    hardware_id: str,
    resident_summary: Mapping[str, Any] | None = None,
    streamed_summary: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically update one compact final artifact from verified summaries.

    An omitted side is preserved only when the existing artifact has the same
    hardware environment, workload, variant, and source identity as the new side.
    """

    hardware_id = _string(hardware_id, field="hardware_id")
    if resident_summary is None and streamed_summary is None:
        raise ContractError("at least one verified summary is required")

    generations: list[dict[str, Any]] = []
    if resident_summary is not None:
        resident_summary = _mapping(resident_summary, field="resident_summary")
        generations.append(_generation_identity(resident_summary, hardware_id))
    if streamed_summary is not None:
        streamed_summary = _mapping(streamed_summary, field="streamed_summary")
        generations.append(_generation_identity(streamed_summary, hardware_id))
    generation = generations[0]
    if any(candidate != generation for candidate in generations[1:]):
        raise ContractError(
            "resident and streamed summaries are not the same generation"
        )

    resident_scope: dict[str, Any] | None = None
    streamed_scope: dict[str, Any] | None = None
    resident_cases: list[dict[str, Any]] = []
    streamed_cases: list[dict[str, Any]] = []
    existing: Mapping[str, Any] | None = None
    if path.exists():
        existing = load_json(path)
        if _existing_generation(existing) != generation:
            existing = None

    if resident_summary is not None:
        resident_scope, resident_cases = _compact_resident(resident_summary)
    elif existing is not None:
        resident_scope, resident_cases = _preserved_side(existing, "resident")

    if streamed_summary is not None:
        streamed_scope, streamed_cases = _compact_streamed(streamed_summary)
    elif existing is not None:
        streamed_scope, streamed_cases = _preserved_side(existing, "streamed")

    case_ids = [case["case_id"] for case in resident_cases + streamed_cases]
    if len(case_ids) != len(set(case_ids)):
        raise ContractError("resident and streamed scopes contain overlapping case IDs")

    scopes: dict[str, Any] = {}
    if resident_scope is not None:
        scopes["resident"] = resident_scope
    if streamed_scope is not None:
        scopes["streamed"] = streamed_scope
    document = {
        "schema_version": FINAL_PERFORMANCE_SCHEMA_VERSION,
        "artifact_kind": FINAL_PERFORMANCE_ARTIFACT_KIND,
        "created_at": utc_now(),
        **generation,
        "scopes": scopes,
        "cases": resident_cases + streamed_cases,
        "definitions": dict(METRIC_DEFINITIONS),
    }
    atomic_replace_json(path, document)
    return path


__all__ = [
    "FINAL_PERFORMANCE_ARTIFACT_KIND",
    "FINAL_PERFORMANCE_SCHEMA_VERSION",
    "METRIC_DEFINITIONS",
    "update_final_performance",
]
