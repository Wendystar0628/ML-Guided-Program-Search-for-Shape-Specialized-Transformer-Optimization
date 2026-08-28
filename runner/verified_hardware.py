"""Run one verified hardware bundle through the shared benchmark runner."""

from __future__ import annotations

import argparse
import math
import platform
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from project_identity import canonical_json_sha256
from route_contracts import (
    RouteTable,
    VerifiedBundleManifest,
    load_verified_bundle,
    resolve_route_result,
)
from runner.candidates import candidate_spec_for_policy, exact_route_policy_ids
from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    RunVariant,
    TransformerShape,
    WorkloadSet,
    load_json,
    load_workload_set,
)
from runner.final_results import update_final_performance
from runner.locking import (
    bundle_lock_path,
    device_measurement_lease,
    exclusive_file_lock,
)
from runner.performance_metrics import (
    LOGICAL_OPERATOR_TRAFFIC_SCOPE,
    derive_project_compute_efficiency,
    project_mfu_metric_definition,
)
from runner.probe import collect_environment
from runner.result_contracts import (
    validate_benchmark_performance,
    validate_correctness,
    validate_workload_execution,
)
from runner.result_layout import final_performance_path, intermediate_results_dir
from runner.routing_contracts import (
    exact_route_key,
    hardware_identity_from_runtime,
    hardware_identity_from_verified_profile,
)
from runner.streamed_service import (
    StreamedBenchmarkRequest,
    StreamedBenchmarkResult,
    StreamedBenchmarkService,
)
from runner.supervisor import CancellationToken
from runner.sweep import (
    BenchmarkSweepRequest,
    BenchmarkSweepResult,
    BenchmarkSweepService,
)
from runner.workload_execution import (
    STREAMED_POLICY_SELECTOR,
    all_benchmark_shapes,
    route_eligible_shapes,
    streamed_benchmark_shapes,
)

_STABLE_HARDWARE_FIELDS = (
    "device_type",
    "device_name",
    "compute_capability",
)


class VerifiedHardwareError(RuntimeError):
    """Raised when a run cannot be attributed to this verified bundle."""


@dataclass(frozen=True)
class BundlePaths:
    """Resolved paths owned by one verified hardware bundle."""

    project_root: Path
    bundle_root: Path
    profile: Path
    routes: Path
    sweeps: Path

    @property
    def manifest(self) -> Path:
        return self.bundle_root / "manifest.json"

    @property
    def final_performance(self) -> Path:
        return final_performance_path(self.project_root, self.bundle_root.name)

    @classmethod
    def from_bundle(cls, bundle_root: Path) -> BundlePaths:
        bundle_root = bundle_root.resolve()
        project_root = bundle_root.parents[1]
        return cls(
            project_root=project_root,
            bundle_root=bundle_root,
            profile=bundle_root / "profile.json",
            routes=bundle_root / "routes.json",
            sweeps=(
                intermediate_results_dir(project_root, "sweeps")
                / "verified"
                / bundle_root.name
            ),
        )


@dataclass(frozen=True)
class LaunchConfig:
    """User-controlled settings forwarded to the shared runner."""

    device: str = "cuda:0"
    preset: str = "formal"
    timeout: float | None = None
    variant: RunVariant = field(default_factory=RunVariant)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def expected_hardware_identity(profile: Mapping[str, Any]) -> dict[str, str]:
    """Extract the stable GPU identity owned by one persisted Bundle."""

    try:
        route_identity = hardware_identity_from_verified_profile(
            profile
        ).as_route_fields()
    except (TypeError, ValueError) as exc:
        raise VerifiedHardwareError(str(exc)) from exc
    return {field: route_identity[field] for field in _STABLE_HARDWARE_FIELDS}


def collect_runtime_identity(
    device_name: str,
    *,
    matmul_precision: str,
    allow_tf32: bool,
) -> dict[str, Any]:
    """Collect exact route facts for the measurement request about to run."""

    try:
        import torch
    except Exception as exc:
        raise VerifiedHardwareError(f"PyTorch is unavailable: {exc}") from exc

    try:
        device = torch.device(device_name)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise VerifiedHardwareError(f"invalid device {device_name!r}: {exc}") from exc
    if device.type != "cuda" or not torch.cuda.is_available():
        raise VerifiedHardwareError("verified GPU bundles require CUDA")

    try:
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(index)
    except (AssertionError, RuntimeError, ValueError) as exc:
        raise VerifiedHardwareError(
            f"cannot inspect CUDA device {device_name!r}: {exc}"
        ) from exc
    environment = collect_environment(device)
    driver = environment.get("driver")

    return {
        "gpu": {
            "name": str(properties.name),
            "compute_capability": f"{properties.major}.{properties.minor}",
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "software": {
            "torch": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "driver": str(driver) if driver else "unavailable",
        },
        "runtime_policy": {
            "matmul_precision": matmul_precision,
            "allow_tf32": allow_tf32,
        },
    }


def validate_hardware_identity(
    expected: Mapping[str, str],
    actual_runtime: Mapping[str, Any],
) -> None:
    """Require the same GPU while leaving software matching to exact routes."""

    try:
        actual = hardware_identity_from_runtime(actual_runtime).as_route_fields()
    except (TypeError, ValueError) as exc:
        raise VerifiedHardwareError(str(exc)) from exc
    labels = {
        "device_type": "device_type",
        "device_name": "gpu.name",
        "compute_capability": "gpu.compute_capability",
    }
    mismatches = [
        f"{labels[field]}: expected {expected.get(field)!r}, got {actual[field]!r}"
        for field in _STABLE_HARDWARE_FIELDS
        if actual[field] != expected.get(field)
    ]
    if mismatches:
        raise VerifiedHardwareError(
            "GPU does not match the verified hardware profile: " + "; ".join(mismatches)
        )


def _portable_source(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _dispatch_source_matches(
    source: object,
    *,
    route_path: Path,
    project_root: Path,
) -> bool:
    if not isinstance(source, str) or not source:
        return False
    if source.replace("\\", "/") == _portable_source(route_path, project_root):
        return True
    candidate = Path(source)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve() == route_path.resolve()


def _route_key(
    shape: TransformerShape,
    variant: RunVariant,
    identity: Mapping[str, Any],
) -> dict[str, object]:
    try:
        return exact_route_key(
            shape,
            variant,
            hardware_identity_from_runtime(identity),
        )
    except (TypeError, ValueError) as exc:
        raise VerifiedHardwareError(str(exc)) from exc


def _expected_route(
    table: RouteTable,
    shape: TransformerShape,
    variant: RunVariant,
    identity: Mapping[str, Any],
) -> tuple[str, str]:
    resolution = resolve_route_result(table, _route_key(shape, variant, identity))
    return resolution.policy, resolution.origin


def _case_id(run: Mapping[str, Any]) -> str | None:
    workload = run.get("workload")
    shape = workload.get("shape") if isinstance(workload, Mapping) else None
    value = shape.get("case_id") if isinstance(shape, Mapping) else None
    return value if isinstance(value, str) and value else None


def enrich_verified_summary_compute_efficiency(
    profile: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    summary: dict[str, Any],
) -> None:
    """Attach project MFU only when persisted runs and Probe roofs support it."""

    anchors = profile.get("performance_anchors")
    performance_anchors = anchors if isinstance(anchors, Mapping) else {}
    indexed_runs: dict[str, Mapping[str, Any]] = {}
    for run in runs:
        case_id = _case_id(run)
        if case_id is None:
            continue
        if case_id in indexed_runs:
            raise VerifiedHardwareError(
                f"{case_id}: duplicate run cannot define verified compute efficiency"
            )
        indexed_runs[case_id] = run

    case_results = summary.get("case_results")
    if not isinstance(case_results, list):
        raise VerifiedHardwareError("verified summary is missing case results")
    for case_result in case_results:
        if not isinstance(case_result, dict):
            raise VerifiedHardwareError("verified summary has an invalid case result")
        case_result.pop("measured_compute_roof_tflops", None)
        case_result.pop("project_estimated_mfu", None)
        case_result.pop("project_estimated_mfu_unavailable_reason", None)
        case_id = case_result.get("case_id")
        run = indexed_runs.get(case_id) if isinstance(case_id, str) else None
        execution_path = run.get("execution_path") if run is not None else None
        metrics, reason = derive_project_compute_efficiency(
            case_result,
            execution_path if isinstance(execution_path, Mapping) else {},
            performance_anchors,
        )
        if metrics:
            case_result.update(metrics)
        else:
            case_result["project_estimated_mfu_unavailable_reason"] = (
                reason or "compute_efficiency_inputs_unavailable"
            )
    summary["metric_definition"] = project_mfu_metric_definition()


def _required_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VerifiedHardwareError(f"streamed result is missing {field}")
    return value


def _required_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise VerifiedHardwareError(f"streamed result is missing {field}")
    return value


def _required_positive_float(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise VerifiedHardwareError(f"streamed result has invalid {field}")
    return float(value)


def _required_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VerifiedHardwareError(f"streamed result has invalid {field}")
    return value


def _validate_probe_profile_runtime(
    profile: Mapping[str, Any],
    actual_runtime: Mapping[str, Any],
) -> None:
    """Require Probe anchors and the streamed run to describe one runtime."""

    try:
        expected = hardware_identity_from_verified_profile(profile).as_route_fields()
        actual = hardware_identity_from_runtime(actual_runtime).as_route_fields()
    except (TypeError, ValueError) as exc:
        raise VerifiedHardwareError(str(exc)) from exc
    mismatches = [
        field
        for field, expected_value in expected.items()
        if actual[field] != expected_value
    ]
    if mismatches:
        raise VerifiedHardwareError(
            "streamed reference runtime does not match its verified Probe anchors: "
            + ", ".join(sorted(mismatches))
        )


def _compact_streamed_case(
    run: Mapping[str, Any],
    *,
    workload_set: WorkloadSet,
    shape: TransformerShape,
    manifest: VerifiedBundleManifest,
) -> dict[str, Any]:
    case_id = shape.case_id
    if run.get("outcome") != "success":
        raise VerifiedHardwareError(
            f"{case_id}: streamed reference did not complete successfully"
        )
    if run.get("target") != "solution" or run.get("comparison_mode") != "target_only":
        raise VerifiedHardwareError(
            f"{case_id}: streamed reference must be target-only Solution timing"
        )

    workload = _required_mapping(run.get("workload"), field="workload")
    if (
        workload.get("set_id") != workload_set.workload_set_id
        or workload.get("sha256") != workload_set.sha256
        or workload.get("shape") != shape.as_dict()
        or workload.get("variant") != manifest.formal_variant
    ):
        raise VerifiedHardwareError(f"{case_id}: streamed workload identity mismatch")

    source = _required_mapping(run.get("source"), field="source")
    if source.get("official_sha256") != manifest.official_snapshot_sha256:
        raise VerifiedHardwareError(f"{case_id}: official source identity mismatch")
    if source.get("solution_sha256") != manifest.solution_implementation_sha256:
        raise VerifiedHardwareError(f"{case_id}: Solution source identity mismatch")

    protocol = _required_mapping(run.get("protocol"), field="protocol")
    performance, performance_error = validate_benchmark_performance(
        run.get("performance"),
        target="solution",
        comparison_mode="target_only",
        repeats=protocol.get("repeats"),
        rounds=protocol.get("rounds"),
        expected_timer="cuda_event",
    )
    if (
        performance_error is not None
        or performance is None
        or performance.target is None
    ):
        raise VerifiedHardwareError(
            f"{case_id}: invalid streamed performance: {performance_error}"
        )
    correctness_error = validate_correctness(
        run.get("correctness"),
        expected_trials=protocol.get("accuracy_trials"),
    )
    if correctness_error is not None:
        raise VerifiedHardwareError(
            f"{case_id}: invalid streamed correctness: {correctness_error}"
        )
    workload_execution_error = validate_workload_execution(
        run.get("workload_execution")
    )
    if workload_execution_error is not None:
        raise VerifiedHardwareError(
            f"{case_id}: invalid streamed schedule: {workload_execution_error}"
        )

    correctness = _required_mapping(run.get("correctness"), field="correctness")
    workload_execution = _required_mapping(
        run.get("workload_execution"), field="workload_execution"
    )
    execution_path = _required_mapping(
        run.get("execution_path"), field="execution_path"
    )
    if (
        correctness.get("validation_level") != "provisional"
        or workload_execution.get("validation_level") != "provisional"
        or workload_execution.get("mode") != "batch_streamed"
    ):
        raise VerifiedHardwareError(
            f"{case_id}: streamed reference must retain provisional validation"
        )
    if any(
        execution_path.get(field) is not None
        for field in (
            "dispatch_source",
            "dispatch_table_sha256",
            "dispatch_policy",
            "route_origin",
        )
    ):
        raise VerifiedHardwareError(
            f"{case_id}: streamed reference cannot claim an exact verified route"
        )

    selected_policy = _required_string(
        run.get("selected_policy"), field="selected_policy"
    )
    actual_policy = _required_string(run.get("actual_policy"), field="actual_policy")
    if (
        run.get("policy_applied") is not True
        or actual_policy != selected_policy
        or execution_path.get("requested_policy") != actual_policy
        or execution_path.get("selected_policy") != actual_policy
    ):
        raise VerifiedHardwareError(
            f"{case_id}: streamed policy lacks observed execution evidence"
        )

    raw_performance = _required_mapping(run.get("performance"), field="performance")
    target_timing = _required_mapping(
        raw_performance.get("target"), field="target timing"
    )
    selection = _required_mapping(
        workload_execution.get("selection"), field="streamed selection"
    )
    attention_fraction = raw_performance.get("attention_flops_fraction")
    if (
        isinstance(attention_fraction, bool)
        or not isinstance(attention_fraction, (int, float))
        or not math.isfinite(float(attention_fraction))
        or not 0.0 <= float(attention_fraction) <= 1.0
    ):
        raise VerifiedHardwareError(
            f"{case_id}: streamed result has invalid attention_flops_fraction"
        )
    logical_traffic_scope = _required_string(
        raw_performance.get("logical_operator_traffic_scope"),
        field="logical_operator_traffic_scope",
    )
    if logical_traffic_scope != LOGICAL_OPERATOR_TRAFFIC_SCOPE:
        raise VerifiedHardwareError(
            f"{case_id}: streamed result has unsupported logical traffic scope"
        )

    return {
        "case_id": case_id,
        "outcome": "success",
        "solution": {
            "median_ms": _required_positive_float(
                target_timing.get("median_ms"), field="target median_ms"
            ),
            "p90_ms": _required_positive_float(
                target_timing.get("p90_ms"), field="target p90_ms"
            ),
        },
        "end_to_end_ms": _required_positive_float(
            raw_performance.get("end_to_end_ms"), field="end_to_end_ms"
        ),
        "peak_device_allocated_bytes": _required_positive_int(
            raw_performance.get("peak_device_allocated_bytes"),
            field="peak_device_allocated_bytes",
        ),
        "useful_matmul_flops": _required_positive_int(
            raw_performance.get("useful_matmul_flops"),
            field="useful_matmul_flops",
        ),
        "attention_flops_fraction": float(attention_fraction),
        "achieved_tflops": _required_positive_float(
            raw_performance.get("achieved_tflops"), field="achieved_tflops"
        ),
        "estimated_logical_operator_bytes": _required_positive_int(
            raw_performance.get("estimated_logical_operator_bytes"),
            field="estimated_logical_operator_bytes",
        ),
        "logical_operator_arithmetic_intensity_flops_per_byte": (
            _required_positive_float(
                raw_performance.get(
                    "logical_operator_arithmetic_intensity_flops_per_byte"
                ),
                field="logical_operator_arithmetic_intensity_flops_per_byte",
            )
        ),
        "estimated_logical_operator_traffic_gbps": _required_positive_float(
            raw_performance.get("estimated_logical_operator_traffic_gbps"),
            field="estimated_logical_operator_traffic_gbps",
        ),
        "logical_operator_traffic_scope": logical_traffic_scope,
        "accuracy": {
            key: correctness[key]
            for key in (
                "passed",
                "trial_count",
                "failed_elements",
                "max_abs_error",
                "max_relative_error",
                "compared_elements",
            )
            if correctness.get(key) is not None
        },
        "selected_policy": selected_policy,
        "policy_applied": True,
        "actual_policy": actual_policy,
        "execution": {
            key: _required_string(execution_path.get(key), field=key)
            for key in (
                "attention_backend",
                "linear_backend",
                "attention_compute_dtype",
                "linear_compute_dtype",
                "residual_norm_backend",
            )
        },
        "schedule": {
            "timing_microbatch_size": _required_positive_int(
                workload_execution.get("timing_microbatch_size"),
                field="timing_microbatch_size",
            ),
            "microbatch_count": _required_positive_int(
                workload_execution.get("microbatch_count"),
                field="microbatch_count",
            ),
            "reference_kind": _required_string(
                workload_execution.get("reference_kind"), field="reference_kind"
            ),
            "reference_scope": _required_string(
                workload_execution.get("reference_scope"), field="reference_scope"
            ),
            "selection_method": _required_string(
                selection.get("method"), field="selection method"
            ),
            "selection_evidence_sha256": _required_string(
                selection.get("evidence_sha256"), field="selection evidence"
            ),
        },
    }


def build_streamed_reference_summary(
    profile: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    *,
    workload_set: WorkloadSet,
    manifest: VerifiedBundleManifest,
    manifest_sha256: str,
    profile_sha256: str,
) -> dict[str, Any]:
    """Build the verified provisional input for the unified final result."""

    expected_shapes = streamed_benchmark_shapes(
        workload_set.shapes,
        RunVariant.from_dict(manifest.formal_variant),
    )
    expected_case_ids = tuple(shape.case_id for shape in expected_shapes)
    if expected_case_ids != manifest.provisional_case_ids:
        raise VerifiedHardwareError(
            "streamed reference scope does not match manifest provisional_case_ids"
        )
    indexed_runs: dict[str, Mapping[str, Any]] = {}
    for run in runs:
        case_id = _case_id(run)
        if case_id is None or case_id in indexed_runs:
            raise VerifiedHardwareError(
                "streamed reference has missing or duplicate cases"
            )
        indexed_runs[case_id] = run
    if tuple(indexed_runs) != expected_case_ids:
        raise VerifiedHardwareError(
            "streamed reference results do not match the provisional workload scope"
        )

    case_results = [
        _compact_streamed_case(
            indexed_runs[shape.case_id],
            workload_set=workload_set,
            shape=shape,
            manifest=manifest,
        )
        for shape in expected_shapes
    ]
    first_run = runs[0]
    protocol = dict(_required_mapping(first_run.get("protocol"), field="protocol"))
    environment = dict(
        _required_mapping(first_run.get("environment"), field="environment")
    )
    source = dict(_required_mapping(first_run.get("source"), field="source"))
    for run in runs[1:]:
        if (
            run.get("protocol") != protocol
            or run.get("environment") != environment
            or run.get("source") != source
        ):
            raise VerifiedHardwareError(
                "streamed reference cases do not share one measurement context"
            )

    source_probe = _required_mapping(profile.get("source_probe"), field="source Probe")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "verified_streamed_reference",
        "created_at": _required_string(first_run.get("created_at"), field="created_at"),
        "validation_level": "provisional",
        "comparison_mode": "target_only",
        "workload_set_id": workload_set.workload_set_id,
        "workload_sha256": workload_set.sha256,
        "variant": dict(manifest.formal_variant),
        "source": source,
        "protocol": protocol,
        "environment": environment,
        "bundle_identity": {
            "manifest_sha256": manifest_sha256,
            "profile_sha256": profile_sha256,
            "source_probe_run_id": _required_string(
                source_probe.get("run_id"), field="source Probe run_id"
            ),
        },
        "case_results": case_results,
    }
    enrich_verified_summary_compute_efficiency(profile, runs, summary)
    for case_result in case_results:
        if "project_estimated_mfu" not in case_result:
            reason = case_result.get("project_estimated_mfu_unavailable_reason")
            raise VerifiedHardwareError(
                f"{case_result['case_id']}: cannot publish streamed MFU: {reason}"
            )
    return summary


def validate_checked_bundle_scope(
    manifest: VerifiedBundleManifest,
    workload_set: WorkloadSet,
    table: RouteTable,
    variant: RunVariant,
) -> None:
    """Require an exact verified/provisional partition of the workload set."""

    all_shapes = all_benchmark_shapes(workload_set.shapes)
    covered_shapes = route_eligible_shapes(all_shapes, variant)
    expected_covered = tuple(shape.case_id for shape in covered_shapes)
    expected_covered_set = frozenset(expected_covered)
    expected_provisional = tuple(
        shape.case_id
        for shape in all_shapes
        if shape.case_id not in expected_covered_set
    )
    if frozenset(manifest.covered_case_ids) != expected_covered_set:
        raise VerifiedHardwareError(
            "verified bundle covered_case_ids do not match the local benchmark scope"
        )
    if frozenset(manifest.provisional_case_ids) != frozenset(expected_provisional):
        raise VerifiedHardwareError(
            "verified bundle provisional_case_ids do not match the streamed scope"
        )
    if manifest.excluded_case_ids:
        raise VerifiedHardwareError("verified bundle excludes executable workloads")
    if manifest.formal_variant != variant.as_dict():
        raise VerifiedHardwareError(
            "verified bundle Formal variant does not match this launch"
        )
    if len(table.routes) != len(covered_shapes):
        raise VerifiedHardwareError(
            "verified bundle must contain one exact route for each of the "
            f"{len(covered_shapes)} covered cases"
        )
    invalid_policies = {
        policy
        for policy in (
            table.default_policy,
            *(policy for _match, policy in table.routes),
        )
        if policy not in exact_route_policy_ids()
    }
    if invalid_policies:
        raise VerifiedHardwareError(
            "verified bundle contains policies that are not eligible for resident "
            "exact routes: " + ", ".join(sorted(invalid_policies))
        )


def validate_workload_route_coverage(
    workload_set: WorkloadSet,
    *,
    variant: RunVariant,
    table: RouteTable,
    identity: Mapping[str, Any],
) -> None:
    """Require an explicit verified decision for every package workload."""

    try:
        route_identity = hardware_identity_from_runtime(identity)
    except (TypeError, ValueError) as exc:
        raise VerifiedHardwareError(
            f"invalid verified runtime identity: {exc}"
        ) from exc
    expected_identity = route_identity.as_route_fields()
    for index, (match, _policy) in enumerate(table.routes):
        mismatches = [
            field
            for field, expected in expected_identity.items()
            if match.get(field) != expected
        ]
        if mismatches:
            raise VerifiedHardwareError(
                f"verified routes[{index}] has a mismatched runtime identity: "
                + ", ".join(sorted(mismatches))
            )

    missing: list[str] = []
    for shape in route_eligible_shapes(workload_set.shapes, variant):
        shape_document = shape.as_dict()
        _, origin = _expected_route(table, shape, variant, identity)
        if origin != "calibrated":
            missing.append(str(shape_document.get("case_id", "<unknown>")))
    if missing:
        raise VerifiedHardwareError(
            "verified route table has no exact decision for: " + ", ".join(missing)
        )


def validate_run_routes(
    runs: Sequence[Mapping[str, Any]],
    *,
    table: RouteTable,
    identity: Mapping[str, Any],
    route_path: Path,
    route_sha256: str,
    project_root: Path,
    manifest: VerifiedBundleManifest,
) -> None:
    """Verify every measured result against one covered exact route."""

    covered = frozenset(manifest.covered_case_ids)
    for run in runs:
        case_id = _case_id(run) or "<missing-case-id>"
        workload = run.get("workload")
        shape_payload = workload.get("shape") if isinstance(workload, Mapping) else None
        variant_payload = (
            workload.get("variant") if isinstance(workload, Mapping) else None
        )
        execution_path = run.get("execution_path")
        if (
            not isinstance(shape_payload, dict)
            or not isinstance(variant_payload, dict)
            or not isinstance(execution_path, Mapping)
        ):
            raise VerifiedHardwareError(f"{case_id}: result is missing route details")
        try:
            shape = TransformerShape.from_dict(shape_payload)
            variant = RunVariant.from_dict(variant_payload)
        except ContractError as exc:
            raise VerifiedHardwareError(
                f"{case_id}: result has an invalid shape variant: {exc}"
            ) from exc
        if case_id not in covered:
            raise VerifiedHardwareError(
                f"{case_id}: result is outside the bundle's verified route scope"
            )
        if not _dispatch_source_matches(
            execution_path.get("dispatch_source"),
            route_path=route_path,
            project_root=project_root,
        ):
            raise VerifiedHardwareError(
                f"{case_id}: result used an unexpected dispatch source"
            )
        if execution_path.get("dispatch_table_sha256") != route_sha256:
            raise VerifiedHardwareError(
                f"{case_id}: result used an unexpected route-table hash"
            )

        expected_policy, expected_origin = _expected_route(
            table,
            shape,
            variant,
            identity,
        )
        if expected_origin != "calibrated":
            raise VerifiedHardwareError(
                f"{case_id}: verified workload resolved through fallback"
            )
        if execution_path.get("dispatch_policy") != expected_policy:
            raise VerifiedHardwareError(
                f"{case_id}: expected dispatch policy {expected_policy!r}, "
                f"got {execution_path.get('dispatch_policy')!r}"
            )
        if execution_path.get("route_origin") != expected_origin:
            raise VerifiedHardwareError(
                f"{case_id}: expected route origin {expected_origin!r}, "
                f"got {execution_path.get('route_origin')!r}"
            )
        try:
            candidate = candidate_spec_for_policy(
                shape,
                variant,
                expected_policy,
                deployable_only=True,
            )
        except (ContractError, RuntimeError, TypeError, ValueError) as exc:
            raise VerifiedHardwareError(
                f"{case_id}: cannot resolve execution evidence for "
                f"{expected_policy!r}: {exc}"
            ) from exc
        if candidate is None:
            raise VerifiedHardwareError(
                f"{case_id}: dispatch policy {expected_policy!r} has no deployable "
                "candidate for this workload"
            )
        if not candidate.exact_route_eligible:
            raise VerifiedHardwareError(
                f"{case_id}: dispatch policy {expected_policy!r} is not eligible "
                "for a resident exact route"
            )
        if not candidate.dispatch_evidence_matches(execution_path):
            raise VerifiedHardwareError(
                f"{case_id}: dispatch selected {expected_policy!r}, but the reported "
                "execution path does not prove that policy ran without fallback"
            )


def run_verified(
    config: LaunchConfig,
    *,
    paths: BundlePaths,
    identity_collector: Callable[..., dict[str, Any]] = collect_runtime_identity,
    sweep_service: BenchmarkSweepService | None = None,
    cancellation_token: CancellationToken | None = None,
) -> Path:
    """Validate and measure one bundle while exclusively owning its GPU."""

    with device_measurement_lease(
        paths.project_root,
        config.device,
        purpose="verified hardware sweep",
    ):
        return _run_verified(
            config,
            paths=paths,
            identity_collector=identity_collector,
            sweep_service=sweep_service,
            cancellation_token=cancellation_token,
        )


def run_verified_streamed(
    config: LaunchConfig,
    *,
    paths: BundlePaths,
    identity_collector: Callable[..., dict[str, Any]] = collect_runtime_identity,
    streamed_service: StreamedBenchmarkService | None = None,
) -> Path:
    """Measure one streamed Bundle scope while exclusively owning its GPU."""

    with device_measurement_lease(
        paths.project_root,
        config.device,
        purpose="verified streamed hardware measurement",
    ):
        return _run_verified_streamed(
            config,
            paths=paths,
            identity_collector=identity_collector,
            streamed_service=streamed_service,
        )


def _run_verified_streamed(
    config: LaunchConfig,
    *,
    paths: BundlePaths,
    identity_collector: Callable[..., dict[str, Any]] = collect_runtime_identity,
    streamed_service: StreamedBenchmarkService | None = None,
) -> Path:
    """Measure and publish only the Bundle's provisional streamed scope."""

    try:
        with exclusive_file_lock(
            bundle_lock_path(paths.bundle_root),
            purpose=f"streamed verification snapshot for {paths.bundle_root.name}",
        ):
            profile = load_json(paths.profile)
            table, route_sha256, manifest = load_verified_bundle(
                paths.routes,
                project_root=paths.project_root,
            )
            manifest_sha256 = canonical_json_sha256(paths.manifest)
            profile_sha256 = canonical_json_sha256(paths.profile)
    except (OSError, TypeError, ValueError) as exc:
        raise VerifiedHardwareError(
            f"verified bundle provenance is stale or invalid: {exc}"
        ) from exc
    protocol = MeasurementProtocol.for_preset(
        config.preset,
        matmul_precision="high",
        allow_tf32=True,
        timeout_seconds=config.timeout,
    )
    actual_identity = identity_collector(
        config.device,
        matmul_precision=protocol.matmul_precision,
        allow_tf32=protocol.allow_tf32,
    )
    validate_hardware_identity(expected_hardware_identity(profile), actual_identity)
    _validate_probe_profile_runtime(profile, actual_identity)

    workload_set = load_workload_set(
        paths.project_root,
        manifest.workload_set_id,
    )
    validate_checked_bundle_scope(manifest, workload_set, table, config.variant)
    completed: StreamedBenchmarkResult = (
        streamed_service or StreamedBenchmarkService()
    ).run(
        StreamedBenchmarkRequest(
            project_root=paths.project_root,
            workload_set_id=manifest.workload_set_id,
            protocol=protocol,
            device=config.device,
            variant=config.variant,
            solution_policy=STREAMED_POLICY_SELECTOR,
            case_ids=manifest.provisional_case_ids,
        )
    )
    if not completed.runs or len(completed.runs) != len(completed.result_paths):
        raise VerifiedHardwareError("streamed verifier returned no complete result set")
    if any(run.get("outcome") == "cancelled" for run in completed.runs):
        return completed.result_paths[-1]
    if any(run.get("outcome") != "success" for run in completed.runs):
        failures = ", ".join(
            f"{_case_id(run) or '<unknown>'}={run.get('outcome')}"
            for run in completed.runs
            if run.get("outcome") != "success"
        )
        raise VerifiedHardwareError(f"streamed verifier failed: {failures}")

    summary = build_streamed_reference_summary(
        profile,
        completed.runs,
        workload_set=workload_set,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        profile_sha256=profile_sha256,
    )
    if config.preset == "formal":
        try:
            with exclusive_file_lock(
                bundle_lock_path(paths.bundle_root),
                purpose=(
                    f"streamed reference publication for {paths.bundle_root.name}"
                ),
            ):
                _table, current_route_sha256, _manifest = load_verified_bundle(
                    paths.routes,
                    project_root=paths.project_root,
                )
                current_manifest_sha256 = canonical_json_sha256(paths.manifest)
                current_profile_sha256 = canonical_json_sha256(paths.profile)
                if (
                    current_route_sha256 != route_sha256
                    or current_manifest_sha256 != manifest_sha256
                    or current_profile_sha256 != profile_sha256
                ):
                    raise VerifiedHardwareError(
                        "verified bundle changed during streamed measurement; "
                        "refusing to publish a mixed-generation reference"
                    )
                update_final_performance(
                    paths.final_performance,
                    hardware_id=paths.bundle_root.name,
                    streamed_summary=summary,
                )
        except VerifiedHardwareError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise VerifiedHardwareError(
                f"cannot revalidate the verified bundle before publication: {exc}"
            ) from exc
        return paths.final_performance
    return completed.result_paths[-1]


def _run_verified(
    config: LaunchConfig,
    *,
    paths: BundlePaths,
    identity_collector: Callable[..., dict[str, Any]] = collect_runtime_identity,
    sweep_service: BenchmarkSweepService | None = None,
    cancellation_token: CancellationToken | None = None,
) -> Path:
    """Validate, execute, attribute, and summarize one verified-device sweep."""

    profile = load_json(paths.profile)
    try:
        table, route_sha256, manifest = load_verified_bundle(
            paths.routes,
            project_root=paths.project_root,
        )
        manifest_sha256 = canonical_json_sha256(paths.manifest)
        profile_sha256 = canonical_json_sha256(paths.profile)
    except (OSError, TypeError, ValueError) as exc:
        raise VerifiedHardwareError(
            f"verified bundle provenance is stale or invalid: {exc}"
        ) from exc
    protocol = MeasurementProtocol.for_preset(
        config.preset,
        matmul_precision="high",
        allow_tf32=True,
        timeout_seconds=config.timeout,
    )
    expected_identity = expected_hardware_identity(profile)
    actual_identity = identity_collector(
        config.device,
        matmul_precision=protocol.matmul_precision,
        allow_tf32=protocol.allow_tf32,
    )
    validate_hardware_identity(expected_identity, actual_identity)

    workload_set = load_workload_set(
        paths.project_root,
        manifest.workload_set_id,
    )
    validate_checked_bundle_scope(manifest, workload_set, table, config.variant)
    validate_workload_route_coverage(
        workload_set,
        variant=config.variant,
        table=table,
        identity=actual_identity,
    )

    def validate_before_persist(
        measured_workload: WorkloadSet,
        runs: Sequence[Mapping[str, Any]],
        _run_paths: Sequence[Path],
        summary: dict[str, Any],
    ) -> None:
        if measured_workload.sha256 != workload_set.sha256:
            raise VerifiedHardwareError("verified workload changed during the sweep")
        if summary.get("sweep_outcome") != "complete":
            failures = summary.get("failed_cases")
            if isinstance(failures, Sequence):
                detail = ", ".join(
                    f"{item.get('case_id')}={item.get('outcome')}"
                    for item in failures
                    if isinstance(item, Mapping)
                )
            else:
                detail = "unknown failure"
            raise VerifiedHardwareError(f"verified sweep is incomplete: {detail}")
        validate_run_routes(
            runs,
            table=table,
            identity=actual_identity,
            route_path=paths.routes,
            route_sha256=route_sha256,
            project_root=paths.project_root,
            manifest=manifest,
        )
        applied_case_ids = {_case_id(run) for run in runs}
        expected_case_ids = set(manifest.covered_case_ids)
        if applied_case_ids != expected_case_ids:
            raise VerifiedHardwareError(
                "verified sweep does not match the bundle workload partition"
            )
        case_results = summary.get("case_results")
        if not isinstance(case_results, list):
            raise VerifiedHardwareError("verified summary is missing case results")
        for case_result in case_results:
            if (
                not isinstance(case_result, dict)
                or case_result.get("case_id") not in applied_case_ids
            ):
                raise VerifiedHardwareError(
                    "verified summary does not match the validated runs"
                )
            if not isinstance(case_result.get("actual_policy"), str):
                raise VerifiedHardwareError(
                    f"{case_result.get('case_id')}: verified summary has no actual policy"
                )
        enrich_verified_summary_compute_efficiency(profile, runs, summary)
        source_probe = _required_mapping(
            profile.get("source_probe"), field="source Probe"
        )
        summary["bundle_identity"] = {
            "manifest_sha256": manifest_sha256,
            "profile_sha256": profile_sha256,
            "source_probe_run_id": _required_string(
                source_probe.get("run_id"), field="source Probe run_id"
            ),
        }

    service = sweep_service or BenchmarkSweepService()
    try:
        sweep: BenchmarkSweepResult = service.run(
            BenchmarkSweepRequest(
                project_root=paths.project_root,
                workload_set_id=manifest.workload_set_id,
                protocol=protocol,
                device=config.device,
                variant=config.variant,
                target="solution",
                solution_policy="dispatch",
                output_root=paths.sweeps,
            ),
            validate_before_persist=validate_before_persist,
            cancellation_token=cancellation_token,
        )
    except VerifiedHardwareError:
        raise
    except (ContractError, OSError, TypeError, ValueError) as exc:
        raise VerifiedHardwareError(f"verified sweep failed: {exc}") from exc
    if sweep.summary.get("sweep_outcome") == "cancelled":
        return sweep.summary_path
    if config.preset != "formal":
        return sweep.summary_path
    try:
        with exclusive_file_lock(
            bundle_lock_path(paths.bundle_root),
            purpose=f"resident result publication for {paths.bundle_root.name}",
        ):
            _table, current_route_sha256, _manifest = load_verified_bundle(
                paths.routes,
                project_root=paths.project_root,
            )
            current_manifest_sha256 = canonical_json_sha256(paths.manifest)
            current_profile_sha256 = canonical_json_sha256(paths.profile)
            if (
                current_route_sha256 != route_sha256
                or current_manifest_sha256 != manifest_sha256
                or current_profile_sha256 != profile_sha256
            ):
                raise VerifiedHardwareError(
                    "verified bundle changed during resident measurement; "
                    "refusing to publish a mixed-generation result"
                )
            update_final_performance(
                paths.final_performance,
                hardware_id=paths.bundle_root.name,
                resident_summary=sweep.summary,
            )
    except VerifiedHardwareError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise VerifiedHardwareError(
            f"cannot revalidate the verified bundle before publication: {exc}"
        ) from exc
    return paths.final_performance


def build_parser(hardware_id: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run the exact verified routes for {hardware_id}"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preset", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--timeout", type=_positive_float)
    parser.add_argument(
        "--scope",
        choices=("resident", "streamed", "all"),
        default="resident",
        help="run exact resident routes, provisional streamed cases, or both",
    )
    return parser


def main_for_bundle(
    bundle_root: Path,
    argv: Sequence[str] | None = None,
) -> int:
    paths = BundlePaths.from_bundle(bundle_root)
    hardware_id = paths.bundle_root.name
    args = build_parser(hardware_id).parse_args(argv)
    config = LaunchConfig(
        device=args.device,
        preset=args.preset,
        timeout=args.timeout,
    )
    result_paths: list[Path] = []

    def record_result(result_path: Path) -> bool:
        result_paths.append(result_path)
        result = load_json(result_path)
        return (
            result.get("sweep_outcome") == "cancelled"
            or result.get("outcome") == "cancelled"
        )

    try:
        if args.scope in {"resident", "all"}:
            result_path = run_verified(config, paths=paths)
            if record_result(result_path):
                print(
                    f"verified {hardware_id} run cancelled: {result_path}",
                    file=sys.stderr,
                )
                return 130
        if args.scope in {"streamed", "all"}:
            result_path = run_verified_streamed(config, paths=paths)
            if record_result(result_path):
                print(
                    f"verified {hardware_id} run cancelled: {result_path}",
                    file=sys.stderr,
                )
                return 130
    except KeyboardInterrupt:
        print(f"verified {hardware_id} run cancelled", file=sys.stderr)
        return 130
    except (ContractError, OSError, VerifiedHardwareError) as exc:
        print(f"verified {hardware_id} run failed: {exc}", file=sys.stderr)
        return 1
    for result_path in result_paths:
        print(f"verified result: {result_path}")
    return 0
