"""Coherent mechanism-family ablations for deployed resident programs."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch

from deployment.environment import EnvironmentFingerprint, ImplementationScope
from deployment.registry import resolve_deployed_config
from official import torch_transformer_benchmark as official
from solution.config import (
    AttentionBackend,
    AttentionOutputBridge,
    ConfigSpec,
    FFNBackend,
    InitialNormBackend,
    PrecisionPlan,
    ProjectionBackend,
    QKVMaterialization,
    ResidualNormBackend,
    RuntimeBackend,
    ScheduleConfig,
    TritonNormParams,
)
from solution.plan import ExecutionContext
from solution.plan_builder import HardwareCapabilities, PlanBuilder

from .config_resolution import shape_fingerprint
from .device_isolation import IsolatedProcessError, run_in_fresh_process
from .measure import measure_paired_configs
from .protocols import (
    MeasurementProtocol,
    RunVariant,
    TransformerShape,
    load_shape,
    write_json,
)

DEFAULT_ABLATION_SHAPES = tuple(
    f"official_{index:02d}" for index in range(1, 14)
)


class AblationFamily(StrEnum):
    """Mechanism families that admit interpretable legal counterfactuals."""

    RUNTIME = "runtime_schedule"
    ATTENTION = "attention_path"
    LAYOUT = "layout_path"
    PROJECTION = "projection_precision"
    FFN = "ffn_path"
    NORM = "norm_boundary"


ABLATION_FAMILIES = tuple(AblationFamily)

FAMILY_LABELS = {
    AblationFamily.RUNTIME: "Runtime schedule",
    AblationFamily.ATTENTION: "Attention path",
    AblationFamily.LAYOUT: "Layout path",
    AblationFamily.PROJECTION: "Projection precision",
    AblationFamily.FFN: "FFN path",
    AblationFamily.NORM: "Norm / boundary",
}


@dataclass(frozen=True, slots=True)
class AblationCandidate:
    """One legal family-level knockout and its interpretation boundary."""

    family: AblationFamily
    config: ConfigSpec
    variant_kind: str
    note: str
    changed_fields: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class AblationSuiteResult:
    path: Path
    summary: dict[str, Any]
    exit_code: int


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_ablation_run_directory(root: Path) -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return root / run_id


def ablation_protocol(case_id: str) -> MeasurementProtocol:
    """Use paired descriptive evidence without deployment-grade repetition."""

    return MeasurementProtocol(
        accuracy_trials=1,
        warmup=2,
        repeats=5,
        rounds=3 if case_id == "official_06" else 5,
    )


def _changes(full: ConfigSpec, ablated: ConfigSpec) -> tuple[dict[str, Any], ...]:
    changes: list[dict[str, Any]] = []

    def compare(on: object, off: object, prefix: str) -> None:
        if isinstance(on, dict) and isinstance(off, dict):
            for key in sorted(set(on) | set(off)):
                path = f"{prefix}.{key}" if prefix else str(key)
                compare(on.get(key), off.get(key), path)
            return
        if on != off:
            changes.append({"field": prefix, "on": on, "off": off})

    compare(full.to_dict(), ablated.to_dict(), "")
    return tuple(changes)


def _candidate(
    full: ConfigSpec,
    family: AblationFamily,
    program: object,
    schedule: ScheduleConfig,
    *,
    variant_kind: str = "atomic",
    note: str,
) -> AblationCandidate:
    ablated = ConfigSpec(program=program, schedule=schedule)
    changed_fields = _changes(full, ablated)
    if not changed_fields:
        raise ValueError(f"{family.value} produced a no-op candidate")
    return AblationCandidate(
        family=family,
        config=ablated,
        variant_kind=variant_kind,
        note=note,
        changed_fields=changed_fields,
    )


def build_ablation_candidate(
    full: ConfigSpec,
    family: AblationFamily,
    shape: TransformerShape,
) -> AblationCandidate | None:
    """Build the nearest legal family fallback without mutating ``full``."""

    family = AblationFamily(family)
    program = full.program
    schedule = full.schedule

    if family is AblationFamily.RUNTIME:
        if schedule.runtime is RuntimeBackend.EAGER:
            return None
        if shape.case_id == "official_06":
            return None
        kind = "atomic"
        note = "Outer capture or compilation is replaced by eager execution."
        if program.initial_norm is InitialNormBackend.TRITON_FUSED_QKV:
            program = replace(program, initial_norm=InitialNormBackend.TORCH)
            schedule = replace(schedule, initial_norm_launch=None)
            kind = "dependency_closure"
            note += " The fused initial-norm/QKV primitive also requires CUDA Graph."
        schedule = replace(
            schedule,
            runtime=RuntimeBackend.EAGER,
            compile_mode=None,
            batch_tile_size=None,
            microbatch_size=None,
            reuse_unchanged_input=False,
        )
        return _candidate(
            full,
            family,
            program,
            schedule,
            variant_kind=kind,
            note=note,
        )

    if family is AblationFamily.ATTENTION:
        if program.attention in {
            AttentionBackend.CAUSAL_SDPA,
            AttentionBackend.REFERENCE_STREAMING,
        }:
            return None
        kind = "atomic"
        note = "The deployed attention backend is replaced by generic causal SDPA."
        bridge = program.attention_output_bridge
        if bridge is AttentionOutputBridge.ATTENTION_DIRECT_BSD:
            bridge = (
                AttentionOutputBridge.TRITON_BHSD_PROJECTION
                if program.residual_norm is ResidualNormBackend.TRITON_LINEAR_MIXED
                else AttentionOutputBridge.TORCH_BHSD_TO_BSD
            )
            kind = "dependency_closure"
            note += " Its direct-BSD output contract is replaced with a legal bridge."
        program = replace(
            program,
            attention=AttentionBackend.CAUSAL_SDPA,
            attention_output_bridge=bridge,
        )
        schedule = replace(
            schedule,
            attention_launch=None,
            attention_output_projection_launch=(
                schedule.attention_output_projection_launch
                if bridge is AttentionOutputBridge.TRITON_BHSD_PROJECTION
                and program.residual_norm
                is not ResidualNormBackend.TRITON_LINEAR_MIXED
                else None
            ),
        )
        return _candidate(
            full,
            family,
            program,
            schedule,
            variant_kind=kind,
            note=note,
        )

    if family is AblationFamily.LAYOUT:
        native_qkv = (
            program.qkv_materialization is QKVMaterialization.TRITON_NATIVE_BHSD
        )
        optimized_bridge = program.attention_output_bridge in {
            AttentionOutputBridge.TRITON_BHSD_PROJECTION,
            AttentionOutputBridge.ATTENTION_DIRECT_BSD,
        }
        if not native_qkv and not optimized_bridge:
            return None
        kind = "atomic"
        notes = ["Native QKV materialization is replaced by a view when present."]
        qkv_materialization = (
            QKVMaterialization.VIEW if native_qkv else program.qkv_materialization
        )
        bridge = program.attention_output_bridge
        residual_norm = program.residual_norm
        ffn = program.ffn
        initial_norm = program.initial_norm

        if bridge is AttentionOutputBridge.TRITON_BHSD_PROJECTION:
            bridge = AttentionOutputBridge.TORCH_BHSD_TO_BSD
            notes.append("The fused BHSD output bridge is replaced by Torch layout conversion.")
            if residual_norm is ResidualNormBackend.TRITON_LINEAR_MIXED:
                residual_norm = ResidualNormBackend.TRITON_MIXED
                schedule = replace(
                    schedule,
                    residual_norm_launch=TritonNormParams(
                        block_rows=2,
                        num_warps=2,
                    ),
                )
                kind = "dependency_closure"
                notes.append("The zero-copy linear boundary falls back to mixed norm.")
                if ffn is FFNBackend.TRITON_FUSED_MLP_BOUNDARY:
                    ffn = FFNBackend.TORCH
                    schedule = replace(
                        schedule,
                        ffn_launch=None,
                        ffn_input_launch=None,
                    )
                    notes.append("Its boundary-coupled FFN also falls back to Torch.")
        elif bridge is AttentionOutputBridge.ATTENTION_DIRECT_BSD:
            if program.attention is AttentionBackend.TRITON_SHAPE13:
                bridge = AttentionOutputBridge.TORCH_BHSD_TO_BSD
                notes.append("Shape-13 direct BSD output is replaced by its BHSD/Torch path.")
            else:
                kind = "partial"
                notes.append(
                    "Direct BSD output remains part of the Dh8 attention contract; "
                    "this cell isolates only the QKV-side layout."
                )

        if initial_norm is InitialNormBackend.TRITON_FUSED_QKV and native_qkv:
            initial_norm = InitialNormBackend.TORCH
            kind = "partial"
            notes.append("Fused initial norm/QKV falls back with native QKV removal.")

        program = replace(
            program,
            qkv_materialization=qkv_materialization,
            attention_output_bridge=bridge,
            residual_norm=residual_norm,
            initial_norm=initial_norm,
            ffn=ffn,
        )
        schedule = replace(
            schedule,
            qkv_launch=None if native_qkv else schedule.qkv_launch,
            attention_output_projection_launch=(
                None
                if bridge is not AttentionOutputBridge.TRITON_BHSD_PROJECTION
                else schedule.attention_output_projection_launch
            ),
            initial_norm_launch=(
                None
                if initial_norm is InitialNormBackend.TORCH
                else schedule.initial_norm_launch
            ),
        )
        return _candidate(
            full,
            family,
            program,
            schedule,
            variant_kind=kind,
            note=" ".join(notes),
        )

    if family is AblationFamily.PROJECTION:
        clean_projection_ablation = (
            program.precision_plan is not PrecisionPlan.INPUT_DTYPE
            and program.qkv_materialization is QKVMaterialization.VIEW
            and program.attention_output_bridge
            is AttentionOutputBridge.TORCH_BHSD_TO_BSD
            and program.ffn is FFNBackend.TORCH
            and program.residual_norm is ResidualNormBackend.TORCH
            and program.initial_norm is InitialNormBackend.TORCH
        )
        if not clean_projection_ablation:
            return None
        program = replace(
            program,
            precision_plan=PrecisionPlan.INPUT_DTYPE,
            qkv_projection=ProjectionBackend.INPUT_DTYPE,
            attention_output_projection=ProjectionBackend.INPUT_DTYPE,
            ffn_input_projection=ProjectionBackend.INPUT_DTYPE,
            ffn_output_projection=ProjectionBackend.INPUT_DTYPE,
        )
        return _candidate(
            full,
            family,
            program,
            schedule,
            note="The clean library path restores all projections to input dtype.",
        )

    if family is AblationFamily.FFN:
        if program.ffn is FFNBackend.TORCH:
            return None
        program = replace(program, ffn=FFNBackend.TORCH)
        schedule = replace(schedule, ffn_launch=None, ffn_input_launch=None)
        return _candidate(
            full,
            family,
            program,
            schedule,
            note="The deployed FFN implementation is replaced by the Torch exact-GELU path.",
        )

    if family is AblationFamily.NORM:
        if (
            program.residual_norm is ResidualNormBackend.TORCH
            and program.initial_norm is InitialNormBackend.TORCH
        ):
            return None
        kind = "atomic"
        notes = ["Initial and residual norm specializations fall back to Torch."]
        bridge = program.attention_output_bridge
        ffn = program.ffn
        if (
            program.residual_norm is ResidualNormBackend.TRITON_LINEAR_MIXED
            and bridge is AttentionOutputBridge.TRITON_BHSD_PROJECTION
        ):
            bridge = AttentionOutputBridge.TORCH_BHSD_TO_BSD
            kind = "dependency_closure"
            notes.append("The boundary-fused output bridge is removed with linear norm.")
        if ffn is FFNBackend.TRITON_FUSED_MLP_BOUNDARY:
            ffn = FFNBackend.TORCH
            kind = "dependency_closure"
            notes.append("The residual-boundary FFN also falls back to Torch.")
        program = replace(
            program,
            residual_norm=ResidualNormBackend.TORCH,
            initial_norm=InitialNormBackend.TORCH,
            attention_output_bridge=bridge,
            ffn=ffn,
        )
        schedule = replace(
            schedule,
            residual_norm_launch=None,
            initial_norm_launch=None,
            attention_output_projection_launch=(
                None
                if bridge is AttentionOutputBridge.TORCH_BHSD_TO_BSD
                else schedule.attention_output_projection_launch
            ),
            ffn_launch=None if ffn is FFNBackend.TORCH else schedule.ffn_launch,
            ffn_input_launch=(
                None if ffn is FFNBackend.TORCH else schedule.ffn_input_launch
            ),
        )
        return _candidate(
            full,
            family,
            program,
            schedule,
            variant_kind=kind,
            note=" ".join(notes),
        )

    raise AssertionError(f"unhandled ablation family: {family}")


def _execution_context(
    shape: TransformerShape,
    variant: RunVariant,
    device: torch.device,
) -> ExecutionContext:
    return ExecutionContext(
        batch_size=shape.batch_size,
        seq_len=shape.seq_len,
        d_model=shape.d_model,
        num_heads=shape.num_heads,
        causal=shape.causal,
        device=device,
        dtype=official.resolve_dtype(variant.dtype),
        training=False,
        grad_enabled=False,
        input_contiguous=True,
        has_valid_token_mask=variant.padding_ratio > 0.0,
        mask_compatible=True,
        ffn_dim=shape.ffn_dim,
        num_layers=shape.num_layers,
    )


def _measurement_view(result: object) -> dict[str, Any]:
    return {
        "median_ms": result.optimized.median_ms,
        "p90_ms": result.optimized.p90_ms,
        "passed": result.passed,
        "execution_matches": result.execution_matches,
        "max_tolerance_ratio": result.max_tolerance_ratio,
        "peak_memory_bytes": result.peak_memory_bytes,
    }


def _measure_pair(
    shape: TransformerShape,
    variant: RunVariant,
    deployed: ConfigSpec,
    candidate: AblationCandidate,
    device: str,
) -> dict[str, Any]:
    resolved_device = official.resolve_device(device)
    context = _execution_context(shape, variant, resolved_device)
    hardware = HardwareCapabilities.detect(resolved_device)
    builder = PlanBuilder()
    for role, config in (("deployed", deployed), ("ablated", candidate.config)):
        compilation = builder.evaluate(config, context, hardware)
        if not compilation.accepted:
            return {
                "status": "static_rejection",
                "error": role,
                "violations": [item.to_dict() for item in compilation.violations],
            }

    protocol = ablation_protocol(shape.case_id)
    paired = measure_paired_configs(
        shape,
        challenger_config=deployed,
        incumbent_config=candidate.config,
        variant=variant,
        protocol=protocol,
        device=resolved_device,
    )
    deployed_result = paired.challenger
    ablated_result = paired.incumbent
    valid = all(
        (
            deployed_result.passed,
            ablated_result.passed,
            deployed_result.execution_matches,
            ablated_result.execution_matches,
        )
    )
    paired_slowdowns = tuple(float(value) for value in paired.paired_ratios)
    ablation_slowdown = None
    retained_performance = None
    if valid:
        ablation_slowdown, retained_performance = _aggregate_ablation_effect(
            deployed_median_ms=float(deployed_result.optimized.median_ms),
            ablated_median_ms=float(ablated_result.optimized.median_ms),
        )
    return {
        "status": "measured" if valid else "correctness_failed",
        "deployed": _measurement_view(deployed_result),
        "ablated": _measurement_view(ablated_result),
        "ablation_slowdown": ablation_slowdown,
        "retained_performance_fraction": retained_performance,
        "paired_ablation_slowdowns": list(paired_slowdowns),
    }


def _aggregate_ablation_effect(
    *,
    deployed_median_ms: float,
    ablated_median_ms: float,
) -> tuple[float, float]:
    """Return slowdown and the exact ablated/full speedup ratio."""

    if deployed_median_ms <= 0 or ablated_median_ms <= 0:
        raise ValueError("ablation medians must be positive")
    return (
        ablated_median_ms / deployed_median_ms,
        deployed_median_ms / ablated_median_ms,
    )


def _deployed_configs(
    project_root: Path,
    case_ids: tuple[str, ...],
    variant: RunVariant,
    device: str,
) -> dict[str, ConfigSpec]:
    hardware = EnvironmentFingerprint.detect(
        torch.device(device),
        project_root=project_root,
        scope=ImplementationScope.RESIDENT,
    )
    configs: dict[str, ConfigSpec] = {}
    for case_id in case_ids:
        shape = load_shape(project_root, case_id)
        if shape.streamed:
            raise ValueError("Shape 14 uses capacity evidence, not resident ablation")
        config = resolve_deployed_config(
            hardware=hardware,
            shape=shape_fingerprint(shape, variant),
        )
        if config is None:
            raise ValueError(f"no deployed config for {case_id} on this device")
        configs[case_id] = config
    return configs


def _base_record(
    shape: TransformerShape,
    deployed: ConfigSpec,
    family: AblationFamily,
) -> dict[str, Any]:
    return {
        "case_id": shape.case_id,
        "mechanism_id": family.value,
        "mechanism_label": FAMILY_LABELS[family],
        "deployed_config_id": deployed.config_id,
        "completed_at": _utc_now(),
    }


def run_component_ablation_suite(
    *,
    project_root: Path,
    case_ids: tuple[str, ...] = DEFAULT_ABLATION_SHAPES,
    families: tuple[AblationFamily, ...] = ABLATION_FAMILIES,
    variant: RunVariant | None = None,
    device: str = "cuda:0",
    output_directory: Path,
) -> AblationSuiteResult:
    """Run a deterministic case-by-family grid in serial fresh processes."""

    if not case_ids:
        raise ValueError("component ablation requires at least one shape")
    if not families:
        raise ValueError("component ablation requires at least one family")
    variant = RunVariant() if variant is None else variant
    summary_path = output_directory / "ablation.json"
    if summary_path.exists():
        raise ValueError(f"ablation output already exists: {summary_path}")
    output_directory.mkdir(parents=True, exist_ok=True)
    deployed_configs = _deployed_configs(
        project_root,
        case_ids,
        variant,
        device,
    )
    started = time.monotonic()
    comparisons: list[dict[str, Any]] = []
    total = len(case_ids) * len(families)
    summary: dict[str, Any] = {
        "schema_version": 2,
        "run_id": output_directory.name,
        "status": "running",
        "device": device,
        "variant": variant.to_dict(),
        "started_at": _utc_now(),
        "finished_at": None,
        "elapsed_seconds": 0.0,
        "progress": {"completed": 0, "total": total, "measured": 0},
        "comparisons": comparisons,
    }
    write_json(summary_path, summary)

    try:
        for case_id in case_ids:
            shape = load_shape(project_root, case_id)
            deployed = deployed_configs[case_id]
            for family in families:
                family = AblationFamily(family)
                record = _base_record(shape, deployed, family)
                candidate = build_ablation_candidate(deployed, family, shape)
                if candidate is None:
                    if (
                        family is AblationFamily.RUNTIME
                        and case_id == "official_06"
                    ):
                        record.update(
                            {
                                "status": "capacity_excluded",
                                "variant_kind": "capacity_failure",
                                "note": (
                                    "Removing batch tiling exposes the B=10000 full batch; "
                                    "it is excluded from latency attribution to avoid an OOM "
                                    "or timeout being misread as a component cost."
                                ),
                            }
                        )
                    elif (
                        family is AblationFamily.PROJECTION
                        and deployed.program.precision_plan
                        is not PrecisionPlan.INPUT_DTYPE
                    ):
                        record.update(
                            {
                                "status": "not_isolatable",
                                "variant_kind": "dependency_coupled",
                                "note": (
                                    "The specialized projection precision is active but "
                                    "coupled to the deployed Triton/layout path, so no "
                                    "clean single-family counterfactual exists."
                                ),
                            }
                        )
                    else:
                        record.update(
                            {
                                "status": "not_applicable",
                                "variant_kind": "not_applicable",
                                "note": "The deployed program already uses the fallback path.",
                            }
                        )
                else:
                    record.update(
                        {
                            "variant_kind": candidate.variant_kind,
                            "note": candidate.note,
                            "ablated_config_id": candidate.config.config_id,
                            "changed_fields": list(candidate.changed_fields),
                            "protocol": {
                                "accuracy_trials": ablation_protocol(
                                    case_id
                                ).accuracy_trials,
                                "warmup": ablation_protocol(case_id).warmup,
                                "repeats": ablation_protocol(case_id).repeats,
                                "rounds": ablation_protocol(case_id).rounds,
                            },
                        }
                    )
                    try:
                        measurement = run_in_fresh_process(
                            _measure_pair,
                            shape,
                            variant,
                            deployed,
                            candidate,
                            device,
                        )
                        record.update(measurement)
                    except IsolatedProcessError as exc:
                        record.update(status="failed", error=str(exc))
                record["completed_at"] = _utc_now()
                comparisons.append(record)
                summary["progress"] = {
                    "completed": len(comparisons),
                    "total": total,
                    "measured": sum(
                        item.get("status") == "measured" for item in comparisons
                    ),
                }
                summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
                write_json(summary_path, summary)
    except KeyboardInterrupt:
        summary.update(
            status="interrupted",
            finished_at=_utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
        )
        write_json(summary_path, summary)
        raise

    failures = [
        item
        for item in comparisons
        if item.get("status") in {"failed", "static_rejection", "correctness_failed"}
    ]
    summary.update(
        status="completed_with_failures" if failures else "completed",
        finished_at=_utc_now(),
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    write_json(summary_path, summary)
    return AblationSuiteResult(
        path=summary_path,
        summary=summary,
        exit_code=1 if failures else 0,
    )


def write_component_ablation_csv(summary: dict[str, Any], path: Path) -> Path:
    """Write the compact, plot-facing table from one immutable run summary."""

    columns = (
        "case_id",
        "mechanism_id",
        "mechanism_label",
        "status",
        "variant_kind",
        "ablation_slowdown",
        "retained_performance_fraction",
        "deployed_median_ms",
        "ablated_median_ms",
        "timed_samples_per_config",
        "correctness_passed",
        "completed_at",
        "note",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in summary.get("comparisons", []):
            protocol = item.get("protocol") or {}
            deployed = item.get("deployed") or {}
            ablated = item.get("ablated") or {}
            valid = item.get("status") == "measured"
            slowdown = item.get("ablation_slowdown")
            retained = item.get("retained_performance_fraction")
            if isinstance(slowdown, float) and not math.isfinite(slowdown):
                slowdown = None
            if isinstance(retained, float) and not math.isfinite(retained):
                retained = None
            writer.writerow(
                {
                    "case_id": item.get("case_id"),
                    "mechanism_id": item.get("mechanism_id"),
                    "mechanism_label": item.get("mechanism_label"),
                    "status": item.get("status"),
                    "variant_kind": item.get("variant_kind"),
                    "ablation_slowdown": slowdown if valid else "",
                    "retained_performance_fraction": retained if valid else "",
                    "deployed_median_ms": deployed.get("median_ms") if valid else "",
                    "ablated_median_ms": ablated.get("median_ms") if valid else "",
                    "timed_samples_per_config": (
                        int(protocol.get("repeats", 0))
                        * int(protocol.get("rounds", 0))
                        if protocol
                        else ""
                    ),
                    "correctness_passed": str(valid).lower(),
                    "completed_at": item.get("completed_at"),
                    "note": item.get("note"),
                }
            )
    temporary.replace(path)
    return path


__all__ = [
    "ABLATION_FAMILIES",
    "DEFAULT_ABLATION_SHAPES",
    "AblationCandidate",
    "AblationFamily",
    "AblationSuiteResult",
    "ablation_protocol",
    "build_ablation_candidate",
    "new_ablation_run_directory",
    "run_component_ablation_suite",
    "write_component_ablation_csv",
]
