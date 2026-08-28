"""Typed IPC and compact-result contracts shared by the runner layers.

JSON dictionaries remain the process and persistence format.  This module owns
the small amount of parsing, compaction, and validation needed at those
boundaries so individual runner stages do not reinterpret the same fields.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    RunVariant,
    TransformerShape,
)

RunKind = Literal["benchmark", "profile", "probe"]
RunTarget = Literal["baseline", "solution"]
RUN_RESULT_SCHEMA_VERSION = 5
EXECUTION_PATH_STRING_FIELDS = (
    "requested_policy",
    "selected_policy",
    "qkv_projection",
    "attention_backend",
    "runtime_wrapper",
    "residual_norm_backend",
    "causal_mask",
    "valid_token_mask",
)

ALLOWED_OUTCOMES = frozenset(
    {
        "success",
        "invalid_output",
        "unsupported",
        "build_error",
        "oom",
        "timeout",
        "cancelled",
        "runtime_error",
    }
)


class WorkerRequestDocument(TypedDict, total=False):
    """Serialized request accepted by ``runner.worker``."""

    run_kind: RunKind
    project_root: str
    shape: dict[str, Any]
    variant: dict[str, Any]
    protocol: dict[str, Any]
    device: str
    target: RunTarget
    solution_policy: str
    probe_mode: str
    matmul_precision: str
    allow_tf32: bool


class WorkerResponseDocument(TypedDict, total=False):
    """Serialized response returned by ``runner.worker``."""

    outcome: str
    solution_source_sha256: str | None
    environment: dict[str, Any] | None
    correctness: dict[str, Any] | None
    performance: dict[str, Any] | None
    profile: dict[str, Any] | None
    probe: dict[str, Any] | None
    execution_path: ExecutionPathDocument | None
    failure: dict[str, Any] | None


class CorrectnessDocument(TypedDict, total=False):
    passed: bool
    trial_count: int
    failed_elements: int
    max_abs_error: float
    max_relative_error: float
    diagnostic: str
    skipped: str


class ExecutionPathDocument(TypedDict, total=False):
    """Common execution identity emitted by baseline and Solution targets."""

    requested_policy: str
    selected_policy: str
    qkv_projection: str
    attention_backend: str
    runtime_wrapper: str
    residual_norm_backend: str
    causal_mask: str
    valid_token_mask: str
    fallback_reasons: list[str]
    execution_mode: Literal["eager", "torch_compile"]
    observed_execution: dict[str, Any]


class TimingDocument(TypedDict, total=False):
    median_ms: float
    p90_ms: float


class BenchmarkPerformanceDocument(TypedDict, total=False):
    timer: str
    sample_count: int
    baseline: TimingDocument
    target: TimingDocument
    speedup: float


@dataclass(frozen=True)
class WorkerRequest:
    """Validated, immutable view of one worker request."""

    run_kind: RunKind
    device: str
    project_root: Path | None = None
    shape: TransformerShape | None = None
    variant: RunVariant | None = None
    protocol: MeasurementProtocol | None = None
    target: RunTarget | None = None
    solution_policy: str | None = None
    probe_mode: str | None = None
    matmul_precision: str | None = None
    allow_tf32: bool | None = None

    def __post_init__(self) -> None:
        if self.run_kind not in {"benchmark", "profile", "probe"}:
            raise ContractError(f"unsupported run_kind: {self.run_kind!r}")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ContractError("worker request device must be a non-empty string")
        if self.run_kind == "probe":
            if self.probe_mode not in {"routing", "diagnostic"}:
                raise ContractError(f"unsupported probe mode: {self.probe_mode!r}")
            if self.matmul_precision not in {"highest", "high", "medium"}:
                raise ContractError(
                    f"unsupported matmul precision: {self.matmul_precision!r}"
                )
            if not isinstance(self.allow_tf32, bool):
                raise ContractError("allow_tf32 must be a boolean")
            return
        if not isinstance(self.project_root, Path):
            raise ContractError("worker request project_root must be a Path")
        if not isinstance(self.shape, TransformerShape):
            raise ContractError("worker request shape must be a TransformerShape")
        if not isinstance(self.variant, RunVariant):
            raise ContractError("worker request variant must be a RunVariant")
        if not isinstance(self.protocol, MeasurementProtocol):
            raise ContractError("worker request protocol must be a MeasurementProtocol")
        if self.target not in {"baseline", "solution"}:
            raise ContractError(f"unsupported {self.run_kind} target: {self.target!r}")
        if self.solution_policy is not None and (
            not isinstance(self.solution_policy, str)
            or not self.solution_policy.strip()
        ):
            raise ContractError("solution_policy must be a non-empty string")
        self.shape.validate()
        self.variant.validate()
        self.protocol.validate()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WorkerRequest:
        raw_kind = value.get("run_kind", "benchmark")
        if raw_kind not in {"benchmark", "profile", "probe"}:
            raise ContractError(f"unsupported run_kind: {raw_kind!r}")
        run_kind: RunKind = raw_kind

        device = value.get("device")
        if not isinstance(device, str) or not device.strip():
            raise ContractError("worker request device must be a non-empty string")

        if run_kind == "probe":
            allowed = {
                "run_kind",
                "device",
                "probe_mode",
                "matmul_precision",
                "allow_tf32",
            }
            extra = sorted(set(value) - allowed)
            if extra:
                raise ContractError(f"unexpected probe request fields: {extra}")
            probe_mode = value.get("probe_mode", "diagnostic")
            if probe_mode not in {"routing", "diagnostic"}:
                raise ContractError(f"unsupported probe mode: {probe_mode!r}")
            matmul_precision = value.get("matmul_precision", "high")
            if matmul_precision not in {"highest", "high", "medium"}:
                raise ContractError(
                    f"unsupported matmul precision: {matmul_precision!r}"
                )
            allow_tf32 = value.get("allow_tf32", True)
            if not isinstance(allow_tf32, bool):
                raise ContractError("allow_tf32 must be a boolean")
            return cls(
                run_kind=run_kind,
                device=device,
                probe_mode=probe_mode,
                matmul_precision=matmul_precision,
                allow_tf32=allow_tf32,
            )

        allowed = {
            "run_kind",
            "project_root",
            "shape",
            "variant",
            "protocol",
            "device",
            "target",
            "solution_policy",
        }
        required = allowed - {"solution_policy"}
        missing = sorted(required - set(value))
        extra = sorted(set(value) - allowed)
        if missing or extra:
            raise ContractError(
                f"invalid {run_kind} request fields; missing={missing}, extra={extra}"
            )
        project_root_value = value.get("project_root")
        if not isinstance(project_root_value, str) or not project_root_value:
            raise ContractError("worker request project_root must be a path string")
        raw_shape = value.get("shape")
        if not isinstance(raw_shape, dict):
            raise ContractError("worker request shape must be an object")
        raw_variant = value.get("variant")
        if not isinstance(raw_variant, dict):
            raise ContractError("worker request variant must be an object")
        raw_protocol = value.get("protocol")
        if not isinstance(raw_protocol, dict):
            raise ContractError("worker request protocol must be an object")
        try:
            protocol = MeasurementProtocol(**raw_protocol)
        except TypeError as exc:
            raise ContractError(f"invalid measurement protocol fields: {exc}") from exc
        protocol.validate()
        target = value.get("target")
        if target not in {"baseline", "solution"}:
            raise ContractError(f"unsupported {run_kind} target: {target!r}")
        solution_policy = value.get("solution_policy")
        if solution_policy is not None and (
            not isinstance(solution_policy, str) or not solution_policy.strip()
        ):
            raise ContractError("solution_policy must be a non-empty string")
        return cls(
            run_kind=run_kind,
            device=device,
            project_root=Path(project_root_value).resolve(),
            shape=TransformerShape.from_dict(raw_shape),
            variant=RunVariant.from_dict(raw_variant),
            protocol=protocol,
            target=target,
            solution_policy=solution_policy,
        )

    def as_dict(self) -> WorkerRequestDocument:
        document: WorkerRequestDocument = {
            "run_kind": self.run_kind,
            "device": self.device,
        }
        if self.run_kind == "probe":
            assert self.probe_mode is not None
            assert self.matmul_precision is not None
            assert self.allow_tf32 is not None
            document.update(
                probe_mode=self.probe_mode,
                matmul_precision=self.matmul_precision,
                allow_tf32=self.allow_tf32,
            )
            return document
        assert self.project_root is not None
        assert self.shape is not None
        assert self.variant is not None
        assert self.protocol is not None
        assert self.target is not None
        document.update(
            project_root=str(self.project_root),
            shape=self.shape.as_dict(),
            variant=self.variant.as_dict(),
            protocol=self.protocol.as_dict(),
            target=self.target,
        )
        if self.solution_policy is not None:
            document["solution_policy"] = self.solution_policy
        return document


@dataclass(frozen=True)
class TimingStats:
    median_ms: float
    p90_ms: float

    @classmethod
    def from_dict(
        cls,
        value: Any,
        *,
        side: str,
    ) -> TimingStats:
        _median, error = validate_timing_side(value, side)
        if error is not None:
            raise ContractError(error)
        assert isinstance(value, Mapping)
        return cls(
            median_ms=float(value["median_ms"]),
            p90_ms=float(value["p90_ms"]),
        )


@dataclass(frozen=True)
class BenchmarkPerformance:
    timer: str
    sample_count: int
    baseline: TimingStats
    target: TimingStats | None
    speedup: float | None


def finite_positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        return None
    return normalized


def compact_correctness(value: Any) -> CorrectnessDocument | None:
    if not isinstance(value, Mapping):
        return None
    compact: CorrectnessDocument = {
        key: value[key]
        for key in (
            "passed",
            "trial_count",
            "failed_elements",
            "max_abs_error",
            "max_relative_error",
            "diagnostic",
            "skipped",
        )
        if value.get(key) is not None
    }
    if value.get("passed") is False and "diagnostic" not in compact:
        trials = value.get("trials")
        if isinstance(trials, list):
            for trial in trials:
                if isinstance(trial, Mapping) and trial.get("error") is not None:
                    compact["diagnostic"] = str(trial["error"])[-500:]
                    break
    return compact or None


def _compact_timing(value: Any) -> TimingDocument | None:
    if not isinstance(value, Mapping):
        return None
    compact: TimingDocument = {
        key: value[key] for key in ("median_ms", "p90_ms") if value.get(key) is not None
    }
    return compact or None


def compact_performance(
    value: Any,
    target: str,
) -> BenchmarkPerformanceDocument | None:
    if not isinstance(value, Mapping):
        return None
    baseline_source = value.get("baseline")
    baseline = _compact_timing(baseline_source)
    if baseline is None:
        return None
    compact: BenchmarkPerformanceDocument = {
        "timer": value.get("timer"),
        "sample_count": baseline_source.get("sample_count")
        if isinstance(baseline_source, Mapping)
        else None,
        "baseline": baseline,
    }
    if target == "solution":
        measured = _compact_timing(value.get("target"))
        if measured is not None:
            compact["target"] = measured
        if value.get("speedup") is not None:
            compact["speedup"] = value["speedup"]
    return compact


def validate_correctness(value: Any, *, expected_trials: Any) -> str | None:
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        return "invalid_correctness"
    trial_count = value.get("trial_count")
    failed_elements = value.get("failed_elements")
    max_abs_error = value.get("max_abs_error")
    if (
        isinstance(expected_trials, bool)
        or not isinstance(expected_trials, int)
        or expected_trials <= 0
        or isinstance(trial_count, bool)
        or not isinstance(trial_count, int)
        or trial_count != expected_trials
        or isinstance(failed_elements, bool)
        or not isinstance(failed_elements, int)
        or failed_elements != 0
        or isinstance(max_abs_error, bool)
        or not isinstance(max_abs_error, (int, float))
        or not math.isfinite(float(max_abs_error))
        or float(max_abs_error) < 0
    ):
        return "invalid_correctness_summary"
    return None


def validate_timing_side(
    value: Any,
    side: str,
) -> tuple[float | None, str | None]:
    if not isinstance(value, Mapping):
        return None, f"missing_{side}_timing"
    median = finite_positive(value.get("median_ms"))
    if median is None:
        return None, f"invalid_{side}_median"
    p90 = finite_positive(value.get("p90_ms"))
    if p90 is None:
        return None, f"invalid_{side}_p90"
    if p90 < median and not math.isclose(p90, median, rel_tol=1e-12, abs_tol=1e-12):
        return None, f"{side}_p90_below_median"
    return median, None


def validate_execution_path(value: Any) -> str | None:
    """Validate the common execution identity shared by both targets."""

    if not isinstance(value, Mapping):
        return "missing_execution_path"
    for field in EXECUTION_PATH_STRING_FIELDS:
        item = value.get(field)
        if not isinstance(item, str) or not item:
            return f"missing_{field}"
    if value.get("execution_mode") not in {"eager", "torch_compile"}:
        return "invalid_execution_mode"
    fallback_reasons = value.get("fallback_reasons")
    if not isinstance(fallback_reasons, list) or not all(
        isinstance(reason, str) and reason for reason in fallback_reasons
    ):
        return "invalid_fallback_reasons"
    return None


def validate_benchmark_performance(
    value: Any,
    *,
    target: str,
    repeats: Any,
    rounds: Any,
    expected_timer: str,
) -> tuple[BenchmarkPerformance | None, str | None]:
    if (
        isinstance(repeats, bool)
        or not isinstance(repeats, int)
        or repeats <= 0
        or isinstance(rounds, bool)
        or not isinstance(rounds, int)
        or rounds <= 0
    ):
        return None, "invalid_protocol_counts"
    if not isinstance(value, Mapping):
        return None, "missing_performance"
    timer = value.get("timer")
    if timer != expected_timer:
        return (
            None,
            "non_cuda_timing" if expected_timer == "cuda_event" else "timer_mismatch",
        )
    expected_count = repeats * rounds
    sample_count = value.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count != expected_count
    ):
        return None, "sample_count_mismatch"
    baseline_median, error = validate_timing_side(value.get("baseline"), "baseline")
    if error is not None:
        return None, error
    assert baseline_median is not None
    baseline = TimingStats.from_dict(value["baseline"], side="baseline")
    if target == "baseline":
        return BenchmarkPerformance(timer, sample_count, baseline, None, None), None

    target_median, error = validate_timing_side(value.get("target"), "target")
    if error is not None:
        return None, error
    assert target_median is not None
    recomputed_speedup = baseline_median / target_median
    stored_speedup = finite_positive(value.get("speedup"))
    if stored_speedup is None:
        return None, "invalid_speedup"
    if not math.isclose(
        stored_speedup,
        recomputed_speedup,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return None, "speedup_mismatch"
    measured = TimingStats.from_dict(value["target"], side="target")
    return (
        BenchmarkPerformance(
            timer,
            sample_count,
            baseline,
            measured,
            recomputed_speedup,
        ),
        None,
    )


def worker_response_error(response: Mapping[str, Any], run_kind: str) -> str | None:
    """Validate the common worker response envelope at the IPC boundary."""

    outcome = response.get("outcome")
    if outcome not in ALLOWED_OUTCOMES:
        return f"worker returned unsupported outcome: {outcome!r}"
    if outcome != "success":
        if not isinstance(response.get("failure"), Mapping):
            return "failed worker response is missing failure details"
        return None
    if response.get("failure") is not None:
        return "successful worker response contains failure details"
    if not isinstance(response.get("environment"), Mapping):
        return "successful worker response is missing environment details"
    required_payload = {
        "benchmark": "performance",
        "profile": "profile",
        "probe": "probe",
    }.get(run_kind)
    if required_payload is None:
        return f"unsupported worker run_kind: {run_kind!r}"
    if not isinstance(response.get(required_payload), Mapping):
        return f"successful {run_kind} response is missing {required_payload}"
    if run_kind == "benchmark" and not isinstance(response.get("correctness"), Mapping):
        return "successful benchmark response is missing correctness"
    if run_kind in {"benchmark", "profile"}:
        path_error = validate_execution_path(response.get("execution_path"))
        if path_error is not None:
            return f"invalid execution path: {path_error}"
    return None


def parse_worker_response(
    response: Mapping[str, Any],
    *,
    run_kind: str,
) -> WorkerResponseDocument:
    """Return a typed response document or reject the IPC envelope."""

    error = worker_response_error(response, run_kind)
    if error is not None:
        raise ContractError(error)
    return cast(WorkerResponseDocument, dict(response))
