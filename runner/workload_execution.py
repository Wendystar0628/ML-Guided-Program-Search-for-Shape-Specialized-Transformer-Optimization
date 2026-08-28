"""Shape-derived execution plans for resident and streamed workloads."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    RunVariant,
    TransformerShape,
)

ExecutionMode = Literal["resident", "batch_streamed"]
ReferenceKind = Literal["live_baseline", "internal_query_block"]
STREAMED_POLICY_SELECTOR = "screen"

DEFAULT_RESIDENT_ATTENTION_LIMIT_BYTES = 8 * 1024**3
_TIMING_MICROBATCH_CHOICES = (1, 2, 4, 8, 16, 32)
_DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}


@dataclass(frozen=True, slots=True)
class WorkloadExecutionPlan:
    """Immutable decision made before allocating workload tensors."""

    execution_mode: ExecutionMode
    reference_kind: ReferenceKind
    estimated_dense_attention_bytes: int
    resident_attention_limit_bytes: int
    validation_microbatch_size: int | None
    timing_microbatch_candidates: tuple[int, ...]
    formal_eligible: bool

    @property
    def is_streamed(self) -> bool:
        return self.execution_mode == "batch_streamed"


def estimate_dense_attention_bytes(
    shape: TransformerShape,
    variant: RunVariant,
) -> int:
    """Estimate the dense score, softmax, and causal-mask working set.

    The estimate mirrors the official operation boundaries: one attention
    matrix in the run dtype, one FP32 softmax representation, and one shared
    boolean causal mask. It is deliberately independent of device identity.
    """

    shape.validate()
    variant.validate()
    dtype_bytes = _DTYPE_BYTES[variant.dtype]
    sequence_squared = shape.seq_len * shape.seq_len
    attention_elements = shape.batch_size * shape.num_heads * sequence_squared
    dense_bytes = attention_elements * (dtype_bytes + 4)
    if shape.causal:
        dense_bytes += sequence_squared
    return dense_bytes


def plan_workload_execution(
    shape: TransformerShape,
    variant: RunVariant | None = None,
    *,
    resident_attention_limit_bytes: int = DEFAULT_RESIDENT_ATTENTION_LIMIT_BYTES,
) -> WorkloadExecutionPlan:
    """Select a resident or batch-streamed path from workload memory shape."""

    if (
        isinstance(resident_attention_limit_bytes, bool)
        or not isinstance(resident_attention_limit_bytes, int)
        or resident_attention_limit_bytes <= 0
    ):
        raise ContractError("resident attention limit must be a positive integer")
    effective_variant = variant or RunVariant()
    dense_bytes = estimate_dense_attention_bytes(shape, effective_variant)
    if dense_bytes <= resident_attention_limit_bytes:
        return WorkloadExecutionPlan(
            execution_mode="resident",
            reference_kind="live_baseline",
            estimated_dense_attention_bytes=dense_bytes,
            resident_attention_limit_bytes=resident_attention_limit_bytes,
            validation_microbatch_size=None,
            timing_microbatch_candidates=(),
            formal_eligible=True,
        )
    timing_candidates = tuple(
        size
        for size in _TIMING_MICROBATCH_CHOICES
        if size <= shape.batch_size and shape.batch_size % size == 0
    )
    return WorkloadExecutionPlan(
        execution_mode="batch_streamed",
        reference_kind="internal_query_block",
        estimated_dense_attention_bytes=dense_bytes,
        resident_attention_limit_bytes=resident_attention_limit_bytes,
        validation_microbatch_size=1,
        timing_microbatch_candidates=timing_candidates,
        formal_eligible=False,
    )


def effective_protocol(
    protocol: MeasurementProtocol,
    plan: WorkloadExecutionPlan,
) -> MeasurementProtocol:
    """Return the truthful measurement counts used by one execution plan."""

    protocol.validate()
    if not plan.is_streamed:
        return protocol
    if protocol.preset == "smoke":
        result = replace(
            protocol,
            accuracy_trials=1,
            warmup=1,
            repeats=1,
            rounds=1,
            timeout_seconds=max(protocol.timeout_seconds, 300.0),
        )
    elif protocol.preset == "formal":
        result = replace(
            protocol,
            accuracy_trials=1,
            warmup=2,
            repeats=1,
            rounds=3,
            timeout_seconds=max(protocol.timeout_seconds, 1200.0),
        )
    else:
        raise ContractError(
            f"streamed execution does not support preset: {protocol.preset!r}"
        )
    result.validate()
    return result


def all_benchmark_shapes(
    shapes: Sequence[TransformerShape],
) -> tuple[TransformerShape, ...]:
    """Return every published shape in its original order."""

    return tuple(shapes)


def resident_benchmark_shapes(
    shapes: Sequence[TransformerShape],
    variant: RunVariant | None = None,
    *,
    resident_attention_limit_bytes: int = DEFAULT_RESIDENT_ATTENTION_LIMIT_BYTES,
) -> tuple[TransformerShape, ...]:
    """Return shapes that fit the ordinary resident benchmark path."""

    effective_variant = variant or RunVariant()
    return tuple(
        shape
        for shape in shapes
        if not plan_workload_execution(
            shape,
            effective_variant,
            resident_attention_limit_bytes=resident_attention_limit_bytes,
        ).is_streamed
    )


def streamed_benchmark_shapes(
    shapes: Sequence[TransformerShape],
    variant: RunVariant | None = None,
    *,
    resident_attention_limit_bytes: int = DEFAULT_RESIDENT_ATTENTION_LIMIT_BYTES,
) -> tuple[TransformerShape, ...]:
    """Return shapes that require the independent streamed benchmark path."""

    effective_variant = variant or RunVariant()
    return tuple(
        shape
        for shape in shapes
        if plan_workload_execution(
            shape,
            effective_variant,
            resident_attention_limit_bytes=resident_attention_limit_bytes,
        ).is_streamed
    )


def route_eligible_shapes(
    shapes: Sequence[TransformerShape],
    variant: RunVariant | None = None,
    *,
    resident_attention_limit_bytes: int = DEFAULT_RESIDENT_ATTENTION_LIMIT_BYTES,
) -> tuple[TransformerShape, ...]:
    """Return shapes whose current reference path supports formal promotion."""

    effective_variant = variant or RunVariant()
    return tuple(
        shape
        for shape in shapes
        if plan_workload_execution(
            shape,
            effective_variant,
            resident_attention_limit_bytes=resident_attention_limit_bytes,
        ).formal_eligible
    )


__all__ = [
    "DEFAULT_RESIDENT_ATTENTION_LIMIT_BYTES",
    "STREAMED_POLICY_SELECTOR",
    "ExecutionMode",
    "ReferenceKind",
    "WorkloadExecutionPlan",
    "all_benchmark_shapes",
    "effective_protocol",
    "estimate_dense_attention_bytes",
    "plan_workload_execution",
    "resident_benchmark_shapes",
    "route_eligible_shapes",
    "streamed_benchmark_shapes",
]
