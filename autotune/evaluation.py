"""Typed measurements shared by the program-search engine and GPU benchmark."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import torch

from solution.config import ConfigSpec
from solution.plan_builder import ConfigRejectedError

from .promotion import (
    PROMOTION_BASE_RATIO,
    PromotionDecision,
    promotion_decision,
)


class Fidelity(StrEnum):
    """Ordered evaluation stages; only SCREEN observations train TPE."""

    SCREEN = "screen"
    ENHANCED = "enhanced"
    FORMAL = "formal"


class EvaluationScope(StrEnum):
    """Resident and Shape-14-style streamed execution scopes."""

    RESIDENT = "resident"
    STREAMED = "streamed"


@dataclass(frozen=True, slots=True)
class FidelityProtocol:
    """Minimal measurement counts for one fidelity level."""

    accuracy_trials: int
    warmup: int
    repeats: int
    rounds: int
    full_logical_batch: bool = True

    def __post_init__(self) -> None:
        values = (self.accuracy_trials, self.warmup, self.repeats, self.rounds)
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise TypeError("fidelity counts must be integers")
        if self.accuracy_trials <= 0 or self.warmup < 0:
            raise ValueError("invalid fidelity accuracy or warmup count")
        if self.repeats <= 0 or self.rounds <= 0:
            raise ValueError("fidelity repeats and rounds must be positive")


RESIDENT_PROTOCOLS: Mapping[Fidelity, FidelityProtocol] = {
    Fidelity.SCREEN: FidelityProtocol(2, 2, 5, 2),
    Fidelity.ENHANCED: FidelityProtocol(3, 5, 20, 2),
    Fidelity.FORMAL: FidelityProtocol(5, 20, 25, 13),
}


STREAMED_PROTOCOLS: Mapping[Fidelity, FidelityProtocol] = {
    Fidelity.SCREEN: FidelityProtocol(1, 1, 3, 1, full_logical_batch=False),
    Fidelity.ENHANCED: FidelityProtocol(2, 2, 5, 2, full_logical_batch=False),
    Fidelity.FORMAL: FidelityProtocol(1, 2, 1, 13, full_logical_batch=True),
}


@dataclass(frozen=True, slots=True)
class ConstraintVector:
    """Continuous feasibility constraints consumed by Optuna TPE."""

    accuracy: float = 0.0
    execution_path: float = 0.0
    runtime: float = 0.0

    def __post_init__(self) -> None:
        values = self.as_tuple()
        if any(not math.isfinite(value) for value in values):
            raise ValueError("constraint values must be finite")

    def as_tuple(self) -> tuple[float, float, float]:
        return (
            float(self.accuracy),
            float(self.execution_path),
            float(self.runtime),
        )

    @property
    def feasible(self) -> bool:
        return all(value <= 0.0 for value in self.as_tuple())

    @property
    def total_violation(self) -> float:
        return sum(max(value, 0.0) for value in self.as_tuple())

    @classmethod
    def from_value(cls, value: object) -> ConstraintVector:
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 3
            or any(isinstance(item, bool) for item in value)
        ):
            raise ValueError("constraints must contain three numeric values")
        return cls(*(float(item) for item in value))


def normalized_accuracy_constraint(
    max_tolerance_ratio: float | None,
) -> float:
    """Convert the official OR-tolerance ratio into ``g_accuracy``."""

    if max_tolerance_ratio is None or not math.isfinite(max_tolerance_ratio):
        return 1.0
    return float(max_tolerance_ratio) - 1.0


def execution_signatures_match(
    expected: Mapping[str, Any] | None,
    actual: Mapping[str, Any] | None,
) -> bool:
    """Compare canonical path signatures without hand-written candidate labels."""

    if expected is None or actual is None:
        return False
    expected_json = json.dumps(
        dict(expected), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    actual_json = json.dumps(
        dict(actual), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return expected_json == actual_json


def classify_infeasible_exception(exc: Exception) -> str | None:
    """Classify only known configuration-domain failures as infeasible.

    Unknown Python, driver, and benchmark-infrastructure exceptions deliberately
    return ``None`` so the caller can preserve the traceback and fail the Trial.
    """

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ConfigRejectedError):
            return "config_rejected"
        if isinstance(current, torch.OutOfMemoryError):
            return "out_of_memory"
        exception_type = type(current)
        if (
            exception_type.__name__ == "OutOfResources"
            and exception_type.__module__.startswith("triton.")
        ):
            return "runtime_resource_exhausted"
        message = str(current).lower()
        candidate_failure_markers = (
            "compiled ffn compilation failed",
            "full-stack compiled forward compilation failed",
            "compiled residual layernorm is ineligible",
            "compiled residual layernorm execution failed",
            "forced fp16 cudnn sdpa is unavailable",
        )
        if any(marker in message for marker in candidate_failure_markers):
            return "candidate_execution_failed"
        if any(
            marker in message
            for marker in (
                "cuda out of memory",
                "cuda error: out of memory",
                "hip out of memory",
            )
        ):
            return "out_of_memory"
        if any(
            marker in message
            for marker in (
                "too many resources requested",
                "launch out of resources",
                "out of resources when launching",
                "out of resource: shared memory",
                "out of resource: registers",
                "exceeds available shared memory",
                "uses too much shared memory",
                "uses too many registers",
                "register allocation failed",
            )
        ):
            return "runtime_resource_exhausted"
        current = current.__cause__ or current.__context__
    return None


@dataclass(frozen=True, slots=True)
class TrialMeasurement:
    """One complete feasible or dynamically infeasible program observation."""

    config_id: str
    fidelity: Fidelity
    scope: EvaluationScope
    objective_ms: float
    constraints: ConstraintVector
    median_ms: float | None = None
    p90_ms: float | None = None
    peak_memory_bytes: int | None = None
    max_tolerance_ratio: float | None = None
    expected_execution_signature: Mapping[str, Any] | None = None
    actual_execution_signature: Mapping[str, Any] | None = None
    failure_kind: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.config_id:
            raise ValueError("config_id must not be empty")
        object.__setattr__(self, "fidelity", Fidelity(self.fidelity))
        object.__setattr__(self, "scope", EvaluationScope(self.scope))
        if not math.isfinite(self.objective_ms) or self.objective_ms <= 0.0:
            raise ValueError("objective_ms must be finite and positive")
        for name in ("median_ms", "p90_ms"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive")
        if self.peak_memory_bytes is not None and self.peak_memory_bytes < 0:
            raise ValueError("peak_memory_bytes must be non-negative")
        if not isinstance(self.constraints, ConstraintVector):
            raise TypeError("constraints must be ConstraintVector")
        if self.constraints.feasible and self.failure_kind is not None:
            raise ValueError("a feasible measurement cannot have failure_kind")
        try:
            json.dumps(
                {
                    "metrics": dict(self.metrics),
                    "expected": self.expected_execution_signature,
                    "actual": self.actual_execution_signature,
                },
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("measurement metadata must be JSON-compatible") from exc

    @property
    def feasible(self) -> bool:
        return self.constraints.feasible

    def to_user_attrs(self, config: ConfigSpec) -> dict[str, Any]:
        """Return compact JSON-compatible attributes for an Optuna Trial."""

        if config.config_id != self.config_id:
            raise ValueError("measurement and config identity disagree")
        return {
            "config": config.to_dict(),
            "config_id": self.config_id,
            "fidelity": self.fidelity.value,
            "scope": self.scope.value,
            "constraints": list(self.constraints.as_tuple()),
            "median_ms": self.median_ms,
            "p90_ms": self.p90_ms,
            "peak_memory_bytes": self.peak_memory_bytes,
            "max_tolerance_ratio": self.max_tolerance_ratio,
            "expected_execution_signature": (
                None
                if self.expected_execution_signature is None
                else dict(self.expected_execution_signature)
            ),
            "actual_execution_signature": (
                None
                if self.actual_execution_signature is None
                else dict(self.actual_execution_signature)
            ),
            "failure_kind": self.failure_kind,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def infeasible(
        cls,
        *,
        config_id: str,
        fidelity: Fidelity,
        scope: EvaluationScope,
        penalty_ms: float,
        constraints: ConstraintVector,
        failure_kind: str,
        metrics: Mapping[str, Any] | None = None,
    ) -> TrialMeasurement:
        if constraints.feasible:
            raise ValueError("an infeasible measurement needs a positive constraint")
        return cls(
            config_id=config_id,
            fidelity=fidelity,
            scope=scope,
            objective_ms=penalty_ms,
            constraints=constraints,
            failure_kind=failure_kind,
            metrics=metrics or {},
        )


@dataclass(frozen=True, slots=True)
class PairedMeasurement:
    """Interleaved challenger-versus-incumbent Formal comparison."""

    incumbent: TrialMeasurement
    challenger: TrialMeasurement
    paired_ratios: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.incumbent.fidelity is not Fidelity.FORMAL:
            raise ValueError("incumbent must use Formal fidelity")
        if self.challenger.fidelity is not Fidelity.FORMAL:
            raise ValueError("challenger must use Formal fidelity")
        if self.incumbent.scope is not self.challenger.scope:
            raise ValueError("paired measurements must use the same scope")
        ratios = tuple(float(value) for value in self.paired_ratios)
        if any(not math.isfinite(value) or value <= 0.0 for value in ratios):
            raise ValueError("paired_ratios must be finite and positive")
        if ratios and promotion_decision(ratios) is PromotionDecision.CONTINUE:
            raise ValueError("paired_ratios must contain a terminal sequential result")
        object.__setattr__(self, "paired_ratios", ratios)

    @property
    def speedup(self) -> float | None:
        """Median paired-block ratio, reported only as an effect size."""

        if not self.paired_ratios:
            return None
        return float(statistics.median(self.paired_ratios))

    @property
    def promotion_wins(self) -> int:
        """Count blocks where the challenger reaches the two-percent target."""

        return sum(ratio >= PROMOTION_BASE_RATIO for ratio in self.paired_ratios)

    @property
    def decision(self) -> PromotionDecision:
        """Return the terminal sequential decision, or reject missing evidence."""

        if not self.paired_ratios:
            return PromotionDecision.REJECT
        return promotion_decision(self.paired_ratios)

    @property
    def promotes(self) -> bool:
        """Apply the sole replacement rule to a complete Formal comparison."""

        return (
            self.incumbent.feasible
            and self.challenger.feasible
            and self.decision is PromotionDecision.PROMOTE
        )


class Evaluator(Protocol):
    """GPU measurement adapter implemented outside the autotune core."""

    def evaluate(
        self,
        config: ConfigSpec,
        fidelity: Fidelity,
    ) -> TrialMeasurement: ...

    def compare(
        self,
        challenger: ConfigSpec,
        incumbent: ConfigSpec,
    ) -> PairedMeasurement: ...


__all__ = [
    "RESIDENT_PROTOCOLS",
    "STREAMED_PROTOCOLS",
    "ConstraintVector",
    "EvaluationScope",
    "Evaluator",
    "Fidelity",
    "FidelityProtocol",
    "PairedMeasurement",
    "TrialMeasurement",
    "classify_infeasible_exception",
    "execution_signatures_match",
    "normalized_accuracy_constraint",
]
