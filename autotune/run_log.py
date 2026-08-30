"""Small append-only run log for search and optimization decisions."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from solution.config import ConfigSpec

from .evaluation import TrialMeasurement

if TYPE_CHECKING:
    from .optimization_loop import OptimizationIteration
    from .search_sweep import ShapeSearchResult


RUN_LOG_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _program_summary(config: ConfigSpec | None) -> dict[str, Any] | None:
    """Keep the two compared programs without copying every searched Trial."""

    if config is None:
        return None
    payload = config.to_dict()
    return {
        "config_id": config.config_id,
        "schema_version": payload["schema_version"],
        "program": payload["program"],
        "schedule": payload["schedule"],
    }


def _failure_message(measurement: TrialMeasurement) -> str | None:
    if measurement.feasible:
        return None
    value = measurement.metrics.get("message")
    if value is None:
        value = measurement.metrics.get("plan_rejection")
    if value is None:
        return None
    if not isinstance(value, str):
        value = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    return value[:500]


def _measurement_summary(measurement: TrialMeasurement) -> dict[str, Any]:
    summary = {
        "config_id": measurement.config_id,
        "median_ms": measurement.median_ms,
        "p90_ms": measurement.p90_ms,
        "max_tolerance_ratio": measurement.max_tolerance_ratio,
        "feasible": measurement.feasible,
        "failure_kind": measurement.failure_kind,
    }
    if not measurement.feasible:
        summary["failure_message"] = _failure_message(measurement)
    return summary


def _decision_outcome(item: ShapeSearchResult) -> str:
    result = item.search_result
    if item.deployment_updated:
        return "published"
    if result.deployment_approved:
        return "approved_unchanged"
    if result.formal_challenger_measurement is not None:
        return "rejected"
    return "not_run"


class SearchRunLog:
    """Append high-value milestones while detailed Trials stay in Optuna."""

    def __init__(
        self,
        *,
        root: Path,
        mode: str,
        target: str,
        request: dict[str, Any],
    ) -> None:
        if mode not in {"search", "optimize"}:
            raise ValueError("mode must be search or optimize")
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        safe_target = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in target
        )
        self.path = root / f"{timestamp}_{mode}_{safe_target}.jsonl"
        self.run_id = self.path.stem
        self._started = time.monotonic()
        self._last_shape = self._started
        self._last_iteration = self._started
        self._iteration = 1
        self._append(
            {
                "event": "run_started",
                "at": _utc_now(),
                "mode": mode,
                "target": target,
                "request": request,
            }
        )

    def _append(self, event: dict[str, Any]) -> None:
        payload = {
            "schema_version": RUN_LOG_SCHEMA_VERSION,
            "run_id": self.run_id,
            **event,
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    def record_shape(self, item: ShapeSearchResult) -> None:
        now = time.monotonic()
        result = item.search_result
        comparison = result.formal_comparison
        formal = result.formal_challenger_measurement
        formal_summary = None
        if formal is not None or comparison is not None:
            incumbent = None if comparison is None else comparison.incumbent
            formal_summary = {
                "incumbent": (
                    None if incumbent is None else _measurement_summary(incumbent)
                ),
                "challenger": (
                    None if formal is None else _measurement_summary(formal)
                ),
                "paired_speedup": (None if comparison is None else comparison.speedup),
                "paired_ratios": (
                    [] if comparison is None else list(comparison.paired_ratios)
                ),
                "promotion_decision": (
                    None if comparison is None else comparison.decision.value
                ),
            }
        stage_timings = result.stage_timings
        timing_seconds = {
            "planning": stage_timings.planning,
            "screen": stage_timings.screen,
            "enhanced": stage_timings.enhanced,
            "formal": stage_timings.formal,
            "total": stage_timings.total,
        }
        budgeted_seconds = (
            stage_timings.screen + stage_timings.enhanced + stage_timings.formal
        )
        self._append(
            {
                "event": "shape_finished",
                "at": _utc_now(),
                "iteration": self._iteration,
                "case_id": item.case_id,
                "elapsed_seconds": round(now - self._last_shape, 3),
                "stop_reason": result.stop_reason,
                "timing_seconds": timing_seconds,
                "budget_seconds": result.budget_seconds,
                "overrun_seconds": max(
                    0.0,
                    budgeted_seconds - result.budget_seconds,
                ),
                "screen": {
                    "branches": result.branch_count,
                    "completed_trials_total": result.completed_level1,
                    "new_trials": result.new_level1_trials,
                    "feasible_trials_total": result.feasible_level1,
                    "best_config_id": result.best_screen_config_id,
                    "best_median_ms": result.best_screen_median_ms,
                    "failure_counts_total": dict(result.screen_failure_counts),
                    "covered_branches": result.covered_branches,
                    "mandatory_coverage_complete": (
                        result.mandatory_coverage_complete
                    ),
                    "scheduler": {
                        "algorithm": result.scheduler_algorithm,
                        "selection_rounds": result.selection_rounds,
                        "pruned_branches": result.pruned_branches,
                        "active_branches": result.active_branches,
                        "space_exhausted": result.level1_space_exhausted,
                    },
                },
                "enhanced": [
                    _measurement_summary(measurement)
                    for measurement in result.enhanced_measurements
                ],
                "decision": {
                    "outcome": _decision_outcome(item),
                    "incumbent": _program_summary(result.incumbent_config),
                    "challenger": _program_summary(result.locked_challenger),
                    "selected_config_id": (
                        None
                        if result.selected_config is None
                        else result.selected_config.config_id
                    ),
                    "formal": formal_summary,
                },
            }
        )
        self._last_shape = now

    def record_iteration(self, iteration: OptimizationIteration) -> None:
        now = time.monotonic()
        self._append(
            {
                "event": "iteration_finished",
                "at": _utc_now(),
                "iteration": iteration.index,
                "elapsed_seconds": round(now - self._last_iteration, 3),
                "deployment_updates": iteration.deployment_updates,
                "shapes_with_level1_progress": (iteration.shapes_with_level1_progress),
                "no_deployment_streak": iteration.no_deployment_streak,
            }
        )
        self._last_iteration = now
        self._iteration = iteration.index + 1

    def finish(
        self,
        *,
        status: str,
        exit_code: int,
        stop_reason: str | None = None,
        iterations: int | None = None,
        total_deployment_updates: int | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "event": "run_finished",
            "at": _utc_now(),
            "status": status,
            "exit_code": exit_code,
            "elapsed_seconds": round(time.monotonic() - self._started, 3),
        }
        if stop_reason is not None:
            event["stop_reason"] = stop_reason
        if iterations is not None:
            event["iterations"] = iterations
        if total_deployment_updates is not None:
            event["total_deployment_updates"] = total_deployment_updates
        self._append(event)

    def fail(self, error: BaseException) -> None:
        self._append(
            {
                "event": "run_failed",
                "at": _utc_now(),
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error)[:500],
                "elapsed_seconds": round(time.monotonic() - self._started, 3),
            }
        )


__all__ = ["SearchRunLog"]
