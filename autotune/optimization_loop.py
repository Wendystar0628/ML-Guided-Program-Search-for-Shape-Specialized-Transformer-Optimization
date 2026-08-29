"""Bounded multi-round optimization built on shape-group search sweeps."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from .search_sweep import SearchSweep, SearchSweepRequest, SearchSweepResult


@dataclass(frozen=True, slots=True)
class OptimizationLoopPolicy:
    """Stop after a deployment plateau or an absolute iteration limit."""

    no_deployment_patience: int
    max_iterations: int

    def __post_init__(self) -> None:
        for name in ("no_deployment_patience", "max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class OptimizationIteration:
    """One complete or interrupted sweep over the selected workloads."""

    index: int
    search_result: SearchSweepResult
    deployment_updates: int
    shapes_with_search_progress: int
    no_deployment_streak: int


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Compact terminal state for one optimization workflow."""

    iterations_run: int
    total_deployment_updates: int
    no_deployment_streak: int
    stop_reason: str
    exit_code: int


IterationObserver = Callable[[OptimizationIteration], None]


class OptimizationLoop:
    """Repeat full search sweeps while deployment progress remains productive."""

    def __init__(self, search_sweep: SearchSweep | None = None) -> None:
        self.search_sweep = search_sweep or SearchSweep()

    def run(
        self,
        request: SearchSweepRequest,
        policy: OptimizationLoopPolicy,
        *,
        observer: IterationObserver | None = None,
    ) -> OptimizationResult:
        no_deployment_streak = 0
        total_deployment_updates = 0

        for iteration_index in range(1, policy.max_iterations + 1):
            search_result = self.search_sweep.run(
                replace(request, seed=request.seed + iteration_index - 1)
            )
            deployment_updates = sum(
                item.deployment_updated for item in search_result.shape_results
            )
            shapes_with_search_progress = sum(
                item.search_result.made_search_progress
                for item in search_result.shape_results
            )
            total_deployment_updates += deployment_updates

            if search_result.exit_code == 0:
                no_deployment_streak = (
                    0 if deployment_updates else no_deployment_streak + 1
                )

            iteration = OptimizationIteration(
                index=iteration_index,
                search_result=search_result,
                deployment_updates=deployment_updates,
                shapes_with_search_progress=shapes_with_search_progress,
                no_deployment_streak=no_deployment_streak,
            )
            if observer is not None:
                observer(iteration)

            if search_result.exit_code != 0:
                return OptimizationResult(
                    iterations_run=iteration_index,
                    total_deployment_updates=total_deployment_updates,
                    no_deployment_streak=no_deployment_streak,
                    stop_reason=(
                        "interrupted" if search_result.exit_code == 130 else "failed"
                    ),
                    exit_code=search_result.exit_code,
                )
            if deployment_updates == 0 and shapes_with_search_progress == 0:
                return OptimizationResult(
                    iterations_run=iteration_index,
                    total_deployment_updates=total_deployment_updates,
                    no_deployment_streak=no_deployment_streak,
                    stop_reason="search_space_exhausted",
                    exit_code=0,
                )
            if no_deployment_streak >= policy.no_deployment_patience:
                return OptimizationResult(
                    iterations_run=iteration_index,
                    total_deployment_updates=total_deployment_updates,
                    no_deployment_streak=no_deployment_streak,
                    stop_reason="no_deployment_patience",
                    exit_code=0,
                )

        return OptimizationResult(
            iterations_run=policy.max_iterations,
            total_deployment_updates=total_deployment_updates,
            no_deployment_streak=no_deployment_streak,
            stop_reason="max_iterations",
            exit_code=0,
        )


__all__ = [
    "OptimizationIteration",
    "OptimizationLoop",
    "OptimizationLoopPolicy",
    "OptimizationResult",
]
