"""Fair branch screening, cost-aware racing, and fidelity promotion."""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

from optuna.study import Study
from optuna.trial import TrialState

from solution.config import ConfigSpec, portable_streamed_config

from .evaluation import (
    ConstraintVector,
    EvaluationScope,
    Evaluator,
    Fidelity,
    PairedMeasurement,
    TrialMeasurement,
)
from .evaluation_cache import EvaluationCache
from .optuna_backend import CompletedTrial, OptunaBackend, startup_trial_count
from .search_space import (
    DEFAULT_MAX_STRUCTURE_BRANCHES,
    BranchSpace,
    PlanBuilderLike,
    ProgramSearchSpace,
    SearchContext,
)
from .study_storage import SearchStorage, StudyIdentity


@dataclass(frozen=True, slots=True)
class SearchBudget:
    """Time is the primary budget; trial count is an optional hard ceiling."""

    max_seconds: float
    max_trials: int | None = None
    max_structure_branches: int = DEFAULT_MAX_STRUCTURE_BRANCHES
    promote_fraction: float = 0.2
    enhanced_top_k: int = 8

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_seconds, bool)
            or not isinstance(self.max_seconds, (int, float))
            or not math.isfinite(self.max_seconds)
            or self.max_seconds <= 0.0
        ):
            raise ValueError("max_seconds must be finite and positive")
        if self.max_trials is not None and (
            isinstance(self.max_trials, bool)
            or not isinstance(self.max_trials, int)
            or self.max_trials <= 0
        ):
            raise ValueError("max_trials must be a positive integer or null")
        for name in ("enhanced_top_k", "max_structure_branches"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 < self.promote_fraction <= 1.0:
            raise ValueError("promote_fraction must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """Inputs that directly affect search or measured performance."""

    case_id: str
    execution_context: Any
    hardware: Any
    scope: EvaluationScope
    environment: str
    search_identity: str
    enhanced_identity: str
    promotion_identity: str
    budget: SearchBudget
    seed: int = 1234
    incumbent: ConfigSpec | None = None
    warm_starts: tuple[ConfigSpec, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "case_id",
            "environment",
            "search_identity",
            "enhanced_identity",
            "promotion_identity",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        object.__setattr__(self, "scope", EvaluationScope(self.scope))
        if not isinstance(self.budget, SearchBudget):
            raise TypeError("budget must be SearchBudget")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.incumbent is not None and not isinstance(self.incumbent, ConfigSpec):
            raise TypeError("incumbent must be ConfigSpec or None")
        starts = tuple(self.warm_starts)
        if any(not isinstance(config, ConfigSpec) for config in starts):
            raise TypeError("warm_starts must contain ConfigSpec values")
        object.__setattr__(self, "warm_starts", starts)


@dataclass(frozen=True, slots=True)
class SearchPlan:
    """Static plan-builder-pruned branches and their persistent Study identities."""

    request: SearchRequest
    search_space: ProgramSearchSpace
    identities: tuple[StudyIdentity, ...]

    def identity_for(self, branch_id: str) -> StudyIdentity:
        for identity in self.identities:
            if identity.branch_id == branch_id:
                return identity
        raise KeyError(f"no Study identity for branch {branch_id}")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Compact outcome; detailed Level-1 trials remain in Optuna SQLite."""

    incumbent_config: ConfigSpec | None
    selected_config: ConfigSpec | None
    selected_measurement: TrialMeasurement | None
    branch_count: int
    completed_level1: int
    enhanced_measurements: tuple[TrialMeasurement, ...]
    locked_challenger: ConfigSpec | None
    formal_challenger_measurement: TrialMeasurement | None
    formal_comparison: PairedMeasurement | None
    stop_reason: str
    new_level1_trials: int = 0
    feasible_level1: int = 0
    best_screen_config_id: str | None = None
    best_screen_median_ms: float | None = None
    screen_failure_counts: tuple[tuple[str, int], ...] = ()
    level1_space_exhausted: bool = False
    halving_rungs: int = 0
    halving_pruned_branches: int = 0
    halving_active_branches: int = 0

    def __post_init__(self) -> None:
        formal = self.formal_challenger_measurement
        if formal is not None:
            if self.locked_challenger is None:
                raise ValueError("formal measurement requires a locked challenger")
            if formal.config_id != self.locked_challenger.config_id:
                raise ValueError("formal challenger identity disagrees")
            if formal.fidelity is not Fidelity.FORMAL:
                raise ValueError("formal challenger must use Formal fidelity")
        comparison = self.formal_comparison
        if comparison is not None:
            if self.incumbent_config is None or self.locked_challenger is None:
                raise ValueError("paired comparison requires incumbent and challenger")
            if comparison.incumbent.config_id != self.incumbent_config.config_id:
                raise ValueError("paired incumbent identity disagrees")
            if comparison.challenger.config_id != self.locked_challenger.config_id:
                raise ValueError("paired challenger identity disagrees")
            if formal != comparison.challenger:
                raise ValueError("formal challenger measurement disagrees")
        if self.selected_measurement is not None:
            if self.selected_config is None:
                raise ValueError("selected measurement requires a selected config")
            if self.selected_measurement.config_id != self.selected_config.config_id:
                raise ValueError("selected config and measurement identities disagree")
        if self.new_level1_trials < 0:
            raise ValueError("new_level1_trials must not be negative")
        for name in (
            "halving_rungs",
            "halving_pruned_branches",
            "halving_active_branches",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not 0 <= self.feasible_level1 <= self.completed_level1:
            raise ValueError("feasible_level1 must be within completed_level1")
        if self.best_screen_median_ms is not None and (
            not math.isfinite(self.best_screen_median_ms)
            or self.best_screen_median_ms <= 0.0
        ):
            raise ValueError("best_screen_median_ms must be finite and positive")
        if (self.best_screen_config_id is None) != (self.best_screen_median_ms is None):
            raise ValueError("best Screen identity and latency must be set together")
        for measurement in self.enhanced_measurements:
            if measurement.fidelity is not Fidelity.ENHANCED:
                raise ValueError("enhanced measurements must use Enhanced fidelity")
        normalized_failures = tuple(
            sorted(
                (str(kind), int(count)) for kind, count in self.screen_failure_counts
            )
        )
        if any(not kind or count <= 0 for kind, count in normalized_failures):
            raise ValueError("screen failure counts must be positive")
        object.__setattr__(self, "screen_failure_counts", normalized_failures)

    @property
    def deployment_approved(self) -> bool:
        challenger = self.locked_challenger
        measurement = self.formal_challenger_measurement
        if (
            self.stop_reason != "completed"
            or challenger is None
            or measurement is None
            or not measurement.feasible
            or self.selected_config != challenger
        ):
            return False
        if self.incumbent_config is None:
            return self.formal_comparison is None
        comparison = self.formal_comparison
        return comparison is not None and comparison.promotes

    @property
    def made_level1_progress(self) -> bool:
        """Whether this run expanded the target Shape's Screen evidence."""

        return self.new_level1_trials > 0

@dataclass(slots=True)
class _BudgetState:
    new_level1_trials: int = 0

    def can_start(
        self,
        *,
        deadline: float,
        max_trials: int | None,
    ) -> bool:
        if time.monotonic() >= deadline:
            return False
        return max_trials is None or self.new_level1_trials < max_trials


@dataclass(slots=True)
class _RunState:
    studies: dict[str, Study]
    budget: _BudgetState = field(default_factory=_BudgetState)
    halving_rungs: int = 0
    halving_pruned_branches: int = 0


_HALVING_ETA = 2
_SCREEN_EVALUATION_SECONDS = "screen_evaluation_seconds"


def _unique_configs(configs: list[ConfigSpec]) -> tuple[ConfigSpec, ...]:
    values: list[ConfigSpec] = []
    seen: set[str] = set()
    for config in configs:
        if config.config_id in seen:
            continue
        seen.add(config.config_id)
        values.append(config)
    return tuple(values)


def _unseen_branches(
    plan: SearchPlan,
    state: _RunState,
    backend: OptunaBackend,
) -> tuple[BranchSpace, ...]:
    """Return branches with at least one terminal configuration still unmeasured."""

    return tuple(
        branch
        for branch in plan.search_space.branches
        if len(
            backend.terminal_config_ids(
                state.studies[branch.branch_id],
                branch,
            )
        )
        < branch.cardinality
    )


def _screening_trial_target(branch: BranchSpace) -> int:
    return min(3, branch.cardinality)


def _completed_config_ids(
    backend: OptunaBackend,
    study: Study,
    branch: BranchSpace,
) -> set[str]:
    return {
        completed.config.config_id
        for completed in backend.completed_trials(study, branch)
    }


def _branch_screen_seconds(
    backend: OptunaBackend,
    study: Study,
    branch: BranchSpace,
) -> float:
    """Return persisted wall time for compatible Screen observations."""

    total = 0.0
    for completed in backend.completed_trials(study, branch):
        measurement = completed.measurement
        raw_cost = measurement.metrics.get(_SCREEN_EVALUATION_SECONDS)
        if (
            not isinstance(raw_cost, bool)
            and isinstance(raw_cost, (int, float))
            and math.isfinite(float(raw_cost))
            and float(raw_cost) > 0.0
        ):
            total += float(raw_cost)
    return total


def _branch_rank(
    backend: OptunaBackend,
    study: Study,
    branch: BranchSpace,
) -> tuple[int, float, str]:
    """Rank by the best feasible Screen median, never by extrapolation."""

    latencies = [
        measurement.median_ms or measurement.objective_ms
        for completed in backend.completed_trials(study, branch)
        if (measurement := completed.measurement).feasible
    ]
    if not latencies:
        return (1, math.inf, branch.branch_id)
    # Each objective is already the median of repeated GPU timings. Using the
    # best observed median preserves TPE's objective without inventing a future
    # improvement curve from sparse branch history.
    return (0, min(latencies), branch.branch_id)


def _screen_failure_kind(measurement: TrialMeasurement) -> str:
    """Return one useful primary reason for an infeasible Screen result."""

    if measurement.constraints.accuracy > 0.0:
        return "accuracy_constraint"
    if measurement.constraints.execution_path > 0.0:
        return "execution_path_constraint"
    if measurement.failure_kind not in {None, "constraint_violation"}:
        return measurement.failure_kind
    return "runtime_constraint"


class SearchEngine:
    """Run generated program search without owning GPU measurement mechanics."""

    def __init__(
        self,
        *,
        storage: SearchStorage,
        evaluator: Evaluator,
        plan_builder: PlanBuilderLike,
        failure_penalty_ms: float = 1_000_000_000.0,
    ) -> None:
        if not isinstance(storage, SearchStorage):
            raise TypeError("storage must be SearchStorage")
        if not math.isfinite(failure_penalty_ms) or failure_penalty_ms <= 0.0:
            raise ValueError("failure_penalty_ms must be finite and positive")
        self.storage = storage
        self.evaluation_cache = EvaluationCache(storage.database_path)
        self.evaluator = evaluator
        self.plan_builder = plan_builder
        self.failure_penalty_ms = float(failure_penalty_ms)

    def plan(self, request: SearchRequest) -> SearchPlan:
        """Generate legal high-level branches without running a benchmark."""

        context = SearchContext(
            execution_context=request.execution_context,
            scope=request.scope.value,
            hardware=request.hardware,
        )
        required_configs: tuple[ConfigSpec, ...] = (
            (portable_streamed_config(microbatch_size=1),)
            if request.scope is EvaluationScope.STREAMED
            else ()
        )
        if request.incumbent is not None:
            required_configs += (request.incumbent,)
        required_configs = _unique_configs([*required_configs, *request.warm_starts])
        search_space = ProgramSearchSpace(
            plan_builder=self.plan_builder,
            context=context,
            max_branches=request.budget.max_structure_branches,
            required_configs=required_configs,
            seed=request.seed,
        )
        identities = tuple(
            StudyIdentity(
                case_id=request.case_id,
                branch_id=branch.branch_id,
                environment=request.environment,
                search_identity=request.search_identity,
            )
            for branch in search_space.branches
        )
        return SearchPlan(
            request=request,
            search_space=search_space,
            identities=identities,
        )

    def run(self, request: SearchRequest) -> SearchResult:
        """Execute fair screening, adaptive TPE, and fidelity promotion."""

        plan = self.plan(request)
        backend = OptunaBackend(self.storage, seed=request.seed)
        run_state = _RunState(
            studies={
                branch.branch_id: backend.create_study(
                    plan.identity_for(branch.branch_id),
                    n_startup_trials=startup_trial_count(branch),
                )
                for branch in plan.search_space.branches
            }
        )
        start = time.monotonic()
        budget = request.budget
        screen_deadline = start + float(budget.max_seconds) * 0.20
        level1_deadline = start + float(budget.max_seconds) * 0.65
        enhanced_deadline = start + float(budget.max_seconds) * 0.82
        final_deadline = start + float(budget.max_seconds)
        promoted: tuple[ConfigSpec, ...] = ()
        enhanced: tuple[tuple[ConfigSpec, TrialMeasurement], ...] = ()
        locked_challenger: ConfigSpec | None = None
        formal_challenger_measurement: TrialMeasurement | None = None
        formal_comparison: PairedMeasurement | None = None
        active: tuple[BranchSpace, ...] = ()

        try:
            self._enqueue_initial_configs(plan, run_state, backend)
            historical_promotions = self._select_promotions(
                plan,
                run_state,
                backend,
            )
            has_historical_promotions = bool(historical_promotions)
            new_trials_before = run_state.budget.new_level1_trials

            screen_complete = self._screen_structures(
                plan,
                run_state,
                backend,
                deadline=screen_deadline,
            )
            fresh_deadline = (
                screen_deadline if has_historical_promotions else level1_deadline
            )

            if (
                _unseen_branches(plan, run_state, backend)
                and run_state.budget.new_level1_trials == new_trials_before
            ):
                self._force_one_new_level1(
                    plan,
                    run_state,
                    backend,
                    deadline=fresh_deadline,
                )

            if time.monotonic() < fresh_deadline:
                active = self._run_successive_halving(
                    plan,
                    run_state,
                    backend,
                    deadline=fresh_deadline,
                )

            ranked_screen = self._ranked_level1(
                run_state,
                backend,
                plan.search_space.branches,
            )
            promoted = self._select_promotions(
                plan,
                run_state,
                backend,
            )
            if not ranked_screen:
                selected_config = None
                selected_measurement = None
                stop_reason = (
                    "no_feasible_screen"
                    if screen_complete
                    else "insufficient_screen_budget"
                )
            else:
                enhanced = self._evaluate_promotions(
                    request,
                    promoted,
                    deadline=enhanced_deadline,
                )
                locked_challenger = self._lock_challenger(
                    enhanced,
                    incumbent=request.incumbent,
                )
                (
                    selected_config,
                    selected_measurement,
                    formal_challenger_measurement,
                    formal_comparison,
                ) = self._run_formal(
                    request,
                    locked_challenger,
                    deadline=final_deadline,
                )
                if (
                    locked_challenger is not None
                    and formal_challenger_measurement is not None
                    and not (
                        selected_config == locked_challenger
                        and (
                            formal_comparison is None
                            or formal_comparison.promotes
                        )
                    )
                ):
                    self.storage.record_challenger_attempt(
                        case_id=request.case_id,
                        environment=request.environment,
                        incumbent_id=(
                            None
                            if request.incumbent is None
                            else request.incumbent.config_id
                        ),
                        challenger_id=locked_challenger.config_id,
                        promotion_identity=request.promotion_identity,
                    )
                stop_reason = (
                    "no_feasible_screen"
                    if not self._ranked_level1(
                        run_state,
                        backend,
                        plan.search_space.branches,
                    )
                    else "no_feasible_enhanced"
                    if locked_challenger is None
                    else "formal_not_run"
                    if formal_challenger_measurement is None
                    else "no_feasible_formal"
                    if selected_config is None
                    else "completed"
                )
        except KeyboardInterrupt:
            # Level-1 observations remain available in Optuna, but an interrupted
            # search never presents an unverified partial result as deployable.
            selected_config = request.incumbent
            selected_measurement = None
            stop_reason = "interrupted"
        except Exception:
            raise

        completed_level1 = sum(
            len(
                _completed_config_ids(
                    backend,
                    run_state.studies[branch.branch_id],
                    branch,
                )
            )
            for branch in plan.search_space.branches
        )
        level1_space_exhausted = all(
            len(
                backend.terminal_config_ids(
                    run_state.studies[branch.branch_id],
                    branch,
                )
            )
            >= branch.cardinality
            for branch in plan.search_space.branches
        )
        (
            feasible_level1,
            best_screen_config_id,
            best_screen_median_ms,
            screen_failure_counts,
        ) = self._screen_summary(plan, run_state, backend)
        return SearchResult(
            incumbent_config=request.incumbent,
            selected_config=selected_config,
            selected_measurement=selected_measurement,
            branch_count=len(plan.search_space.branches),
            completed_level1=completed_level1,
            enhanced_measurements=tuple(measurement for _, measurement in enhanced),
            locked_challenger=locked_challenger,
            formal_challenger_measurement=formal_challenger_measurement,
            formal_comparison=formal_comparison,
            stop_reason=stop_reason,
            new_level1_trials=run_state.budget.new_level1_trials,
            feasible_level1=feasible_level1,
            best_screen_config_id=best_screen_config_id,
            best_screen_median_ms=best_screen_median_ms,
            screen_failure_counts=screen_failure_counts,
            level1_space_exhausted=level1_space_exhausted,
            halving_rungs=run_state.halving_rungs,
            halving_pruned_branches=run_state.halving_pruned_branches,
            halving_active_branches=len(active),
        )

    @staticmethod
    def _screen_summary(
        plan: SearchPlan,
        state: _RunState,
        backend: OptunaBackend,
    ) -> tuple[int, str | None, float | None, tuple[tuple[str, int], ...]]:
        """Summarize Screen evidence without duplicating individual Trials."""

        feasible: list[CompletedTrial] = []
        failures: Counter[str] = Counter()
        for branch in plan.search_space.branches:
            study = state.studies[branch.branch_id]
            completed_trials = backend.completed_trials(study, branch)
            best_feasible: dict[str, CompletedTrial] = {}
            for completed in completed_trials:
                measurement = completed.measurement
                if measurement.feasible:
                    previous = best_feasible.get(completed.config.config_id)
                    if (
                        previous is None
                        or measurement.objective_ms < previous.measurement.objective_ms
                    ):
                        best_feasible[completed.config.config_id] = completed
                    continue
                failures[_screen_failure_kind(measurement)] += 1
            feasible.extend(best_feasible.values())
            for trial in study.get_trials(
                deepcopy=False,
                states=(TrialState.FAIL,),
            ):
                if "duplicate_config_id" in trial.user_attrs:
                    failures["duplicate_proposal"] += 1
                elif error_type := trial.user_attrs.get("infrastructure_error"):
                    failures[f"infrastructure:{error_type}"] += 1
                else:
                    failures["failed_trial"] += 1

        best = min(
            feasible,
            key=lambda item: item.measurement.objective_ms,
            default=None,
        )
        return (
            len(feasible),
            None if best is None else best.config.config_id,
            (
                None
                if best is None
                else best.measurement.median_ms or best.measurement.objective_ms
            ),
            tuple(sorted(failures.items())),
        )

    def _enqueue_initial_configs(
        self,
        plan: SearchPlan,
        state: _RunState,
        backend: OptunaBackend,
    ) -> None:
        request = plan.request
        seeds: list[tuple[str, ConfigSpec]] = []
        if request.scope is EvaluationScope.STREAMED:
            seeds.append(
                ("portable_streamed", portable_streamed_config(microbatch_size=1))
            )
        if request.incumbent is not None:
            seeds.append(("incumbent", request.incumbent))
        seeds.extend(("warm_start", config) for config in request.warm_starts)
        for branch in plan.search_space.branches:
            study = state.studies[branch.branch_id]
            backend.enqueue(
                study,
                branch,
                branch.default_config(),
                source="structure_default",
            )
            # The default establishes a reproducible branch baseline. Optuna's
            # startup sampler chooses the remaining screening points.
            for source, config in seeds:
                backend.enqueue(study, branch, config, source=source)

    def _screen_structures(
        self,
        plan: SearchPlan,
        state: _RunState,
        backend: OptunaBackend,
        *,
        deadline: float,
    ) -> bool:
        mandatory = [
            branch
            for branch in plan.search_space.branches
            if branch.branch_id in plan.search_space.mandatory_branch_ids
        ]
        coverage_complete = True
        for branch in mandatory:
            study = state.studies[branch.branch_id]
            target = _screening_trial_target(branch)
            completed_ids = _completed_config_ids(backend, study, branch)
            while len(completed_ids) < target:
                if not state.budget.can_start(
                    deadline=deadline,
                    max_trials=plan.request.budget.max_trials,
                ):
                    return False
                terminal_before = backend.terminal_config_ids(study, branch)
                self._ask_and_measure(plan, state, backend, branch)
                completed_ids = _completed_config_ids(backend, study, branch)
                terminal_after = backend.terminal_config_ids(study, branch)
                if terminal_after == terminal_before:
                    coverage_complete = False
                    break
                if len(terminal_after) >= branch.cardinality:
                    break
            if len(completed_ids) < target:
                coverage_complete = False
        return coverage_complete

    def _run_successive_halving(
        self,
        plan: SearchPlan,
        state: _RunState,
        backend: OptunaBackend,
        *,
        deadline: float,
    ) -> tuple[BranchSpace, ...]:
        """Race branches in wall-time rungs and keep the fastest half."""

        active = _unseen_branches(plan, state, backend)
        if not active:
            return ()

        budget = plan.request.budget
        remaining_seconds = max(0.0, deadline - time.monotonic())
        planned_sizes: list[int] = []
        size = len(active)
        while True:
            planned_sizes.append(size)
            if size == 1:
                break
            size = math.ceil(size / _HALVING_ETA)
        cost_units = sum(
            rung_size * (_HALVING_ETA**rung)
            for rung, rung_size in enumerate(planned_sizes)
        )
        base_quota_seconds = remaining_seconds / max(1, cost_units)

        for rung, _ in enumerate(planned_sizes):
            if not active or not state.budget.can_start(
                deadline=deadline,
                max_trials=budget.max_trials,
            ):
                break
            quota_seconds = base_quota_seconds * (_HALVING_ETA**rung)
            ordered = list(active)
            random.Random(
                f"{plan.request.seed}:{plan.request.case_id}:halving:{rung}"
            ).shuffle(ordered)
            progressed = False
            complete_rung = True
            stalled: set[str] = set()
            for branch in ordered:
                study = state.studies[branch.branch_id]
                spent_seconds = 0.0
                while spent_seconds < quota_seconds:
                    if not state.budget.can_start(
                        deadline=deadline,
                        max_trials=budget.max_trials,
                    ):
                        complete_rung = False
                        break
                    cost_before = _branch_screen_seconds(backend, study, branch)
                    started = time.monotonic()
                    if not self._ask_and_measure(plan, state, backend, branch):
                        stalled.add(branch.branch_id)
                        break
                    progressed = True
                    cost_after = _branch_screen_seconds(backend, study, branch)
                    measured_cost = cost_after - cost_before
                    if measured_cost <= 0.0:
                        measured_cost = max(time.monotonic() - started, 1e-9)
                    spent_seconds += measured_cost
                if not complete_rung:
                    break
            if stalled:
                active = tuple(
                    branch for branch in active if branch.branch_id not in stalled
                )
            if not progressed:
                break
            if not complete_rung:
                # Partial rungs produce useful Trials but never eliminate a
                # branch on unequal allocation. A later invocation starts a new
                # race from all branches that still contain unseen configs.
                break

            state.halving_rungs += 1
            ranked = sorted(
                active,
                key=lambda branch: _branch_rank(
                    backend,
                    state.studies[branch.branch_id],
                    branch,
                ),
            )
            survivor_count = max(1, math.ceil(len(ranked) / _HALVING_ETA))
            survivor_ids = {branch.branch_id for branch in ranked[:survivor_count]}
            state.halving_pruned_branches += sum(
                branch.branch_id not in survivor_ids
                and len(
                    backend.terminal_config_ids(
                        state.studies[branch.branch_id],
                        branch,
                    )
                )
                < branch.cardinality
                for branch in ranked
            )
            active = tuple(
                branch
                for branch in ranked[:survivor_count]
                if len(
                    backend.terminal_config_ids(
                        state.studies[branch.branch_id],
                        branch,
                    )
                )
                < branch.cardinality
            )
        return active

    def _force_one_new_level1(
        self,
        plan: SearchPlan,
        state: _RunState,
        backend: OptunaBackend,
        *,
        deadline: float,
    ) -> bool:
        """Measure one least-sampled unseen branch before history-based pruning."""

        if not state.budget.can_start(
            deadline=deadline,
            max_trials=plan.request.budget.max_trials,
        ):
            return False

        branches = _unseen_branches(plan, state, backend)
        if not branches:
            return False

        def terminal_count(branch: BranchSpace) -> int:
            return len(
                backend.terminal_config_ids(
                    state.studies[branch.branch_id],
                    branch,
                )
            )

        branch = min(
            branches,
            key=lambda item: (terminal_count(item), item.branch_id),
        )
        terminal_before = terminal_count(branch)
        if not self._ask_and_measure(plan, state, backend, branch):
            return False
        return terminal_count(branch) > terminal_before

    def _ask_and_measure(
        self,
        plan: SearchPlan,
        state: _RunState,
        backend: OptunaBackend,
        branch: BranchSpace,
    ) -> bool:
        study = state.studies[branch.branch_id]
        seen = backend.terminal_config_ids(study, branch)
        if len(seen) >= branch.cardinality:
            return False

        trial, config = backend.ask(study, branch)
        if config.config_id in seen:
            backend.reject_duplicate(study, trial, config.config_id)
            config = self._uniform_unseen_config(
                plan,
                state,
                backend,
                branch,
            )
            if config is None:
                return False
            try:
                measurement = self._measure_screen(plan.request, config)
                backend.record_completed(
                    study,
                    branch,
                    config,
                    measurement,
                    source="uniform_unseen_fallback",
                )
                return True
            finally:
                state.budget.new_level1_trials += 1

        try:
            measurement = self._measure_screen(plan.request, config)
            backend.tell(study, trial, config, measurement)
            return True
        except Exception as exc:
            backend.fail_infrastructure(study, trial, config, exc)
            raise
        finally:
            state.budget.new_level1_trials += 1

    @staticmethod
    def _uniform_unseen_config(
        plan: SearchPlan,
        state: _RunState,
        backend: OptunaBackend,
        branch: BranchSpace,
    ) -> ConfigSpec | None:
        """Sample exactly once from the branch points not yet measured."""

        study = state.studies[branch.branch_id]
        seen_indices = sorted(
            index
            for config in backend.terminal_configs(study, branch)
            if (index := branch.index_for(config)) is not None
        )
        seen_indices = sorted(set(seen_indices))
        remaining = branch.cardinality - len(seen_indices)
        if remaining <= 0:
            return None
        rng = random.Random(
            f"{plan.request.seed}:{branch.branch_id}:{len(seen_indices)}"
        )
        candidate_index = rng.randrange(remaining)
        for seen_index in seen_indices:
            if seen_index > candidate_index:
                break
            candidate_index += 1
        return branch.config_at(candidate_index)

    def _measure_screen(
        self,
        request: SearchRequest,
        config: ConfigSpec,
    ) -> TrialMeasurement:
        started_ns = time.monotonic_ns()
        build_result = self.plan_builder.evaluate(
            config,
            request.execution_context,
            request.hardware,
        )
        if not build_result.accepted:
            rejection = getattr(build_result, "rejection", None)
            details = (
                rejection.to_dict()
                if rejection is not None and hasattr(rejection, "to_dict")
                else {"reason": "plan_rejected"}
            )
            measurement = TrialMeasurement.infeasible(
                config_id=config.config_id,
                fidelity=Fidelity.SCREEN,
                scope=request.scope,
                penalty_ms=self.failure_penalty_ms,
                constraints=ConstraintVector(runtime=1.0),
                failure_kind="plan_rejection",
                metrics={"plan_rejection": details},
            )
        else:
            measurement = self.evaluator.evaluate(config, Fidelity.SCREEN)
            self._validate_measurement(
                measurement,
                config=config,
                fidelity=Fidelity.SCREEN,
                scope=request.scope,
            )
        return replace(
            measurement,
            metrics={
                **measurement.metrics,
                _SCREEN_EVALUATION_SECONDS: max(
                    1,
                    time.monotonic_ns() - started_ns,
                )
                / 1_000_000_000.0,
            },
        )

    def _select_promotions(
        self,
        plan: SearchPlan,
        state: _RunState,
        backend: OptunaBackend,
    ) -> tuple[ConfigSpec, ...]:
        ranked = self._ranked_level1(
            state,
            backend,
            plan.search_space.branches,
        )
        if not ranked:
            return ()
        incumbent_id = (
            None if plan.request.incumbent is None else plan.request.incumbent.config_id
        )
        attempted = self.storage.attempted_challenger_ids(
            case_id=plan.request.case_id,
            environment=plan.request.environment,
            incumbent_id=incumbent_id,
            promotion_identity=plan.request.promotion_identity,
        )
        eligible = tuple(
            config
            for config in _unique_configs([completed.config for completed in ranked])
            if config.config_id != incumbent_id and config.config_id not in attempted
        )
        if not eligible:
            return ()
        budget = plan.request.budget
        target = min(
            max(1, math.ceil(len(eligible) * budget.promote_fraction)),
            budget.enhanced_top_k,
        )
        return eligible[:target]

    def _evaluate_promotions(
        self,
        request: SearchRequest,
        promoted: tuple[ConfigSpec, ...],
        *,
        deadline: float,
    ) -> tuple[tuple[ConfigSpec, TrialMeasurement], ...]:
        values: list[tuple[ConfigSpec, TrialMeasurement]] = []
        for config in promoted:
            cached = self.evaluation_cache.get(
                case_id=request.case_id,
                evidence_identity=request.enhanced_identity,
                config_id=config.config_id,
                fidelity=Fidelity.ENHANCED,
            )
            if cached is not None:
                self._validate_measurement(
                    cached,
                    config=config,
                    fidelity=Fidelity.ENHANCED,
                    scope=request.scope,
                )
                values.append((config, cached))
                continue
            if time.monotonic() >= deadline:
                break
            measurement = self.evaluator.evaluate(config, Fidelity.ENHANCED)
            self._validate_measurement(
                measurement,
                config=config,
                fidelity=Fidelity.ENHANCED,
                scope=request.scope,
            )
            self.evaluation_cache.put(
                case_id=request.case_id,
                evidence_identity=request.enhanced_identity,
                measurement=measurement,
            )
            values.append((config, measurement))
        return tuple(values)

    @staticmethod
    def _lock_challenger(
        enhanced: tuple[tuple[ConfigSpec, TrialMeasurement], ...],
        *,
        incumbent: ConfigSpec | None,
    ) -> ConfigSpec | None:
        """Lock the fastest feasible Enhanced challenger before Formal testing."""

        incumbent_id = None if incumbent is None else incumbent.config_id
        eligible = (
            (config, measurement)
            for config, measurement in enhanced
            if measurement.feasible and config.config_id != incumbent_id
        )
        winner = min(eligible, key=lambda item: item[1].objective_ms, default=None)
        return None if winner is None else winner[0]

    def _run_formal(
        self,
        request: SearchRequest,
        challenger: ConfigSpec | None,
        *,
        deadline: float,
    ) -> tuple[
        ConfigSpec | None,
        TrialMeasurement | None,
        TrialMeasurement | None,
        PairedMeasurement | None,
    ]:
        incumbent = request.incumbent
        if challenger is None or time.monotonic() >= deadline:
            return incumbent, None, None, None

        if incumbent is None:
            measurement = self.evaluator.evaluate(challenger, Fidelity.FORMAL)
            self._validate_measurement(
                measurement,
                config=challenger,
                fidelity=Fidelity.FORMAL,
                scope=request.scope,
            )
            if measurement.feasible:
                return challenger, measurement, measurement, None
            return None, None, measurement, None

        comparison = self.evaluator.compare(challenger, incumbent)
        self._validate_measurement(
            comparison.incumbent,
            config=incumbent,
            fidelity=Fidelity.FORMAL,
            scope=request.scope,
        )
        self._validate_measurement(
            comparison.challenger,
            config=challenger,
            fidelity=Fidelity.FORMAL,
            scope=request.scope,
        )
        if comparison.promotes:
            return (
                challenger,
                comparison.challenger,
                comparison.challenger,
                comparison,
            )
        if comparison.incumbent.feasible:
            return (
                incumbent,
                comparison.incumbent,
                comparison.challenger,
                comparison,
            )
        return None, None, comparison.challenger, comparison

    @staticmethod
    def _validate_measurement(
        measurement: TrialMeasurement,
        *,
        config: ConfigSpec,
        fidelity: Fidelity,
        scope: EvaluationScope,
    ) -> None:
        if not isinstance(measurement, TrialMeasurement):
            raise TypeError("evaluator must return TrialMeasurement")
        if measurement.config_id != config.config_id:
            raise ValueError("evaluator returned a different config_id")
        if measurement.fidelity is not fidelity:
            raise ValueError("evaluator returned the wrong fidelity")
        if measurement.scope is not scope:
            raise ValueError("evaluator returned the wrong scope")

    @staticmethod
    def _ranked_level1(
        state: _RunState,
        backend: OptunaBackend,
        branches: tuple[BranchSpace, ...],
    ) -> list[CompletedTrial]:
        values = [
            completed
            for branch in branches
            for completed in backend.feasible_trials(
                state.studies[branch.branch_id], branch
            )
        ]
        values.sort(key=lambda item: item.measurement.objective_ms)
        return values


__all__ = [
    "SearchBudget",
    "SearchEngine",
    "SearchPlan",
    "SearchRequest",
    "SearchResult",
]
