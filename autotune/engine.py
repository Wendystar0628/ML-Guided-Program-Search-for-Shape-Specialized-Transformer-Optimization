"""Branch screening, constraint-aware TPE, and multi-fidelity promotion."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from optuna.study import Study

from solution.config import ConfigSpec, portable_streamed_config

from .evaluator import (
    ConstraintVector,
    EvaluationScope,
    Evaluator,
    Fidelity,
    PairedMeasurement,
    TrialMeasurement,
)
from .optuna_store import CompletedTrial, OptunaBackend, startup_trial_count
from .space import (
    DEFAULT_MAX_STRUCTURE_BRANCHES,
    BranchSpace,
    PlanBuilderLike,
    ProgramSearchSpace,
    SearchContext,
)
from .storage import SearchStorage, StudyIdentity


@dataclass(frozen=True, slots=True)
class SearchBudget:
    """Time is the primary budget; trial count is an optional hard ceiling."""

    max_seconds: float
    max_trials: int | None = None
    max_structure_branches: int = DEFAULT_MAX_STRUCTURE_BRANCHES
    min_trials_per_branch: int = 5
    survivor_count: int = 3
    promote_fraction: float = 0.2
    enhanced_top_k: int = 8
    local_top_k: int = 3

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
        for name in (
            "min_trials_per_branch",
            "survivor_count",
            "enhanced_top_k",
            "local_top_k",
            "max_structure_branches",
        ):
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
    budget: SearchBudget
    seed: int = 1234
    incumbent: ConfigSpec | None = None
    warm_starts: tuple[ConfigSpec, ...] = ()

    def __post_init__(self) -> None:
        for name in ("case_id", "environment"):
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
    enhanced_configs: tuple[ConfigSpec, ...]
    locked_challenger: ConfigSpec | None
    formal_challenger_measurement: TrialMeasurement | None
    formal_comparison: PairedMeasurement | None
    stop_reason: str

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


def _unique_configs(configs: list[ConfigSpec]) -> tuple[ConfigSpec, ...]:
    values: list[ConfigSpec] = []
    seen: set[str] = set()
    for config in configs:
        if config.config_id in seen:
            continue
        seen.add(config.config_id)
        values.append(config)
    return tuple(values)


def _screening_trial_target(branch: BranchSpace) -> int:
    return min(3, branch.cardinality)


def _screening_trial_total(branches: tuple[BranchSpace, ...]) -> int:
    return sum(_screening_trial_target(branch) for branch in branches)


_MAX_CONSECUTIVE_DUPLICATE_ASKS = 8


def _completed_config_ids(
    backend: OptunaBackend,
    study: Study,
    branch: BranchSpace,
) -> set[str]:
    return {
        completed.config.config_id
        for completed in backend.completed_trials(study, branch)
    }


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
        )
        screening_trial_total = _screening_trial_total(search_space.branches)
        if (
            request.budget.max_trials is not None
            and request.budget.max_trials < screening_trial_total
        ):
            raise ValueError(
                "max_trials is smaller than fair structure screening coverage: "
                f"need at least {screening_trial_total} trials"
            )
        identities = tuple(
            StudyIdentity(
                case_id=request.case_id,
                branch_id=branch.branch_id,
                environment=request.environment,
            )
            for branch in search_space.branches
        )
        return SearchPlan(
            request=request,
            search_space=search_space,
            identities=identities,
        )

    def run(self, request: SearchRequest) -> SearchResult:
        """Execute screening, TPE, local refinement, and promotion in order."""

        plan = self.plan(request)
        backend = OptunaBackend(self.storage, seed=request.seed)
        branch_budgets = {
            branch.branch_id: self._expected_branch_trial_budget(plan, branch)
            for branch in plan.search_space.branches
        }
        run_state = _RunState(
            studies={
                branch.branch_id: backend.create_study(
                    plan.identity_for(branch.branch_id),
                    n_startup_trials=startup_trial_count(
                        branch,
                        branch_budget=branch_budgets[branch.branch_id],
                        scope=request.scope,
                    ),
                )
                for branch in plan.search_space.branches
            }
        )
        start = time.monotonic()
        budget = request.budget
        screen_deadline = start + float(budget.max_seconds) * 0.20
        tpe_deadline = start + float(budget.max_seconds) * 0.55
        level1_deadline = start + float(budget.max_seconds) * 0.65
        enhanced_deadline = start + float(budget.max_seconds) * 0.82
        final_deadline = start + float(budget.max_seconds)
        promoted: tuple[ConfigSpec, ...] = ()
        enhanced: tuple[tuple[ConfigSpec, TrialMeasurement], ...] = ()
        locked_challenger: ConfigSpec | None = None
        formal_challenger_measurement: TrialMeasurement | None = None
        formal_comparison: PairedMeasurement | None = None

        try:
            self._enqueue_initial_configs(plan, run_state, backend)
            screen_complete = self._screen_structures(
                plan,
                run_state,
                backend,
                deadline=screen_deadline,
            )
            if not screen_complete:
                selected_config = None
                selected_measurement = None
                stop_reason = "insufficient_screen_budget"
            else:
                survivors = self._select_survivors(plan, run_state, backend)
                self._run_tpe(
                    plan,
                    run_state,
                    backend,
                    survivors,
                    deadline=tpe_deadline,
                )
                self._run_local_neighbourhood(
                    plan,
                    run_state,
                    backend,
                    survivors,
                    deadline=level1_deadline,
                )
                promoted = self._select_promotions(
                    plan,
                    run_state,
                    backend,
                    survivors,
                )
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
                stop_reason = (
                    "no_feasible_screen"
                    if not survivors
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
            len(backend.completed_trials(run_state.studies[branch.branch_id], branch))
            for branch in plan.search_space.branches
        )
        return SearchResult(
            incumbent_config=request.incumbent,
            selected_config=selected_config,
            selected_measurement=selected_measurement,
            branch_count=len(plan.search_space.branches),
            completed_level1=completed_level1,
            enhanced_configs=promoted,
            locked_challenger=locked_challenger,
            formal_challenger_measurement=formal_challenger_measurement,
            formal_comparison=formal_comparison,
            stop_reason=stop_reason,
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
            for index, config in enumerate(branch.representative_configs(limit=3)):
                backend.enqueue(
                    study,
                    branch,
                    config,
                    source=f"structure_representative_{index}",
                )
            # Representatives define the fair minimum coverage. Incumbent and
            # cross-shape seeds follow them and guide branch-local TPE without
            # displacing structural screening.
            for source, config in seeds:
                backend.enqueue(study, branch, config, source=source)

    @staticmethod
    def _expected_branch_trial_budget(
        plan: SearchPlan,
        branch: BranchSpace,
    ) -> int:
        budget = plan.request.budget
        screening_target = _screening_trial_target(branch)
        if budget.max_trials is None:
            expected = max(screening_target, budget.min_trials_per_branch)
        else:
            screening_total = _screening_trial_total(plan.search_space.branches)
            remaining = max(0, budget.max_trials - screening_total)
            survivor_slots = max(
                1,
                min(budget.survivor_count, len(plan.search_space.branches)),
            )
            expected = screening_target + remaining // survivor_slots
        return min(branch.cardinality, max(1, expected))

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
        optional = [
            branch
            for branch in plan.search_space.branches
            if branch.branch_id not in plan.search_space.mandatory_branch_ids
        ]
        for branch in (*mandatory, *optional):
            study = state.studies[branch.branch_id]
            target = _screening_trial_target(branch)
            completed_ids = _completed_config_ids(backend, study, branch)
            while len(completed_ids) < target:
                if not state.budget.can_start(
                    deadline=deadline,
                    max_trials=plan.request.budget.max_trials,
                ):
                    return False
                self._ask_and_measure(plan, state, backend, branch)
                completed_ids = _completed_config_ids(backend, study, branch)
        return True

    def _select_survivors(
        self,
        plan: SearchPlan,
        state: _RunState,
        backend: OptunaBackend,
    ) -> tuple[BranchSpace, ...]:
        ranked: list[tuple[CompletedTrial, BranchSpace]] = []
        for branch in plan.search_space.branches:
            study = state.studies[branch.branch_id]
            completed_ids = _completed_config_ids(backend, study, branch)
            if len(completed_ids) < _screening_trial_target(branch):
                continue
            best = backend.best_feasible(study, branch)
            if best is not None:
                ranked.append((best, branch))
        ranked.sort(key=lambda item: item[0].measurement.objective_ms)
        survivors = [
            branch for _, branch in ranked[: plan.request.budget.survivor_count]
        ]
        incumbent = plan.request.incumbent
        incumbent_branch = (
            None if incumbent is None else plan.search_space.branch_for(incumbent)
        )
        if incumbent_branch is not None and incumbent_branch not in survivors:
            incumbent_best = backend.best_feasible(
                state.studies[incumbent_branch.branch_id],
                incumbent_branch,
            )
            if incumbent_best is not None:
                if len(survivors) >= plan.request.budget.survivor_count:
                    survivors[-1] = incumbent_branch
                else:
                    survivors.append(incumbent_branch)
        survivors = list(dict.fromkeys(survivors))
        return tuple(survivors)

    def _run_tpe(
        self,
        plan: SearchPlan,
        state: _RunState,
        backend: OptunaBackend,
        survivors: tuple[BranchSpace, ...],
        *,
        deadline: float,
    ) -> None:
        budget = plan.request.budget
        no_progress = {branch.branch_id: 0 for branch in survivors}
        for branch in survivors:
            target = min(budget.min_trials_per_branch, branch.cardinality)
            study = state.studies[branch.branch_id]
            completed_ids = _completed_config_ids(backend, study, branch)
            while len(completed_ids) < target:
                if not state.budget.can_start(
                    deadline=deadline,
                    max_trials=budget.max_trials,
                ):
                    break
                if self._ask_and_measure(plan, state, backend, branch):
                    no_progress[branch.branch_id] = 0
                    completed_ids = _completed_config_ids(backend, study, branch)
                else:
                    no_progress[branch.branch_id] += 1
                    if no_progress[branch.branch_id] >= _MAX_CONSECUTIVE_DUPLICATE_ASKS:
                        break

        active = [
            branch
            for branch in survivors
            if len(
                _completed_config_ids(
                    backend,
                    state.studies[branch.branch_id],
                    branch,
                )
            )
            < branch.cardinality
            and no_progress[branch.branch_id] < _MAX_CONSECUTIVE_DUPLICATE_ASKS
        ]
        cursor = 0
        while active and state.budget.can_start(
            deadline=deadline,
            max_trials=budget.max_trials,
        ):
            index = cursor % len(active)
            branch = active[index]
            if self._ask_and_measure(plan, state, backend, branch):
                no_progress[branch.branch_id] = 0
            else:
                no_progress[branch.branch_id] += 1
            completed_ids = _completed_config_ids(
                backend,
                state.studies[branch.branch_id],
                branch,
            )
            if (
                len(completed_ids) >= branch.cardinality
                or no_progress[branch.branch_id] >= _MAX_CONSECUTIVE_DUPLICATE_ASKS
            ):
                active.pop(index)
                if active:
                    cursor %= len(active)
            else:
                cursor += 1

    def _run_local_neighbourhood(
        self,
        plan: SearchPlan,
        state: _RunState,
        backend: OptunaBackend,
        survivors: tuple[BranchSpace, ...],
        *,
        deadline: float,
    ) -> None:
        ranked = self._ranked_level1(state, backend, survivors)
        for completed in ranked[: plan.request.budget.local_top_k]:
            branch = plan.search_space.branch(completed.branch_id)
            study = state.studies[branch.branch_id]
            for config in branch.neighbours(completed.config):
                if not state.budget.can_start(
                    deadline=deadline,
                    max_trials=plan.request.budget.max_trials,
                ):
                    break
                if backend.measurement_for(study, branch, config.config_id) is not None:
                    continue
                try:
                    measurement = self._measure_screen(plan.request, config)
                    backend.record_completed(
                        study,
                        branch,
                        config,
                        measurement,
                        source="local_neighbour",
                    )
                finally:
                    state.budget.new_level1_trials += 1

    def _ask_and_measure(
        self,
        plan: SearchPlan,
        state: _RunState,
        backend: OptunaBackend,
        branch: BranchSpace,
    ) -> bool:
        study = state.studies[branch.branch_id]
        trial, config = backend.ask(study, branch)
        previous = backend.measurement_for(study, branch, config.config_id)
        try:
            measurement = (
                previous
                if previous is not None
                else self._measure_screen(plan.request, config)
            )
            backend.tell(study, trial, config, measurement)
            return previous is None
        except Exception as exc:  # noqa: BLE001 - persist infrastructure Trial failure
            backend.fail_infrastructure(study, trial, exc)
            return False
        finally:
            state.budget.new_level1_trials += 1

    def _measure_screen(
        self,
        request: SearchRequest,
        config: ConfigSpec,
    ) -> TrialMeasurement:
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
            return TrialMeasurement.infeasible(
                config_id=config.config_id,
                fidelity=Fidelity.SCREEN,
                scope=request.scope,
                penalty_ms=self.failure_penalty_ms,
                constraints=ConstraintVector(runtime=1.0),
                failure_kind="plan_rejection",
                metrics={"plan_rejection": details},
            )
        measurement = self.evaluator.evaluate(config, Fidelity.SCREEN)
        self._validate_measurement(
            measurement,
            config=config,
            fidelity=Fidelity.SCREEN,
            scope=request.scope,
        )
        return measurement

    def _select_promotions(
        self,
        plan: SearchPlan,
        state: _RunState,
        backend: OptunaBackend,
        survivors: tuple[BranchSpace, ...],
    ) -> tuple[ConfigSpec, ...]:
        ranked = self._ranked_level1(state, backend, survivors)
        if not ranked:
            return ()
        budget = plan.request.budget
        target = max(
            1,
            len(survivors),
            math.ceil(len(ranked) * budget.promote_fraction),
        )
        target = min(target, budget.enhanced_top_k, len(ranked))
        configs: list[ConfigSpec] = []
        for branch in survivors:
            best = backend.best_feasible(state.studies[branch.branch_id], branch)
            if best is not None:
                configs.append(best.config)
        configs.extend(completed.config for completed in ranked)
        values = _unique_configs(configs)[:target]
        return values

    def _evaluate_promotions(
        self,
        request: SearchRequest,
        promoted: tuple[ConfigSpec, ...],
        *,
        deadline: float,
    ) -> tuple[tuple[ConfigSpec, TrialMeasurement], ...]:
        values: list[tuple[ConfigSpec, TrialMeasurement]] = []
        for config in promoted:
            if time.monotonic() >= deadline:
                break
            measurement = self.evaluator.evaluate(config, Fidelity.ENHANCED)
            self._validate_measurement(
                measurement,
                config=config,
                fidelity=Fidelity.ENHANCED,
                scope=request.scope,
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
