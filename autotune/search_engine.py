"""Branch screening, constraint-aware TPE, and multi-fidelity promotion."""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
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
    min_trials_per_branch: int = 5
    survivor_count: int = 3
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
        for name in (
            "min_trials_per_branch",
            "survivor_count",
            "enhanced_top_k",
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
        if not 0 <= self.feasible_level1 <= self.completed_level1:
            raise ValueError("feasible_level1 must be within completed_level1")
        if self.best_screen_median_ms is not None and (
            not math.isfinite(self.best_screen_median_ms)
            or self.best_screen_median_ms <= 0.0
        ):
            raise ValueError("best_screen_median_ms must be finite and positive")
        if (self.best_screen_config_id is None) != (
            self.best_screen_median_ms is None
        ):
            raise ValueError("best Screen identity and latency must be set together")
        for measurement in self.enhanced_measurements:
            if measurement.fidelity is not Fidelity.ENHANCED:
                raise ValueError("enhanced measurements must use Enhanced fidelity")
        normalized_failures = tuple(
            sorted((str(kind), int(count)) for kind, count in self.screen_failure_counts)
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
    def made_search_progress(self) -> bool:
        """Whether this run added a new Screen point or Formal decision."""

        return (
            self.new_level1_trials > 0 or self.formal_challenger_measurement is not None
        )


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


def _completed_config_ids(
    backend: OptunaBackend,
    study: Study,
    branch: BranchSpace,
) -> set[str]:
    return {
        completed.config.config_id
        for completed in backend.completed_trials(study, branch)
    }


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
        """Execute screening, TPE, and multi-fidelity promotion in order."""

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
                if (
                    locked_challenger is not None
                    and formal_challenger_measurement is not None
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
            len(
                _completed_config_ids(
                    backend,
                    run_state.studies[branch.branch_id],
                    branch,
                )
            )
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
            enhanced_measurements=tuple(
                measurement for _, measurement in enhanced
            ),
            locked_challenger=locked_challenger,
            formal_challenger_measurement=formal_challenger_measurement,
            formal_comparison=formal_comparison,
            stop_reason=stop_reason,
            new_level1_trials=run_state.budget.new_level1_trials,
            feasible_level1=feasible_level1,
            best_screen_config_id=best_screen_config_id,
            best_screen_median_ms=best_screen_median_ms,
            screen_failure_counts=screen_failure_counts,
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
                        or measurement.objective_ms
                        < previous.measurement.objective_ms
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
                if not self._ask_and_measure(plan, state, backend, branch):
                    return False
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
                if not self._ask_and_measure(plan, state, backend, branch):
                    break
                completed_ids = _completed_config_ids(backend, study, branch)

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
        ]
        cursor = 0
        while active and state.budget.can_start(
            deadline=deadline,
            max_trials=budget.max_trials,
        ):
            index = cursor % len(active)
            branch = active[index]
            progressed = self._ask_and_measure(plan, state, backend, branch)
            completed_ids = _completed_config_ids(
                backend,
                state.studies[branch.branch_id],
                branch,
            )
            if len(completed_ids) >= branch.cardinality or not progressed:
                active.pop(index)
                if active:
                    cursor %= len(active)
            else:
                cursor += 1

    def _ask_and_measure(
        self,
        plan: SearchPlan,
        state: _RunState,
        backend: OptunaBackend,
        branch: BranchSpace,
    ) -> bool:
        study = state.studies[branch.branch_id]
        seen = _completed_config_ids(backend, study, branch)
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
            except Exception:  # noqa: BLE001 - stop this branch on benchmark failure
                return False
            finally:
                state.budget.new_level1_trials += 1

        try:
            measurement = self._measure_screen(plan.request, config)
            backend.tell(study, trial, config, measurement)
            return True
        except Exception as exc:  # noqa: BLE001 - persist infrastructure Trial failure
            backend.fail_infrastructure(study, trial, exc)
            return False
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
            for completed in backend.completed_trials(study, branch)
            if (index := branch.index_for(completed.config)) is not None
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
        incumbent_id = (
            None if plan.request.incumbent is None else plan.request.incumbent.config_id
        )
        attempted = self.storage.attempted_challenger_ids(
            case_id=plan.request.case_id,
            environment=plan.request.environment,
            incumbent_id=incumbent_id,
        )
        values = tuple(
            config
            for config in _unique_configs(configs)
            if config.config_id != incumbent_id and config.config_id not in attempted
        )[:target]
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
