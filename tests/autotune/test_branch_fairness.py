from __future__ import annotations

import time
from dataclasses import dataclass, replace

import pytest
from optuna.trial import TrialState, create_trial

import autotune.search_engine as engine_module
from autotune.evaluation import (
    ConstraintVector,
    EvaluationScope,
    Fidelity,
    TrialMeasurement,
)
from autotune.optuna_backend import (
    OptunaBackend,
    _constraints_from_trial,
    startup_trial_count,
)
from autotune.search_engine import (
    SearchBudget,
    SearchEngine,
    SearchPlan,
    SearchRequest,
    SearchStageTimings,
    _branch_rank,
    _RunState,
    _survivor_allocation,
    _survivor_capacity,
)
from autotune.search_space import BranchSpace, ParameterDomain, StructureSpec
from autotune.study_storage import SearchStorage, StudyIdentity
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
)


@dataclass(frozen=True)
class _Accepted:
    accepted: bool = True


class _PlanBuilder:
    def evaluate(
        self, config: ConfigSpec, context: object, hardware: object
    ) -> _Accepted:
        del config, context, hardware
        return _Accepted()


class _Evaluator:
    def __init__(self, scope: EvaluationScope) -> None:
        self.scope = scope

    def evaluate(self, config: ConfigSpec, fidelity: Fidelity) -> TrialMeasurement:
        tile = config.schedule.batch_tile_size
        if config.program.qkv_projection is ProjectionBackend.INPUT_DTYPE:
            latency = 1.0 if tile == 64 else 100.0
        else:
            latency = 2.0
        return _measurement(config, latency, scope=self.scope, fidelity=fidelity)


class _RecordingEvaluator(_Evaluator):
    def __init__(self, scope: EvaluationScope) -> None:
        super().__init__(scope)
        self.calls: list[tuple[str, Fidelity]] = []

    def evaluate(self, config: ConfigSpec, fidelity: Fidelity) -> TrialMeasurement:
        self.calls.append((config.config_id, fidelity))
        return super().evaluate(config, fidelity)


class _RaisingEvaluator:
    def evaluate(self, config: ConfigSpec, fidelity: Fidelity) -> TrialMeasurement:
        del config, fidelity
        raise RuntimeError("benchmark infrastructure failed")

    def compare(self, challenger: ConfigSpec, incumbent: ConfigSpec) -> object:
        del challenger, incumbent
        raise RuntimeError("paired benchmark infrastructure failed")


class _InfeasibleEvaluator:
    def __init__(self, scope: EvaluationScope) -> None:
        self.scope = scope

    def evaluate(self, config: ConfigSpec, fidelity: Fidelity) -> TrialMeasurement:
        return TrialMeasurement.infeasible(
            config_id=config.config_id,
            fidelity=fidelity,
            scope=self.scope,
            penalty_ms=1_000_000_000.0,
            constraints=ConstraintVector(runtime=1.0),
            failure_kind="constraint_violation",
        )


@dataclass(frozen=True)
class _SearchSpace:
    branches: tuple[BranchSpace, ...]
    mandatory_branch_ids: frozenset[str]

    def branch(self, branch_id: str) -> BranchSpace:
        return next(branch for branch in self.branches if branch.branch_id == branch_id)

    def branch_for(self, config: ConfigSpec) -> BranchSpace | None:
        return next(
            (
                branch
                for branch in self.branches
                if branch.parameters_for(config) is not None
            ),
            None,
        )


def _branch(
    precision_plan: PrecisionPlan,
    *,
    choices: tuple[int, ...] = (32, 64, 128),
    scope: str = "resident",
) -> BranchSpace:
    projection_pattern = (
        "all_input" if precision_plan is PrecisionPlan.INPUT_DTYPE else "all_autocast"
    )
    return BranchSpace(
        structure=StructureSpec(
            attention=AttentionBackend.REFERENCE_STREAMING,
            precision_plan=precision_plan,
            qkv_materialization=QKVMaterialization.VIEW,
            attention_output_bridge=AttentionOutputBridge.TORCH_BHSD_TO_BSD,
            ffn=FFNBackend.TORCH,
            residual_norm=ResidualNormBackend.TORCH,
            initial_norm=InitialNormBackend.TORCH,
            runtime=RuntimeBackend.BATCH_TILED_CUDA_GRAPH,
        ),
        domains=(
            ParameterDomain(
                "projection_pattern",
                (projection_pattern,),
                default=projection_pattern,
            ),
            ParameterDomain(
                "batch_tile_size",
                choices,
                default=choices[0],
            ),
        ),
        scope=scope,
    )


def _measurement(
    config: ConfigSpec,
    latency: float,
    *,
    scope: EvaluationScope = EvaluationScope.RESIDENT,
    fidelity: Fidelity = Fidelity.SCREEN,
) -> TrialMeasurement:
    return TrialMeasurement(
        config_id=config.config_id,
        fidelity=fidelity,
        scope=scope,
        objective_ms=latency,
        median_ms=latency,
        constraints=ConstraintVector(),
    )


def _request(
    *,
    max_trials: int | None = None,
    max_seconds: float = 60.0,
) -> SearchRequest:
    return SearchRequest(
        case_id="fair-screen",
        execution_context=object(),
        hardware=object(),
        scope=EvaluationScope.RESIDENT,
        environment="test",
        search_identity="search-v1",
        enhanced_identity="enhanced-v1",
        promotion_identity="promotion-v1",
        budget=SearchBudget(
            max_seconds=max_seconds,
            max_trials=max_trials,
        ),
    )


def _plan(request: SearchRequest, branches: tuple[BranchSpace, ...]) -> SearchPlan:
    search_space = _SearchSpace(
        branches=branches,
        mandatory_branch_ids=frozenset(branch.branch_id for branch in branches),
    )
    return SearchPlan(
        request=request,
        search_space=search_space,  # type: ignore[arg-type]
        identities=tuple(
            StudyIdentity(
                case_id=request.case_id,
                branch_id=branch.branch_id,
                environment=request.environment,
                search_identity=request.search_identity,
            )
            for branch in branches
        ),
    )


def test_screening_reuses_history_and_gives_every_branch_one_witness(
    tmp_path,
) -> None:
    branches = (
        _branch(PrecisionPlan.INPUT_DTYPE),
        _branch(PrecisionPlan.FP16_QKV_ATTENTION),
    )
    request = _request(max_trials=6)
    plan = _plan(request, branches)
    engine = SearchEngine(
        storage=SearchStorage(tmp_path),
        evaluator=_Evaluator(request.scope),
        plan_builder=_PlanBuilder(),
    )
    backend = OptunaBackend(engine.storage, seed=request.seed)
    studies = {
        branch.branch_id: backend.create_study(plan.identity_for(branch.branch_id))
        for branch in branches
    }
    state = _RunState(studies=studies)

    historical = branches[0].default_config()
    backend.record_completed(
        studies[branches[0].branch_id],
        branches[0],
        historical,
        _measurement(historical, 100.0),
        source="historical",
    )
    engine._enqueue_initial_configs(plan, state, backend)

    assert engine._screen_structures(
        plan,
        state,
        backend,
        deadline=time.monotonic() + 30.0,
    )
    assert state.budget.new_level1_trials == 1
    for branch in branches:
        completed = backend.completed_trials(studies[branch.branch_id], branch)
        assert len({trial.config.config_id for trial in completed}) == 1


def test_duplicate_tpe_proposal_uses_one_unseen_point_without_extra_budget(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branch = _branch(PrecisionPlan.INPUT_DTYPE)
    request = _request(max_trials=3)
    plan = _plan(request, (branch,))
    evaluator = _RecordingEvaluator(request.scope)
    engine = SearchEngine(
        storage=SearchStorage(tmp_path),
        evaluator=evaluator,
        plan_builder=_PlanBuilder(),
    )
    backend = OptunaBackend(engine.storage, seed=request.seed)
    study = backend.create_study(plan.identity_for(branch.branch_id))
    historical = branch.config_at(0)
    backend.record_completed(
        study,
        branch,
        historical,
        _measurement(historical, 1.0),
        source="historical",
    )
    state = _RunState(studies={branch.branch_id: study})

    def _duplicate_ask(*args: object) -> tuple[object, ConfigSpec]:
        del args
        return study.ask(), historical

    monkeypatch.setattr(backend, "ask", _duplicate_ask)

    assert engine._ask_and_measure(plan, state, backend, branch)
    assert state.budget.new_level1_trials == 1
    completed = backend.completed_trials(study, branch)
    assert len({trial.config.config_id for trial in completed}) == 2
    assert evaluator.calls[0][0] != historical.config_id
    failed = study.get_trials(deepcopy=False, states=(TrialState.FAIL,))
    assert failed[-1].user_attrs["duplicate_config_id"] == historical.config_id


def test_exhausted_finite_branch_does_not_ask_or_measure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branch = _branch(PrecisionPlan.INPUT_DTYPE, choices=(32, 64))
    request = _request(max_trials=2)
    plan = _plan(request, (branch,))
    evaluator = _RecordingEvaluator(request.scope)
    engine = SearchEngine(
        storage=SearchStorage(tmp_path),
        evaluator=evaluator,
        plan_builder=_PlanBuilder(),
    )
    backend = OptunaBackend(engine.storage, seed=request.seed)
    study = backend.create_study(plan.identity_for(branch.branch_id))
    for index in range(branch.cardinality):
        config = branch.config_at(index)
        backend.record_completed(
            study,
            branch,
            config,
            _measurement(config, float(index + 1)),
            source="historical",
        )
    backend = OptunaBackend(engine.storage, seed=request.seed + 1)
    study = backend.create_study(plan.identity_for(branch.branch_id))
    state = _RunState(studies={branch.branch_id: study})

    monkeypatch.setattr(
        backend,
        "ask",
        lambda *args: (_ for _ in ()).throw(AssertionError("unexpected ask")),
    )

    assert not engine._ask_and_measure(plan, state, backend, branch)
    assert state.budget.new_level1_trials == 0
    assert evaluator.calls == []
    assert engine._select_promotions(plan, state, backend) == (branch.config_at(0),)


def test_formal_history_skips_rejected_challenger_for_same_incumbent(tmp_path) -> None:
    branch = _branch(PrecisionPlan.INPUT_DTYPE)
    configs = tuple(branch.config_at(index) for index in range(branch.cardinality))
    incumbent, rejected, next_challenger = configs
    request = replace(_request(max_trials=3), incumbent=incumbent)
    plan = _plan(request, (branch,))
    engine = SearchEngine(
        storage=SearchStorage(tmp_path),
        evaluator=_Evaluator(request.scope),
        plan_builder=_PlanBuilder(),
    )
    backend = OptunaBackend(engine.storage, seed=request.seed)
    study = backend.create_study(plan.identity_for(branch.branch_id))
    for config, latency in (
        (incumbent, 0.5),
        (rejected, 1.0),
        (next_challenger, 2.0),
    ):
        backend.record_completed(
            study,
            branch,
            config,
            _measurement(config, latency),
            source="historical",
        )
    state = _RunState(studies={branch.branch_id: study})
    engine.storage.record_challenger_attempt(
        case_id=request.case_id,
        environment=request.environment,
        incumbent_id=incumbent.config_id,
        challenger_id=rejected.config_id,
        promotion_identity=request.promotion_identity,
    )

    promoted = engine._select_promotions(plan, state, backend)

    assert promoted == (next_challenger,)
    assert engine.storage.attempted_challenger_ids(
        case_id=request.case_id,
        environment=request.environment,
        incumbent_id=incumbent.config_id,
        promotion_identity=request.promotion_identity,
    ) == frozenset({rejected.config_id})


def test_enhanced_evidence_cache_avoids_remeasurement(tmp_path) -> None:
    request = _request(max_trials=1)
    config = _branch(PrecisionPlan.INPUT_DTYPE).default_config()
    evaluator = _RecordingEvaluator(request.scope)
    engine = SearchEngine(
        storage=SearchStorage(tmp_path),
        evaluator=evaluator,
        plan_builder=_PlanBuilder(),
    )
    cached = _measurement(
        config,
        0.75,
        scope=request.scope,
        fidelity=Fidelity.ENHANCED,
    )
    assert engine.evaluation_cache.put(
        case_id=request.case_id,
        evidence_identity=request.enhanced_identity,
        measurement=cached,
    )

    values = engine._evaluate_promotions(
        request,
        (config,),
        deadline=time.monotonic() + 1.0,
    )

    assert values == ((config, cached),)
    assert evaluator.calls == []


def test_plan_allows_trial_cap_below_already_persisted_screen_coverage(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branches = (
        _branch(PrecisionPlan.INPUT_DTYPE),
        _branch(PrecisionPlan.FP16_QKV_ATTENTION),
    )
    search_space = _SearchSpace(
        branches, frozenset(branch.branch_id for branch in branches)
    )
    monkeypatch.setattr(
        engine_module,
        "ProgramSearchSpace",
        lambda **kwargs: search_space,
    )
    engine = SearchEngine(
        storage=SearchStorage(tmp_path),
        evaluator=_Evaluator(EvaluationScope.RESIDENT),
        plan_builder=_PlanBuilder(),
    )

    plan = engine.plan(_request(max_trials=1))

    assert len(plan.search_space.branches) == 2


def test_branch_rank_uses_best_measured_median_without_extrapolation(
    tmp_path,
) -> None:
    branch = _branch(PrecisionPlan.INPUT_DTYPE, choices=tuple(range(1, 6)))
    backend = OptunaBackend(SearchStorage(tmp_path), seed=7)
    study = backend.create_study(
        StudyIdentity("rank", branch.branch_id, "test", "search-v1")
    )
    for index, latency in enumerate((10.0, 8.0, 9.0)):
        config = branch.config_at(index)
        backend.record_completed(
            study,
            branch,
            config,
            _measurement(config, latency),
            source="history",
        )

    assert _branch_rank(backend, study, branch) == (0, 8.0, branch.branch_id)


def test_survivor_capacity_concentrates_budget_past_tpe_startup() -> None:
    branches = (
        _branch(PrecisionPlan.INPUT_DTYPE, choices=tuple(range(1, 21))),
        _branch(PrecisionPlan.FP16_QKV_ATTENTION, choices=tuple(range(1, 21))),
        _branch(PrecisionPlan.FP16_ATTENTION_BRANCH, choices=tuple(range(1, 21))),
        _branch(PrecisionPlan.FP16_CORE, choices=tuple(range(1, 21))),
    )
    counts = {branch.branch_id: 1 for branch in branches}

    assert _survivor_capacity(branches, counts, remaining_trials=20) == 2
    assert _survivor_capacity(branches, counts, remaining_trials=9) == 0
    counts[branches[0].branch_id] = 11
    assert _survivor_capacity(branches, counts, remaining_trials=20) == 3

    counts = {branch.branch_id: 1 for branch in branches}
    assert _survivor_allocation(branches, counts, remaining_trials=20) == (1, 2)
    assert _survivor_allocation(branches, counts, remaining_trials=9) == (0, 4)


def test_survivor_scheduler_reaches_the_first_guided_trial(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branches = (
        _branch(PrecisionPlan.INPUT_DTYPE, choices=tuple(range(1, 21))),
        _branch(PrecisionPlan.FP16_QKV_ATTENTION, choices=tuple(range(1, 21))),
        _branch(PrecisionPlan.FP16_ATTENTION_BRANCH, choices=tuple(range(1, 21))),
        _branch(PrecisionPlan.FP16_CORE, choices=tuple(range(1, 21))),
    )
    request = _request(max_trials=20)
    plan = _plan(request, branches)
    engine = SearchEngine(
        storage=SearchStorage(tmp_path),
        evaluator=_Evaluator(request.scope),
        plan_builder=_PlanBuilder(),
    )
    backend = OptunaBackend(engine.storage, seed=request.seed)
    studies = {
        branch.branch_id: backend.create_study(
            plan.identity_for(branch.branch_id),
            n_startup_trials=startup_trial_count(branch),
        )
        for branch in branches
    }
    for index, branch in enumerate(branches):
        config = branch.default_config()
        backend.record_completed(
            studies[branch.branch_id],
            branch,
            config,
            _measurement(config, float(index + 1)),
            source="witness",
        )
    state = _RunState(studies=studies)

    def measure_once(
        current_plan: SearchPlan,
        current_state: _RunState,
        current_backend: OptunaBackend,
        branch: BranchSpace,
    ) -> bool:
        del current_plan
        study = current_state.studies[branch.branch_id]
        seen = current_backend.terminal_config_ids(study, branch)
        config = next(
            branch.config_at(index)
            for index in range(branch.cardinality)
            if branch.config_at(index).config_id not in seen
        )
        current_backend.record_completed(
            study,
            branch,
            config,
            _measurement(config, 1.0),
            source="test",
        )
        current_state.budget.new_level1_trials += 1
        return True

    monkeypatch.setattr(engine, "_ask_and_measure", measure_once)

    engine._run_survivor_tpe(
        plan,
        state,
        backend,
        deadline=time.monotonic() + 30.0,
    )

    counts = [len(backend.completed_trials(studies[b.branch_id], b)) for b in branches]
    assert counts[0] >= startup_trial_count(branches[0]) + 1
    assert any(count > 1 for count in counts[1:])
    assert sum(counts) == len(branches) + 20
    assert state.budget.new_level1_trials == 20
    assert state.selection_rounds == 1
    assert state.pruned_branches == 3


def test_invalid_constraint_history_uses_three_infeasible_dimensions() -> None:
    trial = create_trial(value=1.0, user_attrs={"constraints": "invalid"})

    assert _constraints_from_trial(trial) == (1.0, 1.0, 1.0)


def test_plan_keeps_warm_starts_out_of_the_structure_domain(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branches = (
        _branch(PrecisionPlan.INPUT_DTYPE),
        _branch(PrecisionPlan.FP16_QKV_ATTENTION),
    )
    incumbent = branches[0].default_config()
    transfer = branches[1].default_config()
    request = replace(
        _request(max_trials=6),
        incumbent=incumbent,
        warm_starts=(incumbent, transfer, transfer),
    )
    search_space = _SearchSpace(
        branches, frozenset(branch.branch_id for branch in branches)
    )
    captured: dict[str, object] = {}

    def _search_space_factory(**kwargs: object) -> _SearchSpace:
        captured.update(kwargs)
        return search_space

    monkeypatch.setattr(engine_module, "ProgramSearchSpace", _search_space_factory)
    engine = SearchEngine(
        storage=SearchStorage(tmp_path),
        evaluator=_Evaluator(request.scope),
        plan_builder=_PlanBuilder(),
    )

    engine.plan(request)

    required = captured["required_configs"]
    assert isinstance(required, tuple)
    assert tuple(config.config_id for config in required) == (incumbent.config_id,)


def test_time_limited_partial_screen_has_no_winner(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branch = _branch(PrecisionPlan.INPUT_DTYPE)
    request = _request(max_trials=3, max_seconds=1e-9)
    plan = _plan(request, (branch,))
    engine = SearchEngine(
        storage=SearchStorage(tmp_path),
        evaluator=_Evaluator(request.scope),
        plan_builder=_PlanBuilder(),
    )
    monkeypatch.setattr(engine, "plan", lambda _: plan)

    result = engine.run(request)

    assert result.stop_reason == "insufficient_screen_budget"
    assert result.selected_config is None
    assert result.selected_measurement is None
    assert not result.deployment_approved


def test_run_reports_stage_timings_budget_and_branch_coverage(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branch = _branch(PrecisionPlan.INPUT_DTYPE)
    request = _request(max_trials=3)
    plan = _plan(request, (branch,))
    engine = SearchEngine(
        storage=SearchStorage(tmp_path),
        evaluator=_Evaluator(request.scope),
        plan_builder=_PlanBuilder(),
    )
    monkeypatch.setattr(engine, "plan", lambda _: plan)

    result = engine.run(request)

    timings = result.stage_timings
    assert result.budget_seconds == request.budget.max_seconds
    assert result.covered_branches == 1
    assert result.mandatory_coverage_complete
    assert min(
        timings.planning,
        timings.screen,
        timings.enhanced,
        timings.formal,
        timings.total,
    ) >= 0.0
    assert timings.total >= (
        timings.planning + timings.screen + timings.enhanced + timings.formal
    )


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_stage_timings_require_finite_non_negative_seconds(value: float) -> None:
    with pytest.raises(ValueError):
        SearchStageTimings(total=value)


def test_resident_optional_branches_do_not_block_promotion(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mandatory = _branch(PrecisionPlan.INPUT_DTYPE)
    optional = _branch(PrecisionPlan.FP16_QKV_ATTENTION)
    request = _request(max_trials=1)
    search_space = _SearchSpace(
        branches=(mandatory, optional),
        mandatory_branch_ids=frozenset({mandatory.branch_id}),
    )
    plan = SearchPlan(
        request=request,
        search_space=search_space,  # type: ignore[arg-type]
        identities=tuple(
            StudyIdentity(
                case_id=request.case_id,
                branch_id=branch.branch_id,
                environment=request.environment,
                search_identity=request.search_identity,
            )
            for branch in search_space.branches
        ),
    )
    engine = SearchEngine(
        storage=SearchStorage(tmp_path),
        evaluator=_Evaluator(request.scope),
        plan_builder=_PlanBuilder(),
    )
    monkeypatch.setattr(engine, "plan", lambda _: plan)

    result = engine.run(request)

    backend = OptunaBackend(engine.storage, seed=request.seed)
    optional_study = backend.create_study(plan.identity_for(optional.branch_id))
    assert result.new_level1_trials == 1
    assert result.mandatory_coverage_complete
    assert result.stop_reason == "completed"
    assert result.enhanced_measurements
    assert result.formal_challenger_measurement is not None
    assert result.deployment_approved
    assert engine.storage.attempted_challenger_ids(
        case_id=request.case_id,
        environment=request.environment,
        incumbent_id=None,
        promotion_identity=request.promotion_identity,
    ) == frozenset()
    assert backend.completed_trials(optional_study, optional) == ()


def test_small_budget_adds_a_real_exploration_point(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branch = _branch(PrecisionPlan.INPUT_DTYPE)
    request = _request(max_trials=1)
    plan = _plan(request, (branch,))
    engine = SearchEngine(
        storage=SearchStorage(tmp_path),
        evaluator=_Evaluator(request.scope),
        plan_builder=_PlanBuilder(),
    )
    backend = OptunaBackend(engine.storage, seed=request.seed)
    study = backend.create_study(plan.identity_for(branch.branch_id))
    challenger = branch.config_at(1)
    backend.record_completed(
        study,
        branch,
        challenger,
        _measurement(challenger, 1.0),
        source="previous_pass",
    )
    monkeypatch.setattr(engine, "plan", lambda _: plan)

    result = engine.run(request)

    assert result.new_level1_trials == 1
    assert result.stop_reason == "completed"
    assert result.enhanced_measurements
    assert len(backend.completed_trials(study, branch)) == 2


def test_exhausted_space_can_close_history_without_new_screen_point(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branch = _branch(PrecisionPlan.INPUT_DTYPE, choices=(32, 64))
    request = _request(max_trials=1)
    plan = _plan(request, (branch,))
    engine = SearchEngine(
        storage=SearchStorage(tmp_path),
        evaluator=_Evaluator(request.scope),
        plan_builder=_PlanBuilder(),
    )
    backend = OptunaBackend(engine.storage, seed=request.seed)
    study = backend.create_study(plan.identity_for(branch.branch_id))
    for index in range(branch.cardinality):
        config = branch.config_at(index)
        backend.record_completed(
            study,
            branch,
            config,
            _measurement(config, float(index + 1)),
            source="previous_pass",
        )
    monkeypatch.setattr(engine, "plan", lambda _: plan)

    result = engine.run(request)

    assert result.level1_space_exhausted
    assert result.new_level1_trials == 0
    assert result.stop_reason == "completed"
    assert result.enhanced_measurements


def test_new_screen_point_can_outrank_historical_candidate(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branch = _branch(PrecisionPlan.INPUT_DTYPE, choices=(32, 64))
    historical = branch.config_at(0)
    new_winner = branch.config_at(1)
    request = _request(max_trials=1)
    plan = _plan(request, (branch,))
    engine = SearchEngine(
        storage=SearchStorage(tmp_path),
        evaluator=_Evaluator(request.scope),
        plan_builder=_PlanBuilder(),
    )
    backend = OptunaBackend(engine.storage, seed=request.seed)
    study = backend.create_study(plan.identity_for(branch.branch_id))
    backend.record_completed(
        study,
        branch,
        historical,
        _measurement(historical, 10.0),
        source="previous_pass",
    )
    monkeypatch.setattr(engine, "plan", lambda _: plan)

    result = engine.run(request)

    assert result.new_level1_trials == 1
    assert result.best_screen_config_id == new_winner.config_id
    assert result.locked_challenger == new_winner
    assert any(
        measurement.config_id == new_winner.config_id
        for measurement in result.enhanced_measurements
    )


def test_tpe_startup_uses_ten_or_the_finite_branch_cardinality(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normal = _branch(PrecisionPlan.INPUT_DTYPE, choices=tuple(range(1, 17)))
    huge = BranchSpace(
        structure=normal.structure,
        domains=(
            ParameterDomain("x", tuple(range(32)), default=0),
            ParameterDomain("y", tuple(range(32)), default=0),
        ),
        scope="resident",
    )

    tiny = _branch(PrecisionPlan.INPUT_DTYPE, choices=(1, 2, 3))

    assert startup_trial_count(normal) == 10
    assert startup_trial_count(huge) == 10
    assert startup_trial_count(tiny) == 3

    backend = OptunaBackend(SearchStorage(tmp_path), seed=7)
    identity = StudyIdentity("startup", normal.branch_id, "test", "search-v1")
    study = backend.create_study(identity, n_startup_trials=10)
    for index in range(10):
        config = normal.config_at(index)
        backend.record_completed(
            study,
            normal,
            config,
            _measurement(config, float(index + 1)),
            source="preloaded",
        )

    assert study.sampler._n_startup_trials == 10
    assert len(backend.completed_trials(study, normal)) == 10

    def _unexpected_random_sample(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("completed preloads did not count toward TPE startup")

    monkeypatch.setattr(
        study.sampler._random_sampler,
        "sample_independent",
        _unexpected_random_sample,
    )
    backend.ask(study, normal)


def test_duplicate_fallback_benchmark_error_propagates(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branch = _branch(PrecisionPlan.INPUT_DTYPE)
    request = _request(max_trials=3)
    plan = _plan(request, (branch,))
    engine = SearchEngine(
        storage=SearchStorage(tmp_path),
        evaluator=_RaisingEvaluator(),  # type: ignore[arg-type]
        plan_builder=_PlanBuilder(),
    )
    backend = OptunaBackend(engine.storage, seed=request.seed)
    study = backend.create_study(plan.identity_for(branch.branch_id))
    historical = branch.default_config()
    backend.record_completed(
        study,
        branch,
        historical,
        _measurement(historical, 1.0),
        source="seed",
    )
    state = _RunState(studies={branch.branch_id: study})

    monkeypatch.setattr(
        backend,
        "ask",
        lambda *args: (study.ask(), historical),
    )

    with pytest.raises(RuntimeError, match="benchmark infrastructure failed"):
        engine._ask_and_measure(plan, state, backend, branch)
    assert state.budget.new_level1_trials == 1


def test_one_infrastructure_failure_remains_retryable(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branch = _branch(PrecisionPlan.INPUT_DTYPE)
    request = _request(max_trials=3)
    plan = _plan(request, (branch,))
    engine = SearchEngine(
        storage=SearchStorage(tmp_path),
        evaluator=_RaisingEvaluator(),  # type: ignore[arg-type]
        plan_builder=_PlanBuilder(),
    )
    backend = OptunaBackend(engine.storage, seed=request.seed)
    study = backend.create_study(plan.identity_for(branch.branch_id))
    failed_config = branch.default_config()
    state = _RunState(studies={branch.branch_id: study})
    monkeypatch.setattr(
        backend,
        "ask",
        lambda *args: (study.ask(), failed_config),
    )

    with pytest.raises(RuntimeError, match="benchmark infrastructure failed"):
        engine._ask_and_measure(plan, state, backend, branch)

    assert failed_config.config_id not in backend.terminal_config_ids(study, branch)
    assert backend.enqueue(
        study,
        branch,
        failed_config,
        source="retry",
    )


def test_unknown_benchmark_errors_propagate_after_screening(tmp_path) -> None:
    branch = _branch(PrecisionPlan.INPUT_DTYPE)
    request = _request(max_trials=3)
    engine = SearchEngine(
        storage=SearchStorage(tmp_path),
        evaluator=_RaisingEvaluator(),  # type: ignore[arg-type]
        plan_builder=_PlanBuilder(),
    )
    middle = branch.config_at(1)

    with pytest.raises(RuntimeError, match="benchmark infrastructure failed"):
        engine._evaluate_promotions(
            request,
            (middle,),
            deadline=time.monotonic() + 30.0,
        )
    with pytest.raises(RuntimeError, match="benchmark infrastructure failed"):
        engine._run_formal(
            request,
            middle,
            deadline=time.monotonic() + 30.0,
        )
