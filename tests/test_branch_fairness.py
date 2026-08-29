from __future__ import annotations

import time
from dataclasses import dataclass, replace

import pytest

import autotune.search_engine as engine_module
from autotune.evaluation import (
    ConstraintVector,
    EvaluationScope,
    Fidelity,
    TrialMeasurement,
)
from autotune.optuna_backend import OptunaBackend, startup_trial_count
from autotune.search_engine import (
    SearchBudget,
    SearchEngine,
    SearchPlan,
    SearchRequest,
    _RunState,
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


class _RaisingEvaluator:
    def evaluate(self, config: ConfigSpec, fidelity: Fidelity) -> TrialMeasurement:
        del config, fidelity
        raise RuntimeError("benchmark infrastructure failed")

    def compare(self, challenger: ConfigSpec, incumbent: ConfigSpec) -> object:
        del challenger, incumbent
        raise RuntimeError("paired benchmark infrastructure failed")


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
                ordered=True,
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
        budget=SearchBudget(
            max_seconds=max_seconds,
            max_trials=max_trials,
            survivor_count=1,
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
            )
            for branch in branches
        ),
    )


def test_screening_uses_three_unique_configs_and_best_feasible_latency(
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

    historical = branches[0].representative_configs(limit=3)[0]
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
    assert state.budget.new_level1_trials == 5
    for branch in branches:
        completed = backend.completed_trials(studies[branch.branch_id], branch)
        assert len({trial.config.config_id for trial in completed}) == 3

    # The first branch wins because its best representative is 1 ms, even
    # though its other completed observations are 100 ms.
    assert engine._select_survivors(plan, state, backend) == (branches[0],)


def test_plan_rejects_trial_cap_below_fair_screening_total(
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

    with pytest.raises(ValueError, match="fair structure screening.*at least 6"):
        engine.plan(_request(max_trials=5))


def test_plan_keeps_unique_warm_start_branches(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branches = (
        _branch(PrecisionPlan.INPUT_DTYPE),
        _branch(PrecisionPlan.FP16_QKV_ATTENTION),
    )
    incumbent = branches[0].representative_configs(limit=1)[0]
    transfer = branches[1].representative_configs(limit=1)[0]
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
    assert tuple(config.config_id for config in required) == (
        incumbent.config_id,
        transfer.config_id,
    )


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


def test_tpe_startup_is_dynamic_and_uses_an_absolute_history_threshold(
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

    assert (
        startup_trial_count(
            normal,
            branch_budget=5,
            scope=EvaluationScope.RESIDENT,
        )
        == 4
    )
    assert (
        startup_trial_count(
            huge,
            branch_budget=10,
            scope=EvaluationScope.RESIDENT,
        )
        == 6
    )
    assert (
        startup_trial_count(
            huge,
            branch_budget=10,
            scope=EvaluationScope.STREAMED,
        )
        == 4
    )
    assert (
        startup_trial_count(
            huge,
            branch_budget=3,
            scope=EvaluationScope.STREAMED,
        )
        == 3
    )

    backend = OptunaBackend(SearchStorage(tmp_path), seed=7)
    identity = StudyIdentity("startup", normal.branch_id, "test")
    study = backend.create_study(identity, n_startup_trials=4)
    for index, config in enumerate(normal.representative_configs(limit=4), start=1):
        backend.record_completed(
            study,
            normal,
            config,
            _measurement(config, float(index)),
            source="preloaded",
        )

    assert study.sampler._n_startup_trials == 4
    assert len(backend.completed_trials(study, normal)) == 4

    def _unexpected_random_sample(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("completed preloads did not count toward TPE startup")

    monkeypatch.setattr(
        study.sampler._random_sampler,
        "sample_independent",
        _unexpected_random_sample,
    )
    backend.ask(study, normal)


def test_unknown_benchmark_errors_propagate_after_screening(tmp_path) -> None:
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
    middle = branch.representative_configs(limit=3)[1]
    backend.record_completed(
        study,
        branch,
        middle,
        _measurement(middle, 1.0),
        source="seed",
    )
    state = _RunState(studies={branch.branch_id: study})

    with pytest.raises(RuntimeError, match="benchmark infrastructure failed"):
        engine._run_local_neighbourhood(
            plan,
            state,
            backend,
            (branch,),
            deadline=time.monotonic() + 30.0,
        )
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
