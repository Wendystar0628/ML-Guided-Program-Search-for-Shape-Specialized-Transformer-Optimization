from __future__ import annotations

from types import SimpleNamespace

import optuna

from autotune import search_sweep
from autotune.evaluation import (
    ConstraintVector,
    EvaluationScope,
    Fidelity,
    PairedMeasurement,
    TrialMeasurement,
)
from autotune.meta_warmstart import (
    WarmStartCandidate,
    best_screen_candidates,
    select_meta_warm_starts,
)
from autotune.search_engine import SearchResult
from autotune.study_storage import StudyIdentity
from benchmarking.protocols import TransformerShape
from deployment.registry import EnvironmentFingerprint, ShapeFingerprint
from solution.config import (
    ConfigSpec,
    RuntimeBackend,
    ScheduleConfig,
    portable_config,
)


def _graph_config() -> ConfigSpec:
    return ConfigSpec(
        program=portable_config().program,
        schedule=ScheduleConfig(runtime=RuntimeBackend.CUDA_GRAPH),
    )


def _hardware() -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        device_name="Test GPU",
        compute_capability="9.9",
        driver_version="600.1",
        torch_version="2.12.0",
        cuda_runtime_version="13.2",
        cudnn_version="92000",
        triton_version="3.7.0",
        matmul_precision="highest",
        allow_tf32=False,
        allow_fp16_reduced_precision_reduction=False,
        cudnn_allow_tf32=False,
        official_definitions_digest="official-test",
        solution_implementation_digest="solution-test",
    )


def _measurement(
    config: ConfigSpec,
    latency_ms: float,
    *,
    fidelity: Fidelity = Fidelity.FORMAL,
) -> TrialMeasurement:
    return TrialMeasurement(
        config_id=config.config_id,
        fidelity=fidelity,
        scope=EvaluationScope.RESIDENT,
        objective_ms=latency_ms,
        median_ms=latency_ms,
        p90_ms=latency_ms,
        constraints=ConstraintVector(),
    )


def _completed_result(
    config: ConfigSpec,
    latency_ms: float,
    *,
    incumbent: ConfigSpec | None = None,
) -> SearchResult:
    measurement = _measurement(config, latency_ms)
    comparison = (
        None
        if incumbent is None
        else PairedMeasurement(
            incumbent=_measurement(incumbent, latency_ms * 1.01),
            challenger=measurement,
            paired_ratios=(1.02,) * 11 + (1.0,) * 2,
        )
    )
    return SearchResult(
        incumbent_config=incumbent,
        selected_config=config,
        selected_measurement=measurement,
        branch_count=1,
        completed_level1=1,
        enhanced_measurements=(
            _measurement(config, latency_ms, fidelity=Fidelity.ENHANCED),
        ),
        locked_challenger=config,
        formal_challenger_measurement=measurement,
        formal_comparison=comparison,
        stop_reason="completed",
    )


def _shape(
    case_id: str,
    *,
    batch_size: int,
    heads: int,
    seq_len: int = 128,
) -> TransformerShape:
    return TransformerShape(
        case_id=case_id,
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=128,
        num_heads=heads,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )


class _AcceptingSearchSpace:
    def accepted(self, config: ConfigSpec) -> bool:
        return True

    def branch_for(self, config: ConfigSpec) -> object:
        return object()


def test_service_combines_registry_family_and_earlier_approved_winner(
    monkeypatch,
) -> None:
    winner = _graph_config()
    registry_config = ConfigSpec(
        program=portable_config().program,
        schedule=ScheduleConfig(runtime=RuntimeBackend.COMPILED_FORWARD),
    )
    winner_measurement = _measurement(winner, 1.0)
    incumbent = portable_config()
    first_result = SearchResult(
        incumbent_config=incumbent,
        selected_config=winner,
        selected_measurement=winner_measurement,
        branch_count=2,
        completed_level1=2,
        enhanced_measurements=(
            _measurement(winner, 1.0, fidelity=Fidelity.ENHANCED),
            _measurement(registry_config, 1.1, fidelity=Fidelity.ENHANCED),
        ),
        locked_challenger=winner,
        formal_challenger_measurement=winner_measurement,
        formal_comparison=PairedMeasurement(
            incumbent=_measurement(incumbent, 1.01),
            challenger=winner_measurement,
            paired_ratios=(1.02,) * 11 + (1.0,) * 2,
        ),
        stop_reason="completed",
    )
    shapes = {
        "first": _shape("first", batch_size=32, heads=4),
        "second": _shape("second", batch_size=64, heads=16),
        "registry": _shape("registry", batch_size=64, heads=2),
    }
    hardware = _hardware()
    registry_shape = ShapeFingerprint(
        batch_size=64,
        qkv_dim=128,
        heads=2,
        seq_len=128,
        layers=4,
        causal=True,
        ffn_dim=128,
        dtype="float32",
        padding_ratio=0.0,
        input_scale=1.0,
    )
    observed_requests = []

    class _SearchEngine:
        def __init__(self, **kwargs) -> None:
            pass

        def plan(self, request):
            return SimpleNamespace(search_space=_AcceptingSearchSpace())

        def run(self, request):
            observed_requests.append(request)
            if request.case_id == "first":
                return first_result
            return _completed_result(winner, 1.0, incumbent=request.incumbent)

    monkeypatch.setattr(search_sweep.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        search_sweep.EnvironmentFingerprint,
        "detect",
        classmethod(lambda cls, device, **kwargs: hardware),
    )
    monkeypatch.setattr(
        search_sweep.HardwareCapabilities,
        "detect",
        classmethod(lambda cls, device: object()),
    )
    monkeypatch.setattr(
        search_sweep,
        "load_shape",
        lambda project_root, case_id: shapes[case_id],
    )
    monkeypatch.setattr(
        search_sweep, "load_shapes", lambda project_root: tuple(shapes.values())
    )
    monkeypatch.setattr(search_sweep, "load_study_summaries", lambda storage: ())
    monkeypatch.setattr(search_sweep, "resolve_deployed_config", lambda **kwargs: None)
    monkeypatch.setattr(
        search_sweep,
        "iter_deployed_configs",
        lambda **kwargs: ((registry_shape, registry_config),),
    )
    monkeypatch.setattr(search_sweep, "publish_deployed_config", lambda **kwargs: None)
    monkeypatch.setattr(search_sweep, "SearchStorage", lambda root: object())
    monkeypatch.setattr(search_sweep, "PlanBuilder", lambda: object())
    monkeypatch.setattr(search_sweep, "BenchmarkEvaluator", lambda **kwargs: object())
    monkeypatch.setattr(search_sweep, "SearchEngine", _SearchEngine)

    observed_shapes = []
    result = search_sweep.SearchSweep(observed_shapes.append).run(
        search_sweep.SearchSweepRequest(
            project_root=search_sweep.Path("."),
            case_ids=("first", "second"),
            budget_seconds=1.0,
        )
    )

    assert result.exit_code == 0
    assert observed_requests[0].warm_starts == (registry_config,)
    assert observed_requests[1].warm_starts == (registry_config, winner)
    assert observed_requests[0].case_id == "first"
    assert observed_requests[1].case_id == "second"
    assert [item.case_id for item in observed_shapes] == ["first", "second"]


def test_service_stops_the_shape_sweep_immediately_after_interrupt(monkeypatch) -> None:
    shapes = {
        "first": _shape("first", batch_size=32, heads=4),
        "second": _shape("second", batch_size=64, heads=4),
    }
    observed_cases: list[str] = []

    class _SearchEngine:
        def __init__(self, **kwargs) -> None:
            pass

        def plan(self, request):
            return SimpleNamespace(search_space=_AcceptingSearchSpace())

        def run(self, request):
            observed_cases.append(request.case_id)
            return SearchResult(
                incumbent_config=request.incumbent,
                selected_config=request.incumbent,
                selected_measurement=None,
                branch_count=1,
                completed_level1=0,
                enhanced_measurements=(),
                locked_challenger=None,
                formal_challenger_measurement=None,
                formal_comparison=None,
                stop_reason="interrupted",
            )

    monkeypatch.setattr(search_sweep.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        search_sweep.EnvironmentFingerprint,
        "detect",
        classmethod(lambda cls, device, **kwargs: _hardware()),
    )
    monkeypatch.setattr(
        search_sweep.HardwareCapabilities,
        "detect",
        classmethod(lambda cls, device: object()),
    )
    monkeypatch.setattr(
        search_sweep,
        "load_shape",
        lambda project_root, case_id: shapes[case_id],
    )
    monkeypatch.setattr(
        search_sweep, "load_shapes", lambda project_root: tuple(shapes.values())
    )
    monkeypatch.setattr(search_sweep, "load_study_summaries", lambda storage: ())
    monkeypatch.setattr(search_sweep, "resolve_deployed_config", lambda **kwargs: None)
    monkeypatch.setattr(search_sweep, "iter_deployed_configs", lambda **kwargs: ())
    monkeypatch.setattr(search_sweep, "SearchStorage", lambda root: object())
    monkeypatch.setattr(search_sweep, "PlanBuilder", lambda: object())
    monkeypatch.setattr(search_sweep, "BenchmarkEvaluator", lambda **kwargs: object())
    monkeypatch.setattr(search_sweep, "SearchEngine", _SearchEngine)

    result = search_sweep.SearchSweep().run(
        search_sweep.SearchSweepRequest(
            project_root=search_sweep.Path("."),
            case_ids=("first", "second"),
            budget_seconds=1.0,
        )
    )

    assert result.exit_code == 130
    assert observed_cases == ["first"]


def test_meta_warm_start_uses_nearest_tasks_and_target_acceptance() -> None:
    target = _shape("target", batch_size=64, heads=4)
    nearest = _shape("nearest", batch_size=32, heads=4)
    second = _shape("second", batch_size=64, heads=2)
    distant = _shape("distant", batch_size=1, heads=16, seq_len=1024)
    portable = portable_config()
    graph = _graph_config()
    incompatible = ConfigSpec(
        program=portable.program,
        schedule=ScheduleConfig(runtime=RuntimeBackend.COMPILED_FORWARD),
    )

    class _SearchSpace:
        def accepted(self, config: ConfigSpec) -> bool:
            return config != incompatible

    candidates = [
        WarmStartCandidate(nearest, search_sweep.RunVariant(), portable, 0, 0),
        WarmStartCandidate(nearest, search_sweep.RunVariant(), portable, 1, 1),
        WarmStartCandidate(nearest, search_sweep.RunVariant(), incompatible, 0, 2),
        WarmStartCandidate(second, search_sweep.RunVariant(), graph, 0, 3),
        WarmStartCandidate(distant, search_sweep.RunVariant(), graph, 0, 4),
    ]

    selected = select_meta_warm_starts(
        candidates=candidates,
        target=target,
        variant=search_sweep.RunVariant(),
        reference_shapes=(target, nearest, second, distant),
        incumbent=None,
        search_space=_SearchSpace(),
        limit=1,
    )

    assert selected == (portable,)


def test_meta_warm_start_does_not_cross_scope_or_run_variant() -> None:
    target = _shape("target", batch_size=64, heads=4)
    valid_source = _shape("valid", batch_size=32, heads=4)
    streamed_source = _shape("official_14", batch_size=32, heads=4)
    variant = search_sweep.RunVariant()
    graph = _graph_config()

    candidates = (
        WarmStartCandidate(valid_source, variant, graph, 0, 0),
        WarmStartCandidate(
            streamed_source,
            variant,
            portable_config(),
            0,
            1,
        ),
        WarmStartCandidate(
            valid_source,
            search_sweep.RunVariant(dtype="float16"),
            portable_config(),
            0,
            2,
        ),
    )

    selected = select_meta_warm_starts(
        candidates=candidates,
        target=target,
        variant=variant,
        reference_shapes=(target, valid_source, streamed_source),
        incumbent=None,
        search_space=_AcceptingSearchSpace(),
    )

    assert selected == (graph,)


def test_best_screen_candidate_uses_only_the_exact_environment() -> None:
    shape = _shape("source", batch_size=64, heads=4)
    variant = search_sweep.RunVariant()
    portable = portable_config()
    graph = _graph_config()

    def summary(environment: str, config: ConfigSpec, latency_ms: float):
        measurement = _measurement(
            config,
            latency_ms,
            fidelity=Fidelity.SCREEN,
        )
        return SimpleNamespace(
            study_name=StudyIdentity(
                case_id=shape.case_id,
                branch_id=f"branch-{config.config_id}",
                environment=environment,
            ).study_name,
            best_trial=optuna.trial.create_trial(
                value=latency_ms,
                user_attrs=measurement.to_user_attrs(config),
            ),
        )

    selected = best_screen_candidates(
        (
            summary("current", portable, 2.0),
            summary("other", graph, 1.0),
        ),
        shape=shape,
        variant=variant,
        environment="current",
        source_order=7,
    )

    assert len(selected) == 1
    assert selected[0].config == portable
    assert selected[0].source_order == 7
