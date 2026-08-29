from __future__ import annotations

from types import SimpleNamespace

from autotune import search_sweep
from autotune.evaluation import (
    ConstraintVector,
    EvaluationScope,
    Fidelity,
    PairedMeasurement,
    TrialMeasurement,
)
from autotune.search_engine import SearchResult
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
            paired_ratios=(1.01,) * 10 + (1.0,) * 3,
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
            paired_ratios=(1.01,) * 10 + (1.0,) * 3,
        ),
        stop_reason="completed",
    )
    shapes = {
        "first": _shape("first", batch_size=32, heads=4),
        "second": _shape("second", batch_size=64, heads=16),
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
    assert observed_requests[1].warm_starts == (
        registry_config,
        winner,
    )
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


def test_family_filter_deduplicates_limits_and_requires_plan_compatibility() -> None:
    target = ShapeFingerprint(
        batch_size=64,
        qkv_dim=128,
        heads=4,
        seq_len=128,
        layers=4,
        causal=True,
        ffn_dim=128,
        dtype="float32",
        padding_ratio=0.0,
        input_scale=1.0,
    )
    portable = portable_config()
    graph = _graph_config()
    incompatible = ConfigSpec(
        program=portable.program,
        schedule=ScheduleConfig(runtime=RuntimeBackend.COMPILED_FORWARD),
    )
    other_family = ShapeFingerprint(**{**target.to_dict(), "seq_len": 256, "heads": 2})
    same_family = ShapeFingerprint(
        **{**target.to_dict(), "batch_size": 32, "heads": 16}
    )

    class _SearchSpace:
        def accepted(self, config: ConfigSpec) -> bool:
            return config != incompatible

        def branch_for(self, config: ConfigSpec) -> object | None:
            return object()

    candidates = [
        search_sweep._WarmStartCandidate(same_family, portable, 0, 0),
        search_sweep._WarmStartCandidate(same_family, portable, 1, 1),
        search_sweep._WarmStartCandidate(same_family, incompatible, 0, 2),
        search_sweep._WarmStartCandidate(other_family, graph, 0, 3),
        search_sweep._WarmStartCandidate(same_family, graph, 0, 4),
    ]

    selected = search_sweep._compatible_family_warm_starts(
        candidates=candidates,
        target=target,
        incumbent=None,
        search_space=_SearchSpace(),
        limit=1,
    )

    assert selected == (portable,)
