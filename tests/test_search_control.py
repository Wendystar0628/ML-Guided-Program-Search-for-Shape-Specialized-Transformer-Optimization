from __future__ import annotations

import json
from pathlib import Path

from autotune.evaluation import (
    PROMOTION_BLOCK_WIN_RATIO,
    ConstraintVector,
    EvaluationScope,
    Fidelity,
    PairedMeasurement,
    TrialMeasurement,
)
from autotune.search_engine import SearchEngine, SearchResult
from deployment.registry import (
    EnvironmentFingerprint,
    ShapeFingerprint,
    publish_deployed_config,
    resolve_deployed_config,
)
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


def _compiled_config() -> ConfigSpec:
    return ConfigSpec(
        program=portable_config().program,
        schedule=ScheduleConfig(runtime=RuntimeBackend.COMPILED_FORWARD),
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


def test_deployment_uses_the_single_paired_block_rule() -> None:
    incumbent = portable_config()
    challenger = _graph_config()
    incumbent_measurement = _measurement(incumbent, 101.0)
    challenger_measurement = _measurement(challenger, 100.0)
    comparison = PairedMeasurement(
        incumbent=incumbent_measurement,
        challenger=challenger_measurement,
        paired_ratios=(PROMOTION_BLOCK_WIN_RATIO,) * 10 + (1.0,) * 3,
    )
    result = SearchResult(
        incumbent_config=incumbent,
        selected_config=challenger,
        selected_measurement=challenger_measurement,
        branch_count=1,
        completed_level1=1,
        enhanced_measurements=(
            _measurement(challenger, 100.0, fidelity=Fidelity.ENHANCED),
        ),
        locked_challenger=challenger,
        formal_challenger_measurement=challenger_measurement,
        formal_comparison=comparison,
        stop_reason="completed",
    )

    assert comparison.promotes
    assert result.deployment_approved


def test_formal_locks_only_the_fastest_feasible_enhanced_challenger() -> None:
    incumbent = portable_config()
    slower = _graph_config()
    faster = _compiled_config()
    enhanced = (
        (incumbent, _measurement(incumbent, 0.5, fidelity=Fidelity.ENHANCED)),
        (slower, _measurement(slower, 2.0, fidelity=Fidelity.ENHANCED)),
        (faster, _measurement(faster, 1.0, fidelity=Fidelity.ENHANCED)),
    )

    locked = SearchEngine._lock_challenger(enhanced, incumbent=incumbent)

    assert locked == faster


def test_only_completed_formal_selection_is_deployable() -> None:
    config = portable_config()
    interrupted = SearchResult(
        incumbent_config=config,
        selected_config=config,
        selected_measurement=_measurement(config, 1.0, fidelity=Fidelity.SCREEN),
        branch_count=1,
        completed_level1=1,
        enhanced_measurements=(),
        locked_challenger=None,
        formal_challenger_measurement=None,
        formal_comparison=None,
        stop_reason="interrupted",
    )
    completed = SearchResult(
        incumbent_config=None,
        selected_config=config,
        selected_measurement=_measurement(config, 1.0),
        branch_count=1,
        completed_level1=1,
        enhanced_measurements=(
            _measurement(config, 1.0, fidelity=Fidelity.ENHANCED),
        ),
        locked_challenger=config,
        formal_challenger_measurement=_measurement(config, 1.0),
        formal_comparison=None,
        stop_reason="completed",
    )

    assert not interrupted.deployment_approved
    assert completed.deployment_approved


def test_deployment_key_separates_input_variants(tmp_path: Path) -> None:
    path = tmp_path / "deployed.json"
    hardware = _hardware()
    default_shape = ShapeFingerprint(
        batch_size=1,
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
    variant_shape = ShapeFingerprint(
        batch_size=1,
        qkv_dim=128,
        heads=4,
        seq_len=128,
        layers=4,
        causal=True,
        ffn_dim=128,
        dtype="float32",
        padding_ratio=0.25,
        input_scale=2.0,
    )
    default_config = portable_config()
    variant_config = _graph_config()

    publish_deployed_config(
        hardware=hardware,
        shape=default_shape,
        config=default_config,
        path=path,
    )
    publish_deployed_config(
        hardware=hardware,
        shape=variant_shape,
        config=variant_config,
        path=path,
    )

    assert (
        resolve_deployed_config(
            hardware=hardware,
            shape=default_shape,
            path=path,
        )
        == default_config
    )
    assert (
        resolve_deployed_config(
            hardware=hardware,
            shape=variant_shape,
            path=path,
        )
        == variant_config
    )


def test_checked_in_deployments_use_complete_shape_keys() -> None:
    path = Path(__file__).resolve().parents[1] / "deployment" / "deployed_configs.json"
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["schema_version"] == 2

    for bundle in document["bundles"]:
        for entry in bundle["entries"]:
            key = ShapeFingerprint.from_dict(entry["shape"])
            assert key.padding_ratio == 0.0
            assert key.input_scale == 1.0
