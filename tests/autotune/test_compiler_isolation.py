from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from autotune import compiler_isolation, search_sweep
from autotune.evaluation import Fidelity
from benchmarking.measure import BenchmarkResult, PairedBenchmarkResult, TimingStats
from benchmarking.protocols import RunVariant, TransformerShape
from solution.config import (
    ConfigSpec,
    FFNBackend,
    ResidualNormBackend,
    RuntimeBackend,
    portable_config,
)


def _config(
    *,
    runtime: RuntimeBackend = RuntimeBackend.EAGER,
    ffn: FFNBackend = FFNBackend.TORCH,
    residual_norm: ResidualNormBackend = ResidualNormBackend.TORCH,
) -> ConfigSpec:
    portable = portable_config()
    return replace(
        portable,
        program=replace(portable.program, ffn=ffn, residual_norm=residual_norm),
        schedule=replace(portable.schedule, runtime=runtime),
    )


@pytest.mark.parametrize(
    "config",
    (
        _config(runtime=RuntimeBackend.COMPILED_FORWARD),
        _config(ffn=FFNBackend.COMPILED),
        _config(residual_norm=ResidualNormBackend.COMPILED),
    ),
)
def test_compile_heavy_configurations_are_detected(config: ConfigSpec) -> None:
    assert compiler_isolation.uses_torch_compile(config)


def test_portable_configuration_does_not_reset_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resets: list[None] = []
    monkeypatch.setattr(torch.compiler, "reset", lambda: resets.append(None))

    with compiler_isolation.isolate_compiler_state(portable_config()):
        pass

    assert resets == []


def test_compiler_isolation_resets_before_and_after_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(torch.compiler, "reset", lambda: events.append("reset"))

    with (
        pytest.raises(RuntimeError, match="measurement failed"),
        compiler_isolation.isolate_compiler_state(
            _config(ffn=FFNBackend.COMPILED)
        ),
    ):
        events.append("measure")
        raise RuntimeError("measurement failed")

    assert events == ["reset", "measure", "reset"]


def test_compiler_isolation_prunes_impossible_shared_memory_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.compiler, "reset", lambda: None)
    original = (
        compiler_isolation.inductor_config
        .max_autotune_prune_choices_based_on_shared_mem
    )

    with compiler_isolation.isolate_compiler_state(
        _config(runtime=RuntimeBackend.COMPILED_FORWARD)
    ):
        assert (
            compiler_isolation.inductor_config
            .max_autotune_prune_choices_based_on_shared_mem
        )

    assert (
        compiler_isolation.inductor_config
        .max_autotune_prune_choices_based_on_shared_mem
        is original
    )


def _shape() -> TransformerShape:
    return TransformerShape("tiny", 1, 2, 4, 1, 8, 1, True)


def _result(config: ConfigSpec) -> BenchmarkResult:
    signature = {"config_id": config.config_id}
    return BenchmarkResult(
        case_id="tiny",
        config=config,
        passed=True,
        max_tolerance_ratio=0.0,
        optimized=TimingStats(median_ms=1.0, p90_ms=1.0),
        peak_memory_bytes=0,
        expected_execution_signature=signature,
        actual_execution_signature=dict(signature),
    )


def test_evaluate_isolates_one_compile_heavy_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(runtime=RuntimeBackend.COMPILED_FORWARD)
    events: list[str] = []
    monkeypatch.setattr(torch.compiler, "reset", lambda: events.append("reset"))

    def measure(*args: object, **kwargs: object) -> BenchmarkResult:
        events.append("measure")
        return _result(config)

    monkeypatch.setattr(search_sweep, "measure_config", measure)
    evaluator = search_sweep.BenchmarkEvaluator(
        shape=_shape(),
        variant=RunVariant(),
        device=torch.device("cpu"),
    )

    evaluator.evaluate(config, Fidelity.SCREEN)

    assert events == ["reset", "measure", "reset"]


def test_compare_isolates_a_compile_heavy_pair_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    challenger = portable_config()
    incumbent = _config(residual_norm=ResidualNormBackend.COMPILED)
    events: list[str] = []
    monkeypatch.setattr(torch.compiler, "reset", lambda: events.append("reset"))

    def measure(*args: object, **kwargs: object) -> PairedBenchmarkResult:
        events.append("measure")
        return PairedBenchmarkResult(
            incumbent=_result(incumbent),
            challenger=_result(challenger),
            paired_ratios=(1.10,) * 6,
        )

    monkeypatch.setattr(search_sweep, "measure_paired_configs", measure)
    evaluator = search_sweep.BenchmarkEvaluator(
        shape=_shape(),
        variant=RunVariant(),
        device=torch.device("cpu"),
    )

    evaluator.compare(challenger, incumbent)

    assert events == ["reset", "measure", "reset"]
