import pytest
import torch
from torch import nn

import benchmarking.measure as measure_module
from autotune.evaluator import (
    ConstraintVector,
    EvaluationScope,
    Fidelity,
    PairedMeasurement,
    TrialMeasurement,
    classify_infeasible_exception,
)
from benchmarking.measure import measure_paired_configs
from benchmarking.protocols import MeasurementProtocol, RunVariant, TransformerShape
from solution.config import portable_config


def _formal_measurement(config_id: str, objective_ms: float) -> TrialMeasurement:
    return TrialMeasurement(
        config_id=config_id,
        fidelity=Fidelity.FORMAL,
        scope=EvaluationScope.RESIDENT,
        objective_ms=objective_ms,
        median_ms=objective_ms,
        constraints=ConstraintVector(),
    )


def test_formal_rounds_alternate_ab_ba_and_keep_paired_ratios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incumbent = nn.Identity()
    challenger = nn.Identity()
    labels = {id(incumbent): "A", id(challenger): "B"}
    samples = {
        "A": iter(([4.0, 6.0], [6.0, 10.0], [3.0, 3.0])),
        "B": iter(([2.0, 2.0], [4.0, 4.0], [2.0, 4.0])),
    }
    order: list[str] = []

    monkeypatch.setattr(measure_module.official, "warmup_model", lambda *args: None)

    def benchmark_once(
        model: nn.Module,
        _x: torch.Tensor,
        _mask: torch.Tensor,
        _iterations: int,
        _device: torch.device,
    ) -> list[float]:
        label = labels[id(model)]
        order.append(label)
        return list(next(samples[label]))

    monkeypatch.setattr(measure_module.official, "benchmark_once", benchmark_once)
    x = torch.zeros(1)
    mask = torch.ones(1, dtype=torch.bool)
    incumbent_samples, challenger_samples, ratios = measure_module._interleaved_timings(
        incumbent,
        (x, mask),
        challenger,
        (x, mask),
        MeasurementProtocol(
            accuracy_trials=1,
            warmup=2,
            repeats=2,
            rounds=3,
        ),
        torch.device("cpu"),
    )

    assert order == ["A", "B", "B", "A", "A", "B"]
    assert incumbent_samples == [4.0, 6.0, 6.0, 10.0, 3.0, 3.0]
    assert challenger_samples == [2.0, 2.0, 4.0, 4.0, 2.0, 4.0]
    assert ratios == pytest.approx((2.5, 2.0, 1.0))


def test_public_paired_measurement_returns_one_ratio_per_round() -> None:
    result = measure_paired_configs(
        TransformerShape(
            case_id="tiny",
            batch_size=1,
            seq_len=2,
            d_model=4,
            num_heads=1,
            ffn_dim=8,
            num_layers=1,
            causal=True,
        ),
        portable_config(),
        portable_config(),
        RunVariant(),
        MeasurementProtocol(
            accuracy_trials=1,
            warmup=0,
            repeats=1,
            rounds=2,
        ),
        "cpu",
    )

    assert result.incumbent.case_id == "tiny"
    assert result.challenger.case_id == "tiny"
    assert len(result.paired_ratios) == 2
    assert result.median_speedup > 0.0


def test_promotion_speedup_is_the_median_of_paired_ratios() -> None:
    comparison = PairedMeasurement(
        incumbent=_formal_measurement("incumbent", 20.0),
        challenger=_formal_measurement("challenger", 10.0),
        paired_ratios=(1.50, 1.01, 1.02),
        exceeds_noise_margin=True,
    )

    assert comparison.speedup == pytest.approx(1.02)


def test_paired_measurement_rejects_missing_rounds() -> None:
    with pytest.raises(ValueError, match="paired_ratios"):
        PairedMeasurement(
            incumbent=_formal_measurement("incumbent", 20.0),
            challenger=_formal_measurement("challenger", 10.0),
            paired_ratios=(),
            exceeds_noise_margin=False,
        )


def test_only_known_config_domain_exceptions_are_infeasible() -> None:
    assert classify_infeasible_exception(torch.OutOfMemoryError("OOM")) == (
        "out_of_memory"
    )
    assert (
        classify_infeasible_exception(
            RuntimeError("CUDA launch failed: too many resources requested")
        )
        == "runtime_resource_exhausted"
    )
    assert (
        classify_infeasible_exception(
            RuntimeError("out of resource: shared memory, Required: 65536")
        )
        == "runtime_resource_exhausted"
    )

    wrapped = RuntimeError("compiled execution failed")
    wrapped.__cause__ = torch.OutOfMemoryError("CUDA out of memory")
    assert classify_infeasible_exception(wrapped) == "out_of_memory"
    assert classify_infeasible_exception(RuntimeError("worker connection lost")) is None
    assert classify_infeasible_exception(ValueError("unexpected Python bug")) is None
