import pytest
import torch
from torch import nn

import benchmarking.measure as measure_module
from autotune.evaluation import (
    RESIDENT_PROTOCOLS,
    STREAMED_PROTOCOLS,
    ConstraintVector,
    EvaluationScope,
    Fidelity,
    PairedMeasurement,
    TrialMeasurement,
    classify_infeasible_exception,
)
from autotune.promotion import (
    PROMOTION_BASE_RATIO,
    PROMOTION_BASE_WINS,
    PROMOTION_MAX_BLOCKS,
    PromotionDecision,
    promotion_decision,
    promotion_should_stop,
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


def test_interleaved_timings_stops_after_a_terminal_sequential_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incumbent = nn.Identity()
    challenger = nn.Identity()
    calls = 0

    monkeypatch.setattr(measure_module.official, "warmup_model", lambda *args: None)

    def benchmark_once(
        model: nn.Module,
        _x: torch.Tensor,
        _mask: torch.Tensor,
        _iterations: int,
        _device: torch.device,
    ) -> list[float]:
        nonlocal calls
        calls += 1
        return [11.0] if model is incumbent else [10.0]

    monkeypatch.setattr(measure_module.official, "benchmark_once", benchmark_once)
    x = torch.zeros(1)
    mask = torch.ones(1, dtype=torch.bool)
    _, _, ratios = measure_module._interleaved_timings(
        incumbent,
        (x, mask),
        challenger,
        (x, mask),
        MeasurementProtocol(
            accuracy_trials=1,
            warmup=0,
            repeats=1,
            rounds=PROMOTION_MAX_BLOCKS,
        ),
        torch.device("cpu"),
        stop_when=promotion_should_stop,
    )

    assert len(ratios) == 6
    assert calls == 12
    assert promotion_decision(ratios) is PromotionDecision.PROMOTE


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
        paired_ratios=(
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.00,
            1.02,
            1.02,
            1.02,
            1.03,
            1.04,
            1.05,
            1.50,
        ),
    )

    assert comparison.speedup == pytest.approx(1.02)
    assert comparison.promotion_wins == 7
    assert not comparison.promotes


def test_sequential_promotion_uses_stronger_evidence_at_earlier_looks() -> None:
    incumbent = _formal_measurement("incumbent", 20.0)
    challenger = _formal_measurement("challenger", 10.0)
    strong = PairedMeasurement(
        incumbent=incumbent,
        challenger=challenger,
        paired_ratios=(1.10,) * 6,
    )
    medium = PairedMeasurement(
        incumbent=incumbent,
        challenger=challenger,
        paired_ratios=(1.05,) * 8 + (1.03,),
    )
    close = PairedMeasurement(
        incumbent=incumbent,
        challenger=challenger,
        paired_ratios=(PROMOTION_BASE_RATIO,) * 11 + (1.0,) * 2,
    )
    insufficient = PairedMeasurement(
        incumbent=incumbent,
        challenger=challenger,
        paired_ratios=(PROMOTION_BASE_RATIO,) * 10 + (1.0,) * 3,
    )

    assert PROMOTION_MAX_BLOCKS == 13
    assert PROMOTION_BASE_WINS == 11
    assert strong.promotes
    assert medium.promotes
    assert close.promotes
    assert not insufficient.promotes


def test_sequential_rule_rejects_when_final_target_is_unreachable() -> None:
    ratios = (1.0, 1.01, 1.019)

    assert promotion_decision(ratios) is PromotionDecision.REJECT


def test_sequential_false_promotion_bound_is_below_five_percent() -> None:
    strong_look = 1 / 64
    medium_look = 10 / 512
    final_look = 92 / 8192

    assert strong_look + medium_look + final_look < 0.05


def test_incomplete_paired_measurement_cannot_promote() -> None:
    comparison = PairedMeasurement(
        incumbent=_formal_measurement("incumbent", 20.0),
        challenger=_formal_measurement("challenger", 10.0),
        paired_ratios=(),
    )

    assert comparison.speedup is None
    assert comparison.promotion_wins == 0
    assert not comparison.promotes

    with pytest.raises(ValueError, match="terminal sequential result"):
        PairedMeasurement(
            incumbent=_formal_measurement("incumbent", 20.0),
            challenger=_formal_measurement("challenger", 10.0),
            paired_ratios=(1.03,) * 5,
        )


def test_formal_promotion_protocols_use_thirteen_paired_blocks() -> None:
    resident = RESIDENT_PROTOCOLS[Fidelity.FORMAL]
    streamed = STREAMED_PROTOCOLS[Fidelity.FORMAL]

    assert (resident.warmup, resident.repeats, resident.rounds) == (20, 25, 13)
    assert (streamed.repeats, streamed.rounds) == (1, 13)
    assert streamed.full_logical_batch


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
    for message in (
        "compiled FFN compilation failed",
        "full-stack compiled forward compilation failed",
        "compiled residual LayerNorm is ineligible for the requested inputs",
        "compiled residual LayerNorm execution failed",
    ):
        assert (
            classify_infeasible_exception(RuntimeError(message))
            == "candidate_execution_failed"
        )
    assert classify_infeasible_exception(RuntimeError("worker connection lost")) is None
    assert classify_infeasible_exception(ValueError("unexpected Python bug")) is None
