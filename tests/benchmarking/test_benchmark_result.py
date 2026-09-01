import math

import pytest
import torch
from torch import nn

import benchmarking.measure as measure_module
from benchmarking import measurement_core, resident_measure
from benchmarking.measure import BenchmarkResult, TimingStats
from benchmarking.protocols import MeasurementProtocol, RunVariant, TransformerShape
from solution.config import portable_config


def test_benchmark_result_serializes_only_the_compact_public_fields() -> None:
    config = portable_config()
    signature = {"config_id": config.config_id, "runtime_backend": "eager"}
    result = BenchmarkResult(
        case_id="official_01",
        config=config,
        passed=True,
        max_tolerance_ratio=0.25,
        optimized=TimingStats(median_ms=2.0, p90_ms=2.2),
        baseline=TimingStats(median_ms=4.0, p90_ms=4.5),
        peak_memory_bytes=1024,
        expected_execution_signature=signature,
        actual_execution_signature=dict(signature),
    )

    assert result.to_dict() == {
        "case_id": "official_01",
        "config_id": config.config_id,
        "passed": True,
        "max_tolerance_ratio": 0.25,
        "median_ms": 2.0,
        "p90_ms": 2.2,
        "baseline_median_ms": 4.0,
        "speedup": 2.0,
        "peak_memory_bytes": 1024,
        "execution_matches": True,
    }


def test_shape14_result_serializes_scoped_correctness_evidence() -> None:
    config = portable_config()
    signature = {"config_id": config.config_id, "runtime_backend": "streamed"}
    result = BenchmarkResult(
        case_id="official_14",
        config=config,
        passed=True,
        max_tolerance_ratio=0.25,
        optimized=TimingStats(median_ms=10.0, p90_ms=10.0),
        peak_memory_bytes=2048,
        expected_execution_signature=signature,
        actual_execution_signature=dict(signature),
        local_b1_semantic_pass=True,
        full_logical_execution_completed=True,
        sampled_execution_digest="sampled-digest",
        official_b32_io_pass=None,
        official_b32_io_status="not_available",
    )

    serialized = result.to_dict()

    assert serialized["local_b1_semantic_pass"] is True
    assert serialized["full_logical_execution_completed"] is True
    assert serialized["sampled_execution_digest"] == "sampled-digest"
    assert serialized["official_b32_io_pass"] is None
    assert serialized["official_b32_io_status"] == "not_available"
    assert "output_digest" not in serialized


def test_comparator_rejects_nonzero_error_when_both_tolerances_are_zero() -> None:
    reference = torch.zeros(1)
    candidate = torch.tensor([torch.finfo(torch.float32).eps / 2])

    passed, ratio = measurement_core._comparison_metrics(
        reference,
        candidate,
        rtol=0.0,
        atol=0.0,
    )

    assert not passed
    assert math.isinf(ratio)


def test_comparator_accepts_exact_zero_with_zero_tolerances() -> None:
    passed, ratio = measurement_core._comparison_metrics(
        torch.zeros(2),
        torch.zeros(2),
        rtol=0.0,
        atol=0.0,
    )

    assert passed
    assert ratio == 0.0


def test_measure_config_interleaves_baseline_and_solution_timings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = nn.Identity()
    solution = nn.Identity()
    x = torch.zeros(1)
    mask = torch.ones(1, dtype=torch.bool)
    calls: list[tuple[nn.Module, nn.Module]] = []

    monkeypatch.setattr(
        resident_measure,
        "_build_models",
        lambda *args, **kwargs: (baseline, solution),
    )
    monkeypatch.setattr(
        resident_measure,
        "_correctness",
        lambda *args, **kwargs: (True, 0.25),
    )
    monkeypatch.setattr(
        resident_measure.official,
        "generate_random_case",
        lambda *args, **kwargs: (x, mask),
    )
    monkeypatch.setattr(
        resident_measure,
        "_execution_signatures",
        lambda *args: ({"path": "solution"}, {"path": "solution"}),
    )

    def interleaved(
        incumbent: nn.Module,
        _incumbent_input: tuple[torch.Tensor, torch.Tensor],
        challenger: nn.Module,
        _challenger_input: tuple[torch.Tensor, torch.Tensor],
        *_args: object,
    ) -> tuple[list[float], list[float], tuple[float, ...]]:
        calls.append((incumbent, challenger))
        return [4.0, 6.0], [2.0, 3.0], (2.0,)

    monkeypatch.setattr(resident_measure, "_interleaved_timings", interleaved)
    monkeypatch.setattr(resident_measure, "_peak_memory", lambda *args: 0)
    monkeypatch.setattr(
        resident_measure,
        "_timings",
        lambda *args: pytest.fail("standalone timing must not be used"),
    )

    result = measure_module.measure_config(
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
        RunVariant(),
        MeasurementProtocol(accuracy_trials=1, warmup=0, repeats=1, rounds=2),
        "cpu",
        include_baseline=True,
    )

    assert calls == [(baseline, solution)]
    assert result.baseline is not None
    assert result.baseline.median_ms == 5.0
    assert result.baseline.p90_ms == pytest.approx(5.8)
    assert result.optimized.median_ms == 2.5
    assert result.optimized.p90_ms == pytest.approx(2.9)
