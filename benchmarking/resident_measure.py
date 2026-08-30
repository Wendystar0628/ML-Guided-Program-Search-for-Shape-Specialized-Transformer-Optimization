"""Measurement and profiling for resident Shapes 01-13."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from benchmarking.protocols import MeasurementProtocol, RunVariant, TransformerShape
from official import torch_transformer_benchmark as official
from solution.config import ConfigSpec
from solution.transformer import UserOptimizedTransformer, copy_model_weights

from .measurement_core import (
    BenchmarkResult,
    PairedBenchmarkResult,
    TimingStats,
    _build_models,
    _correctness,
    _execution_signatures,
    _interleaved_timings,
    _official_config,
    _peak_memory,
    _timings,
)


def measure_paired_resident_configs(
    shape: TransformerShape,
    challenger_config: ConfigSpec,
    incumbent_config: ConfigSpec,
    variant: RunVariant,
    protocol: MeasurementProtocol,
    device: torch.device,
    stop_when: Callable[[tuple[float, ...]], bool] | None,
) -> PairedBenchmarkResult:
    model_config = _official_config(shape)
    dtype = official.resolve_dtype(variant.dtype)
    baseline = official.BaselineTransformer(model_config)
    incumbent = UserOptimizedTransformer(model_config)
    challenger = UserOptimizedTransformer(model_config)
    copy_model_weights(baseline, incumbent, strict=True)
    copy_model_weights(baseline, challenger, strict=True)
    baseline = baseline.to(device=device, dtype=dtype).eval()
    incumbent = incumbent.to(device=device, dtype=dtype).eval()
    challenger = challenger.to(device=device, dtype=dtype).eval()
    incumbent.configure_execution(config=incumbent_config)
    challenger.configure_execution(config=challenger_config)

    incumbent_passed, incumbent_ratio = _correctness(
        baseline,
        incumbent,
        model_config,
        variant,
        protocol,
        device,
    )
    challenger_passed, challenger_ratio = _correctness(
        baseline,
        challenger,
        model_config,
        variant,
        protocol,
        device,
    )
    del baseline
    x, mask = official.generate_random_case(
        model_config,
        device,
        dtype,
        protocol.seed + 100000,
        variant.padding_ratio,
        variant.input_scale,
    )
    incumbent_expected, incumbent_actual = _execution_signatures(incumbent, x, mask)
    challenger_expected, challenger_actual = _execution_signatures(
        challenger,
        x,
        mask,
    )
    incumbent_samples, challenger_samples, paired_ratios = _interleaved_timings(
        incumbent,
        (x, mask),
        challenger,
        (x, mask),
        protocol,
        device,
        stop_when=stop_when,
    )
    incumbent_peak = _peak_memory(incumbent, x, mask, device)
    challenger_peak = _peak_memory(challenger, x, mask, device)
    return PairedBenchmarkResult(
        incumbent=BenchmarkResult(
            case_id=shape.case_id,
            config=incumbent_config,
            passed=incumbent_passed,
            max_tolerance_ratio=incumbent_ratio,
            optimized=TimingStats.from_samples(incumbent_samples),
            peak_memory_bytes=incumbent_peak,
            expected_execution_signature=incumbent_expected,
            actual_execution_signature=incumbent_actual,
        ),
        challenger=BenchmarkResult(
            case_id=shape.case_id,
            config=challenger_config,
            passed=challenger_passed,
            max_tolerance_ratio=challenger_ratio,
            optimized=TimingStats.from_samples(challenger_samples),
            peak_memory_bytes=challenger_peak,
            expected_execution_signature=challenger_expected,
            actual_execution_signature=challenger_actual,
        ),
        paired_ratios=paired_ratios,
    )


def measure_resident_config(
    shape: TransformerShape,
    config: ConfigSpec,
    variant: RunVariant,
    protocol: MeasurementProtocol,
    device: torch.device,
    *,
    include_baseline: bool = False,
) -> BenchmarkResult:
    baseline, solution = _build_models(
        shape,
        variant,
        device,
        config,
        include_baseline=True,
    )
    assert baseline is not None
    model_config = _official_config(shape)
    passed, ratio = _correctness(
        baseline,
        solution,
        model_config,
        variant,
        protocol,
        device,
    )
    dtype = official.resolve_dtype(variant.dtype)
    x, mask = official.generate_random_case(
        model_config,
        device,
        dtype,
        protocol.seed + 100000,
        variant.padding_ratio,
        variant.input_scale,
    )
    expected, actual = _execution_signatures(solution, x, mask)
    baseline_stats = None
    if include_baseline:
        baseline_samples, optimized_samples, _ = _interleaved_timings(
            baseline,
            (x, mask),
            solution,
            (x, mask),
            protocol,
            device,
        )
        baseline_stats = TimingStats.from_samples(baseline_samples)
    else:
        optimized_samples = _timings(solution, x, mask, protocol, device)
    peak = _peak_memory(solution, x, mask, device)
    return BenchmarkResult(
        case_id=shape.case_id,
        config=config,
        passed=passed,
        max_tolerance_ratio=ratio,
        optimized=TimingStats.from_samples(optimized_samples),
        peak_memory_bytes=peak,
        expected_execution_signature=expected,
        actual_execution_signature=actual,
        baseline=baseline_stats,
    )


def profile_config(
    shape: TransformerShape,
    config: ConfigSpec,
    variant: RunVariant,
    device: str | torch.device,
    *,
    iterations: int = 5,
) -> list[dict[str, Any]]:
    """Return the hottest operations for one explicit resident program."""

    if shape.streamed:
        raise ValueError("profile currently targets resident shapes")
    resolved_device = official.resolve_device(str(device))
    _, solution = _build_models(
        shape,
        variant,
        resolved_device,
        config,
        include_baseline=False,
    )
    dtype = official.resolve_dtype(variant.dtype)
    x, mask = official.generate_random_case(
        _official_config(shape),
        resolved_device,
        dtype,
        101234,
        variant.padding_ratio,
        variant.input_scale,
    )
    official.warmup_model(solution, x, mask, 2, resolved_device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if resolved_device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with (
        torch.profiler.profile(activities=activities) as profiler,
        torch.inference_mode(),
    ):
        for _ in range(iterations):
            solution(x, mask)
    events = list(profiler.key_averages())
    events.sort(
        key=lambda event: float(
            getattr(event, "self_device_time_total", 0.0)
            or getattr(event, "self_cpu_time_total", 0.0)
        ),
        reverse=True,
    )
    return [
        {
            "name": event.key,
            "self_time_us_per_forward": float(
                getattr(event, "self_device_time_total", 0.0)
                or getattr(event, "self_cpu_time_total", 0.0)
            )
            / iterations,
            "calls": int(event.count),
        }
        for event in events[:20]
    ]


__all__ = [
    "measure_paired_resident_configs",
    "measure_resident_config",
    "profile_config",
]
