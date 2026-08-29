"""Direct official-shape measurement used by both the CLI and TPE search."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from official import torch_transformer_benchmark as official
from runner.contracts import MeasurementProtocol, RunVariant, TransformerShape
from solution.config import ConfigSpec, RuntimeBackend, ScheduleConfig, portable_config
from solution.transformer import UserOptimizedTransformer, copy_model_weights


@dataclass(frozen=True, slots=True)
class TimingStats:
    median_ms: float
    p90_ms: float

    @classmethod
    def from_samples(cls, samples: list[float]) -> "TimingStats":
        return cls(
            median_ms=float(statistics.median(samples)),
            p90_ms=float(official.percentile(samples, 0.90)),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    case_id: str
    config: ConfigSpec
    passed: bool
    max_tolerance_ratio: float
    optimized: TimingStats
    peak_memory_bytes: int
    expected_execution_signature: dict[str, Any]
    actual_execution_signature: dict[str, Any]
    baseline: TimingStats | None = None

    @property
    def speedup(self) -> float | None:
        if self.baseline is None:
            return None
        return self.baseline.median_ms / self.optimized.median_ms

    @property
    def execution_matches(self) -> bool:
        return self.expected_execution_signature == self.actual_execution_signature

    def to_dict(self) -> dict[str, Any]:
        """Return the compact persisted result; configs and traces live elsewhere."""

        return {
            "case_id": self.case_id,
            "config_id": self.config.config_id,
            "passed": self.passed,
            "max_tolerance_ratio": self.max_tolerance_ratio,
            "median_ms": self.optimized.median_ms,
            "p90_ms": self.optimized.p90_ms,
            "baseline_median_ms": (
                None if self.baseline is None else self.baseline.median_ms
            ),
            "speedup": self.speedup,
            "peak_memory_bytes": self.peak_memory_bytes,
            "execution_matches": self.execution_matches,
        }


def _official_config(
    shape: TransformerShape,
    *,
    batch_size: int | None = None,
) -> official.TransformerConfig:
    config = official.TransformerConfig(
        batch_size=shape.batch_size if batch_size is None else batch_size,
        seq_len=shape.seq_len,
        d_model=shape.d_model,
        num_heads=shape.num_heads,
        ffn_dim=shape.ffn_dim,
        num_layers=shape.num_layers,
        causal=shape.causal,
    )
    config.validate()
    return config


def _build_models(
    shape: TransformerShape,
    variant: RunVariant,
    device: torch.device,
    config: ConfigSpec,
    *,
    batch_size: int | None = None,
    include_baseline: bool,
) -> tuple[nn.Module | None, UserOptimizedTransformer]:
    model_config = _official_config(shape, batch_size=batch_size)
    dtype = official.resolve_dtype(variant.dtype)
    baseline = official.BaselineTransformer(model_config)
    solution = UserOptimizedTransformer(model_config)
    copy_model_weights(baseline, solution, strict=True)
    baseline = baseline.to(device=device, dtype=dtype).eval()
    solution = solution.to(device=device, dtype=dtype).eval()
    solution.configure_execution(config=config)
    if not include_baseline:
        del baseline
        baseline = None
    return baseline, solution


def _max_tolerance_ratio(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    chunk_size: int = 4096,
) -> float:
    """Return the continuous form of the official OR comparator."""

    if reference.shape != candidate.shape:
        return math.inf
    flat_reference = reference.detach().reshape(-1)
    flat_candidate = candidate.detach().reshape(-1)
    maximum = 0.0
    epsilon = torch.finfo(torch.float32).eps
    for start in range(0, flat_reference.numel(), chunk_size):
        ref = flat_reference[start : start + chunk_size].float()
        value = flat_candidate[start : start + chunk_size].float()
        finite = torch.isfinite(ref) & torch.isfinite(value)
        if not bool(finite.all().item()):
            return math.inf
        error = (value - ref).abs()
        absolute_ratio = error / max(float(atol), epsilon)
        relative_ratio = error / (float(rtol) * ref.abs() + epsilon)
        local = torch.minimum(absolute_ratio, relative_ratio).max().item()
        maximum = max(maximum, float(local))
    return maximum


def _execution_signatures(
    model: UserOptimizedTransformer,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.set_execution_observation(True)
    with torch.inference_mode():
        model(x, mask)
    description = model.describe_execution_path()
    model.set_execution_observation(False)
    expected = description.get("expected_execution_signature")
    actual = description.get("observed_execution_signature")
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        raise RuntimeError("solution did not report its execution path")
    return dict(expected), dict(actual)


def _correctness(
    reference: nn.Module,
    candidate: nn.Module,
    model_config: official.TransformerConfig,
    variant: RunVariant,
    protocol: MeasurementProtocol,
    device: torch.device,
) -> float:
    dtype = official.resolve_dtype(variant.dtype)
    maximum = 0.0
    with torch.inference_mode():
        for trial in range(protocol.accuracy_trials):
            x, mask = official.generate_random_case(
                model_config,
                device,
                dtype,
                protocol.seed + trial,
                variant.padding_ratio,
                variant.input_scale,
            )
            reference_output = reference(x, mask)
            candidate_output = candidate(x, mask)
            maximum = max(
                maximum,
                _max_tolerance_ratio(
                    reference_output,
                    candidate_output,
                    rtol=protocol.rtol,
                    atol=protocol.atol,
                ),
            )
    return maximum


def _timings(
    model: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    protocol: MeasurementProtocol,
    device: torch.device,
) -> list[float]:
    official.warmup_model(model, x, mask, protocol.warmup, device)
    samples: list[float] = []
    for _ in range(protocol.rounds):
        samples.extend(
            official.benchmark_once(model, x, mask, protocol.repeats, device)
        )
    return samples


def _peak_memory(
    model: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> int:
    if device.type != "cuda":
        return 0
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        model(x, mask)
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))


def _inner_streamed_config(config: ConfigSpec) -> ConfigSpec:
    if config.schedule.runtime is not RuntimeBackend.STREAMED:
        raise ValueError("Shape 14 requires a streamed ConfigSpec")
    schedule = config.schedule
    return ConfigSpec(
        program=config.program,
        schedule=ScheduleConfig(
            runtime=RuntimeBackend.EAGER,
            attention_launch=schedule.attention_launch,
            residual_norm_launch=schedule.residual_norm_launch,
            initial_norm_launch=schedule.initial_norm_launch,
        ),
    )


def _as_outer_streamed_signature(
    signature: dict[str, Any],
    config: ConfigSpec,
) -> dict[str, Any]:
    value = dict(signature)
    value["config_id"] = config.config_id
    value["runtime_backend"] = RuntimeBackend.STREAMED.value
    return value


def _measure_streamed(
    shape: TransformerShape,
    config: ConfigSpec,
    variant: RunVariant,
    protocol: MeasurementProtocol,
    device: torch.device,
) -> BenchmarkResult:
    microbatch = config.schedule.microbatch_size
    if microbatch is None or shape.batch_size % microbatch:
        raise ValueError("streamed microbatch_size must divide the logical batch")
    inner = _inner_streamed_config(config)

    # Shape 14 cannot materialize the official SxS baseline. The same weights
    # are compared against the exact reference-order streaming program at B=1.
    reference_config = portable_config()
    baseline_weights = official.BaselineTransformer(_official_config(shape, batch_size=1))
    reference = UserOptimizedTransformer(_official_config(shape, batch_size=1))
    candidate_b1 = UserOptimizedTransformer(_official_config(shape, batch_size=1))
    copy_model_weights(baseline_weights, reference, strict=True)
    copy_model_weights(baseline_weights, candidate_b1, strict=True)
    dtype = official.resolve_dtype(variant.dtype)
    reference = reference.to(device=device, dtype=dtype).eval()
    candidate_b1 = candidate_b1.to(device=device, dtype=dtype).eval()
    reference.configure_execution(config=reference_config)
    candidate_b1.configure_execution(config=inner)
    model_config_b1 = _official_config(shape, batch_size=1)
    ratio = _correctness(
        reference,
        candidate_b1,
        model_config_b1,
        variant,
        protocol,
        device,
    )
    del baseline_weights, reference, candidate_b1

    _, model = _build_models(
        shape,
        variant,
        device,
        inner,
        batch_size=microbatch,
        include_baseline=False,
    )
    x, mask = official.generate_random_case(
        _official_config(shape, batch_size=microbatch),
        device,
        dtype,
        protocol.seed + 100000,
        variant.padding_ratio,
        variant.input_scale,
    )
    expected, actual = _execution_signatures(model, x, mask)
    chunks = shape.batch_size // microbatch

    class LogicalBatch(nn.Module):
        def forward(self, value: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
            output = value
            for _ in range(chunks):
                output = model(value, valid_mask)
            return output

    if protocol.full_logical_batch:
        optimized_samples = _timings(LogicalBatch().eval(), x, mask, protocol, device)
    else:
        optimized_samples = [
            sample * chunks
            for sample in _timings(model, x, mask, protocol, device)
        ]
    peak = _peak_memory(model, x, mask, device)
    return BenchmarkResult(
        case_id=shape.case_id,
        config=config,
        passed=ratio <= 1.0,
        max_tolerance_ratio=ratio,
        optimized=TimingStats.from_samples(optimized_samples),
        peak_memory_bytes=peak,
        expected_execution_signature=_as_outer_streamed_signature(expected, config),
        actual_execution_signature=_as_outer_streamed_signature(actual, config),
    )


def measure_config(
    shape: TransformerShape,
    config: ConfigSpec,
    variant: RunVariant,
    protocol: MeasurementProtocol,
    device: str | torch.device,
    *,
    include_baseline: bool = False,
) -> BenchmarkResult:
    """Measure one explicit program."""

    resolved_device = official.resolve_device(str(device))
    torch.manual_seed(protocol.seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(protocol.seed)
    if shape.streamed:
        return _measure_streamed(shape, config, variant, protocol, resolved_device)
    if config.schedule.runtime is RuntimeBackend.STREAMED:
        raise ValueError("streamed runtime is only valid for Shape 14")

    baseline, solution = _build_models(
        shape,
        variant,
        resolved_device,
        config,
        include_baseline=True,
    )
    assert baseline is not None
    model_config = _official_config(shape)
    ratio = _correctness(
        baseline,
        solution,
        model_config,
        variant,
        protocol,
        resolved_device,
    )
    dtype = official.resolve_dtype(variant.dtype)
    x, mask = official.generate_random_case(
        model_config,
        resolved_device,
        dtype,
        protocol.seed + 100000,
        variant.padding_ratio,
        variant.input_scale,
    )
    expected, actual = _execution_signatures(solution, x, mask)
    optimized_samples = _timings(solution, x, mask, protocol, resolved_device)
    baseline_stats = None
    if include_baseline:
        baseline_samples = _timings(
            baseline,
            x,
            mask,
            protocol,
            resolved_device,
        )
        baseline_stats = TimingStats.from_samples(baseline_samples)
    peak = _peak_memory(solution, x, mask, resolved_device)
    return BenchmarkResult(
        case_id=shape.case_id,
        config=config,
        passed=ratio <= 1.0,
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
    with torch.profiler.profile(activities=activities) as profiler:
        with torch.inference_mode():
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


__all__ = ["BenchmarkResult", "TimingStats", "measure_config", "profile_config"]
