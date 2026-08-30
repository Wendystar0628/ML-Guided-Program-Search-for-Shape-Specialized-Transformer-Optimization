"""Direct official-shape measurement used by both the CLI and TPE search."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from benchmarking.protocols import MeasurementProtocol, RunVariant, TransformerShape
from official import torch_transformer_benchmark as official
from solution.config import ConfigSpec, RuntimeBackend, ScheduleConfig, portable_config
from solution.transformer import UserOptimizedTransformer, copy_model_weights


@dataclass(frozen=True, slots=True)
class TimingStats:
    median_ms: float
    p90_ms: float

    @classmethod
    def from_samples(cls, samples: list[float]) -> TimingStats:
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


@dataclass(frozen=True, slots=True)
class PairedBenchmarkResult:
    """Two configs measured in alternating Formal rounds on one workload."""

    incumbent: BenchmarkResult
    challenger: BenchmarkResult
    paired_ratios: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.incumbent.case_id != self.challenger.case_id:
            raise ValueError("paired results must use the same workload")
        ratios = tuple(float(value) for value in self.paired_ratios)
        if not ratios or any(
            not math.isfinite(value) or value <= 0.0 for value in ratios
        ):
            raise ValueError("paired_ratios must be finite and positive")
        object.__setattr__(self, "paired_ratios", ratios)

    @property
    def median_speedup(self) -> float:
        """Return the median of same-round incumbent/challenger ratios."""

        return float(statistics.median(self.paired_ratios))


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

    return _comparison_metrics(
        reference,
        candidate,
        rtol=rtol,
        atol=atol,
        chunk_size=chunk_size,
    )[1]


def _comparison_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    chunk_size: int = 4096,
) -> tuple[bool, float]:
    """Return the exact official pass result and a continuous error ratio."""

    if reference.shape != candidate.shape:
        return False, math.inf
    flat_reference = reference.detach().reshape(-1)
    flat_candidate = candidate.detach().reshape(-1)
    passed = True
    maximum = 0.0
    for start in range(0, flat_reference.numel(), chunk_size):
        ref = flat_reference[start : start + chunk_size].float()
        value = flat_candidate[start : start + chunk_size].float()
        finite = torch.isfinite(ref) & torch.isfinite(value)
        if not bool(finite.all().item()):
            return False, math.inf
        error = (value - ref).abs()
        absolute_allowance = torch.full_like(error, float(atol))
        relative_allowance = float(rtol) * ref.abs()
        passed = passed and bool(
            ((error <= absolute_allowance) | (error <= relative_allowance)).all().item()
        )

        infinite = torch.full_like(error, math.inf)
        zero_error_ratio = torch.where(error == 0, torch.zeros_like(error), infinite)
        absolute_ratio = torch.where(
            absolute_allowance > 0,
            error / absolute_allowance,
            zero_error_ratio,
        )
        relative_ratio = torch.where(
            relative_allowance > 0,
            error / relative_allowance,
            zero_error_ratio,
        )
        local = torch.minimum(absolute_ratio, relative_ratio).max().item()
        maximum = max(maximum, float(local))
    return passed, maximum


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
        raise RuntimeError(  # noqa: TRY004 - runtime observation contract failed
            "solution did not report its execution path"
        )
    return dict(expected), dict(actual)


def _correctness(
    reference: nn.Module,
    candidate: nn.Module,
    model_config: official.TransformerConfig,
    variant: RunVariant,
    protocol: MeasurementProtocol,
    device: torch.device,
) -> tuple[bool, float]:
    dtype = official.resolve_dtype(variant.dtype)
    passed = True
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
            trial_passed, trial_ratio = _comparison_metrics(
                reference_output,
                candidate_output,
                rtol=protocol.rtol,
                atol=protocol.atol,
            )
            passed = passed and trial_passed
            maximum = max(maximum, trial_ratio)
    return passed, maximum


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


def _interleaved_timings(
    incumbent: nn.Module,
    incumbent_input: tuple[torch.Tensor, torch.Tensor],
    challenger: nn.Module,
    challenger_input: tuple[torch.Tensor, torch.Tensor],
    protocol: MeasurementProtocol,
    device: torch.device,
    *,
    incumbent_scale: int = 1,
    challenger_scale: int = 1,
    stop_when: Callable[[tuple[float, ...]], bool] | None = None,
) -> tuple[list[float], list[float], tuple[float, ...]]:
    """Measure AB/BA rounds and retain one latency ratio per paired round."""

    if incumbent_scale <= 0 or challenger_scale <= 0:
        raise ValueError("timing scales must be positive")
    incumbent_x, incumbent_mask = incumbent_input
    challenger_x, challenger_mask = challenger_input
    official.warmup_model(
        incumbent,
        incumbent_x,
        incumbent_mask,
        protocol.warmup,
        device,
    )
    official.warmup_model(
        challenger,
        challenger_x,
        challenger_mask,
        protocol.warmup,
        device,
    )

    incumbent_samples: list[float] = []
    challenger_samples: list[float] = []
    paired_ratios: list[float] = []

    def one_round(
        model: nn.Module,
        value: torch.Tensor,
        valid_mask: torch.Tensor,
        scale: int,
    ) -> list[float]:
        return [
            float(sample) * scale
            for sample in official.benchmark_once(
                model,
                value,
                valid_mask,
                protocol.repeats,
                device,
            )
        ]

    for round_index in range(protocol.rounds):
        if round_index % 2 == 0:
            incumbent_round = one_round(
                incumbent,
                incumbent_x,
                incumbent_mask,
                incumbent_scale,
            )
            challenger_round = one_round(
                challenger,
                challenger_x,
                challenger_mask,
                challenger_scale,
            )
        else:
            challenger_round = one_round(
                challenger,
                challenger_x,
                challenger_mask,
                challenger_scale,
            )
            incumbent_round = one_round(
                incumbent,
                incumbent_x,
                incumbent_mask,
                incumbent_scale,
            )
        incumbent_median = float(statistics.median(incumbent_round))
        challenger_median = float(statistics.median(challenger_round))
        if incumbent_median <= 0.0 or challenger_median <= 0.0:
            raise RuntimeError("benchmark produced a non-positive paired latency")
        incumbent_samples.extend(incumbent_round)
        challenger_samples.extend(challenger_round)
        paired_ratios.append(incumbent_median / challenger_median)
        if stop_when is not None and stop_when(tuple(paired_ratios)):
            break
    return incumbent_samples, challenger_samples, tuple(paired_ratios)


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
            qkv_launch=schedule.qkv_launch,
            attention_output_projection_launch=(
                schedule.attention_output_projection_launch
            ),
            residual_norm_launch=schedule.residual_norm_launch,
            initial_norm_launch=schedule.initial_norm_launch,
            ffn_launch=schedule.ffn_launch,
            ffn_input_launch=schedule.ffn_input_launch,
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
    baseline_weights = official.BaselineTransformer(
        _official_config(shape, batch_size=1)
    )
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
    passed, ratio = _correctness(
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
        def forward(
            self, value: torch.Tensor, valid_mask: torch.Tensor
        ) -> torch.Tensor:
            output = value
            for _ in range(chunks):
                output = model(value, valid_mask)
            return output

    if protocol.full_logical_batch:
        optimized_samples = _timings(LogicalBatch().eval(), x, mask, protocol, device)
    else:
        optimized_samples = [
            sample * chunks for sample in _timings(model, x, mask, protocol, device)
        ]
    peak = _peak_memory(model, x, mask, device)
    return BenchmarkResult(
        case_id=shape.case_id,
        config=config,
        passed=passed,
        max_tolerance_ratio=ratio,
        optimized=TimingStats.from_samples(optimized_samples),
        peak_memory_bytes=peak,
        expected_execution_signature=_as_outer_streamed_signature(expected, config),
        actual_execution_signature=_as_outer_streamed_signature(actual, config),
    )


def _measure_paired_resident(
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


def _measure_paired_streamed(
    shape: TransformerShape,
    challenger_config: ConfigSpec,
    incumbent_config: ConfigSpec,
    variant: RunVariant,
    protocol: MeasurementProtocol,
    device: torch.device,
    stop_when: Callable[[tuple[float, ...]], bool] | None,
) -> PairedBenchmarkResult:
    configs = (incumbent_config, challenger_config)
    microbatches = tuple(config.schedule.microbatch_size for config in configs)
    if any(
        microbatch is None or shape.batch_size % microbatch
        for microbatch in microbatches
    ):
        raise ValueError("streamed microbatch_size must divide the logical batch")
    incumbent_microbatch, challenger_microbatch = microbatches
    assert incumbent_microbatch is not None
    assert challenger_microbatch is not None
    incumbent_inner = _inner_streamed_config(incumbent_config)
    challenger_inner = _inner_streamed_config(challenger_config)

    reference_config = portable_config()
    model_config_b1 = _official_config(shape, batch_size=1)
    baseline_weights = official.BaselineTransformer(model_config_b1)
    reference = UserOptimizedTransformer(model_config_b1)
    incumbent_b1 = UserOptimizedTransformer(model_config_b1)
    challenger_b1 = UserOptimizedTransformer(model_config_b1)
    copy_model_weights(baseline_weights, reference, strict=True)
    copy_model_weights(baseline_weights, incumbent_b1, strict=True)
    copy_model_weights(baseline_weights, challenger_b1, strict=True)
    dtype = official.resolve_dtype(variant.dtype)
    reference = reference.to(device=device, dtype=dtype).eval()
    incumbent_b1 = incumbent_b1.to(device=device, dtype=dtype).eval()
    challenger_b1 = challenger_b1.to(device=device, dtype=dtype).eval()
    reference.configure_execution(config=reference_config)
    incumbent_b1.configure_execution(config=incumbent_inner)
    challenger_b1.configure_execution(config=challenger_inner)
    incumbent_passed, incumbent_ratio = _correctness(
        reference,
        incumbent_b1,
        model_config_b1,
        variant,
        protocol,
        device,
    )
    challenger_passed, challenger_ratio = _correctness(
        reference,
        challenger_b1,
        model_config_b1,
        variant,
        protocol,
        device,
    )
    del baseline_weights, reference, incumbent_b1, challenger_b1

    _, incumbent = _build_models(
        shape,
        variant,
        device,
        incumbent_inner,
        batch_size=incumbent_microbatch,
        include_baseline=False,
    )
    _, challenger = _build_models(
        shape,
        variant,
        device,
        challenger_inner,
        batch_size=challenger_microbatch,
        include_baseline=False,
    )
    incumbent_x, incumbent_mask = official.generate_random_case(
        _official_config(shape, batch_size=incumbent_microbatch),
        device,
        dtype,
        protocol.seed + 100000,
        variant.padding_ratio,
        variant.input_scale,
    )
    challenger_x, challenger_mask = official.generate_random_case(
        _official_config(shape, batch_size=challenger_microbatch),
        device,
        dtype,
        protocol.seed + 100000,
        variant.padding_ratio,
        variant.input_scale,
    )
    incumbent_expected, incumbent_actual = _execution_signatures(
        incumbent,
        incumbent_x,
        incumbent_mask,
    )
    challenger_expected, challenger_actual = _execution_signatures(
        challenger,
        challenger_x,
        challenger_mask,
    )
    incumbent_chunks = shape.batch_size // incumbent_microbatch
    challenger_chunks = shape.batch_size // challenger_microbatch

    class LogicalBatch(nn.Module):
        def __init__(self, model: nn.Module, chunks: int) -> None:
            super().__init__()
            self.model = model
            self.chunks = chunks

        def forward(
            self, value: torch.Tensor, valid_mask: torch.Tensor
        ) -> torch.Tensor:
            output = value
            for _ in range(self.chunks):
                output = self.model(value, valid_mask)
            return output

    incumbent_timed: nn.Module = incumbent
    challenger_timed: nn.Module = challenger
    incumbent_scale = incumbent_chunks
    challenger_scale = challenger_chunks
    if protocol.full_logical_batch:
        incumbent_timed = LogicalBatch(incumbent, incumbent_chunks).eval()
        challenger_timed = LogicalBatch(challenger, challenger_chunks).eval()
        incumbent_scale = 1
        challenger_scale = 1
    incumbent_samples, challenger_samples, paired_ratios = _interleaved_timings(
        incumbent_timed,
        (incumbent_x, incumbent_mask),
        challenger_timed,
        (challenger_x, challenger_mask),
        protocol,
        device,
        incumbent_scale=incumbent_scale,
        challenger_scale=challenger_scale,
        stop_when=stop_when,
    )
    incumbent_peak = _peak_memory(
        incumbent,
        incumbent_x,
        incumbent_mask,
        device,
    )
    challenger_peak = _peak_memory(
        challenger,
        challenger_x,
        challenger_mask,
        device,
    )
    return PairedBenchmarkResult(
        incumbent=BenchmarkResult(
            case_id=shape.case_id,
            config=incumbent_config,
            passed=incumbent_passed,
            max_tolerance_ratio=incumbent_ratio,
            optimized=TimingStats.from_samples(incumbent_samples),
            peak_memory_bytes=incumbent_peak,
            expected_execution_signature=_as_outer_streamed_signature(
                incumbent_expected,
                incumbent_config,
            ),
            actual_execution_signature=_as_outer_streamed_signature(
                incumbent_actual,
                incumbent_config,
            ),
        ),
        challenger=BenchmarkResult(
            case_id=shape.case_id,
            config=challenger_config,
            passed=challenger_passed,
            max_tolerance_ratio=challenger_ratio,
            optimized=TimingStats.from_samples(challenger_samples),
            peak_memory_bytes=challenger_peak,
            expected_execution_signature=_as_outer_streamed_signature(
                challenger_expected,
                challenger_config,
            ),
            actual_execution_signature=_as_outer_streamed_signature(
                challenger_actual,
                challenger_config,
            ),
        ),
        paired_ratios=paired_ratios,
    )


def measure_paired_configs(
    shape: TransformerShape,
    challenger_config: ConfigSpec,
    incumbent_config: ConfigSpec,
    variant: RunVariant,
    protocol: MeasurementProtocol,
    device: str | torch.device,
    *,
    stop_when: Callable[[tuple[float, ...]], bool] | None = None,
) -> PairedBenchmarkResult:
    """Measure challenger and incumbent in alternating AB/BA rounds."""

    resolved_device = official.resolve_device(str(device))
    torch.manual_seed(protocol.seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(protocol.seed)
    if shape.streamed:
        return _measure_paired_streamed(
            shape,
            challenger_config,
            incumbent_config,
            variant,
            protocol,
            resolved_device,
            stop_when,
        )
    if any(
        config.schedule.runtime is RuntimeBackend.STREAMED
        for config in (challenger_config, incumbent_config)
    ):
        raise ValueError("streamed runtime is only valid for Shape 14")
    return _measure_paired_resident(
        shape,
        challenger_config,
        incumbent_config,
        variant,
        protocol,
        resolved_device,
        stop_when,
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
    passed, ratio = _correctness(
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
    baseline_stats = None
    if include_baseline:
        baseline_samples, optimized_samples, _ = _interleaved_timings(
            baseline,
            (x, mask),
            solution,
            (x, mask),
            protocol,
            resolved_device,
        )
        baseline_stats = TimingStats.from_samples(baseline_samples)
    else:
        optimized_samples = _timings(solution, x, mask, protocol, resolved_device)
    peak = _peak_memory(solution, x, mask, resolved_device)
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
    "BenchmarkResult",
    "PairedBenchmarkResult",
    "TimingStats",
    "measure_config",
    "measure_paired_configs",
    "profile_config",
]
