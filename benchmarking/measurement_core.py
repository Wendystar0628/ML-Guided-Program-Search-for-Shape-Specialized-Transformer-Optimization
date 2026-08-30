"""Shared result types and primitives for resident and streamed measurement."""

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
from solution.config import ConfigSpec
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
    estimated_model_flops: int | None = None
    latency_kind: str | None = None
    output_digest: str | None = None

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

        value = {
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
        if self.estimated_model_flops is not None:
            value["estimated_model_flops"] = self.estimated_model_flops
        if self.latency_kind is not None:
            value["latency_kind"] = self.latency_kind
        if self.output_digest is not None:
            value["output_digest"] = self.output_digest
        return value


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
