"""Shape 14 streaming measurement and paired comparison."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from official import torch_transformer_benchmark as official
from solution.config import (
    ConfigSpec,
    RuntimeBackend,
    ScheduleConfig,
    portable_config,
)
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
from .protocols import MeasurementProtocol, RunVariant, TransformerShape


def _estimated_model_flops(shape: TransformerShape) -> int:
    """Estimate useful forward FLOPs for the full logical batch."""

    batch = shape.batch_size
    sequence = shape.seq_len
    width = shape.d_model
    attention_factor = 2 if shape.causal else 4
    attention = attention_factor * batch * sequence * sequence * width
    projections = 8 * batch * sequence * width * width
    ffn = 4 * batch * sequence * width * shape.ffn_dim
    return shape.num_layers * (attention + projections + ffn)


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


class _DistinctLogicalBatch(nn.Module):
    """Run ordered, distinct microbatches and retain only a tiny output summary."""

    def __init__(self, model: nn.Module, chunks: int) -> None:
        super().__init__()
        self.model = model
        self.chunks = chunks
        self.last_summary: torch.Tensor | None = None

    def forward(
        self,
        value: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        summary: torch.Tensor | None = None
        applied_offsets = 0
        offset = 2.0**-10
        try:
            for chunk_index in range(self.chunks):
                if chunk_index:
                    value.add_(offset)
                    applied_offsets += 1
                output = self.model(value, valid_mask)
                flat = output.reshape(-1)
                stride = max(flat.numel() // 32, 1)
                sample = flat[::stride][:32].float()
                weighted = sample * float(chunk_index + 1)
                summary = weighted if summary is None else summary + weighted
        finally:
            if applied_offsets:
                value.sub_(offset * applied_offsets)
        if summary is None:
            raise RuntimeError("logical batch must contain at least one chunk")
        self.last_summary = summary.detach()
        return summary


def _logical_output_digest(logical_batch: _DistinctLogicalBatch) -> str | None:
    summary = logical_batch.last_summary
    if summary is None:
        return None
    payload = summary.float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def measure_shape14_config(
    shape: TransformerShape,
    config: ConfigSpec,
    variant: RunVariant,
    protocol: MeasurementProtocol,
    device: torch.device,
) -> BenchmarkResult:
    """Measure one streamed Shape 14 configuration."""

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

    output_digest = None
    if protocol.full_logical_batch:
        logical_batch = _DistinctLogicalBatch(model, chunks).eval()
        optimized_samples = _timings(logical_batch, x, mask, protocol, device)
        latency_kind = "end_to_end_distinct_microbatches"
        output_digest = _logical_output_digest(logical_batch)
    else:
        optimized_samples = [
            sample * chunks for sample in _timings(model, x, mask, protocol, device)
        ]
        latency_kind = "model_compute_estimate"
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
        estimated_model_flops=_estimated_model_flops(shape),
        latency_kind=latency_kind,
        output_digest=output_digest,
    )


def measure_paired_shape14_configs(
    shape: TransformerShape,
    challenger_config: ConfigSpec,
    incumbent_config: ConfigSpec,
    variant: RunVariant,
    protocol: MeasurementProtocol,
    device: torch.device,
    stop_when: Callable[[tuple[float, ...]], bool] | None,
) -> PairedBenchmarkResult:
    """Measure two streamed Shape 14 configurations in paired rounds."""

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

    incumbent_timed: nn.Module = incumbent
    challenger_timed: nn.Module = challenger
    incumbent_scale = incumbent_chunks
    challenger_scale = challenger_chunks
    if protocol.full_logical_batch:
        incumbent_timed = _DistinctLogicalBatch(incumbent, incumbent_chunks).eval()
        challenger_timed = _DistinctLogicalBatch(
            challenger,
            challenger_chunks,
        ).eval()
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
    latency_kind = (
        "end_to_end_distinct_microbatches"
        if protocol.full_logical_batch
        else "model_compute_estimate"
    )
    incumbent_digest = (
        _logical_output_digest(incumbent_timed)
        if isinstance(incumbent_timed, _DistinctLogicalBatch)
        else None
    )
    challenger_digest = (
        _logical_output_digest(challenger_timed)
        if isinstance(challenger_timed, _DistinctLogicalBatch)
        else None
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
            estimated_model_flops=_estimated_model_flops(shape),
            latency_kind=latency_kind,
            output_digest=incumbent_digest,
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
            estimated_model_flops=_estimated_model_flops(shape),
            latency_kind=latency_kind,
            output_digest=challenger_digest,
        ),
        paired_ratios=paired_ratios,
    )


__all__ = ["measure_paired_shape14_configs", "measure_shape14_config"]
