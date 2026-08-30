"""Public dispatch for resident and streamed official-shape measurement."""

from __future__ import annotations

from collections.abc import Callable

import torch

from benchmarking.protocols import MeasurementProtocol, RunVariant, TransformerShape
from official import torch_transformer_benchmark as official
from solution.config import ConfigSpec, RuntimeBackend

from .measurement_core import BenchmarkResult, PairedBenchmarkResult, TimingStats
from .resident_measure import (
    measure_paired_resident_configs,
    measure_resident_config,
    profile_config,
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
        from .shape14_measure import measure_paired_shape14_configs

        return measure_paired_shape14_configs(
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
    return measure_paired_resident_configs(
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
        from .shape14_measure import measure_shape14_config

        return measure_shape14_config(
            shape,
            config,
            variant,
            protocol,
            resolved_device,
        )
    if config.schedule.runtime is RuntimeBackend.STREAMED:
        raise ValueError("streamed runtime is only valid for Shape 14")
    return measure_resident_config(
        shape,
        config,
        variant,
        protocol,
        resolved_device,
        include_baseline=include_baseline,
    )


__all__ = [
    "BenchmarkResult",
    "PairedBenchmarkResult",
    "TimingStats",
    "measure_config",
    "measure_paired_configs",
    "profile_config",
]
