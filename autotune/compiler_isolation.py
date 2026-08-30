"""Bound compiler state to one compile-heavy search measurement."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch._inductor import config as inductor_config

from solution.config import (
    ConfigSpec,
    FFNBackend,
    ResidualNormBackend,
    RuntimeBackend,
)


def uses_torch_compile(config: ConfigSpec) -> bool:
    """Return whether measuring this configuration invokes ``torch.compile``."""

    return (
        config.schedule.runtime is RuntimeBackend.COMPILED_FORWARD
        or config.program.ffn is FFNBackend.COMPILED
        or config.program.residual_norm is ResidualNormBackend.COMPILED
    )


@contextmanager
def isolate_compiler_state(*configs: ConfigSpec) -> Iterator[None]:
    """Reset compiler state before and after a compile-heavy measurement."""

    if not any(uses_torch_compile(config) for config in configs):
        yield
        return

    torch.compiler.reset()
    try:
        # This removes only launch configurations whose shared-memory demand
        # exceeds the current device limit, so the best feasible kernel is
        # unchanged while avoidable Triton compilation work disappears.
        with inductor_config.patch(
            max_autotune_prune_choices_based_on_shared_mem=True,
        ):
            yield
    finally:
        torch.compiler.reset()


__all__ = ["isolate_compiler_state", "uses_torch_compile"]
