"""Reject officially defined shapes that are unsafe for local execution."""

from __future__ import annotations

from runner.contracts import ContractError, TransformerShape


class ResourceGuardError(ContractError):
    """Raised before model or tensor allocation for an excluded shape."""


def ensure_local_benchmark_allowed(shape: TransformerShape) -> None:
    """Fail before allocation when the official shape is locally excluded."""

    if shape.case_id == "official_14":
        raise ResourceGuardError(
            "official_14 is defined by the official workload but is excluded from "
            "this local benchmark run to prevent an unsafe allocation"
        )


def local_benchmark_shapes(
    shapes: tuple[TransformerShape, ...],
) -> tuple[TransformerShape, ...]:
    """Return the ordered shapes that are safe for the default local sweep."""

    return tuple(shape for shape in shapes if shape.case_id != "official_14")


__all__ = [
    "ResourceGuardError",
    "ensure_local_benchmark_allowed",
    "local_benchmark_shapes",
]
