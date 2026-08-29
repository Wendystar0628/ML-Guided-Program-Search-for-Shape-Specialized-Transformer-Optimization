"""Feed-forward-network operator primitives."""

from .triton_exact_gelu import (
    TRITON_EXACT_GELU_BACKEND,
    can_use_triton_exact_gelu,
    prevalidated_triton_exact_gelu,
    triton_exact_gelu,
    triton_exact_gelu_available,
)

__all__ = [
    "TRITON_EXACT_GELU_BACKEND",
    "can_use_triton_exact_gelu",
    "prevalidated_triton_exact_gelu",
    "triton_exact_gelu",
    "triton_exact_gelu_available",
]
