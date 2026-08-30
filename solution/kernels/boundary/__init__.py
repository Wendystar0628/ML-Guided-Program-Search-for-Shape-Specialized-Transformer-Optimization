"""Cross-operator kernels that remove materialized Transformer boundaries."""

from .triton_linear_residual_norm import (
    TRITON_LINEAR_RESIDUAL_NORM_BACKEND,
    can_use_triton_linear_residual_norm,
    triton_linear_residual_norm,
    triton_linear_residual_norm_available,
)

__all__ = [
    "TRITON_LINEAR_RESIDUAL_NORM_BACKEND",
    "can_use_triton_linear_residual_norm",
    "triton_linear_residual_norm",
    "triton_linear_residual_norm_available",
]
