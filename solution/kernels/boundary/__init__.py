"""Cross-operator kernels that remove materialized Transformer boundaries."""

from .triton_ffn_residual_norm import (
    TRITON_FFN_RESIDUAL_NORM_BACKEND,
    can_use_triton_ffn_residual_norm,
    triton_ffn_residual_norm,
    triton_ffn_residual_norm_available,
)
from .triton_linear_residual_norm import (
    TRITON_LINEAR_RESIDUAL_NORM_BACKEND,
    can_use_triton_linear_residual_norm,
    triton_linear_residual_norm,
    triton_linear_residual_norm_available,
)

__all__ = [
    "TRITON_FFN_RESIDUAL_NORM_BACKEND",
    "TRITON_LINEAR_RESIDUAL_NORM_BACKEND",
    "can_use_triton_ffn_residual_norm",
    "can_use_triton_linear_residual_norm",
    "triton_ffn_residual_norm",
    "triton_ffn_residual_norm_available",
    "triton_linear_residual_norm",
    "triton_linear_residual_norm_available",
]
