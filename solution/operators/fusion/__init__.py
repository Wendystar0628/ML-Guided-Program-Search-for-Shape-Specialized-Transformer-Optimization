"""Cross-operator fusion primitives."""

from .triton_d32_fusion import (
    TRITON_D32_RESIDUAL_LAYER_NORM_BACKEND,
    can_use_triton_d32_residual_layer_norm,
    prevalidated_triton_d32_residual_layer_norm,
    triton_d32_residual_layer_norm,
    triton_d32_residual_layer_norm_available,
)

__all__ = [
    "TRITON_D32_RESIDUAL_LAYER_NORM_BACKEND",
    "can_use_triton_d32_residual_layer_norm",
    "prevalidated_triton_d32_residual_layer_norm",
    "triton_d32_residual_layer_norm",
    "triton_d32_residual_layer_norm_available",
]
