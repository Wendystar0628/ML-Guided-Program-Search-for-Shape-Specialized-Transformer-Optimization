"""Normalization and residual operator implementations."""

from .compiled_residual import can_use_residual_layer_norm, residual_layer_norm
from .residual_add import can_use_residual_add, residual_add
from .triton_initial import (
    TRITON_INITIAL_FP16_LAYER_NORM_BACKEND,
    can_use_triton_initial_fp16_layer_norm,
    triton_initial_fp16_layer_norm,
    triton_initial_fp16_layer_norm_available,
)
from .triton_mixed_residual import (
    TRITON_MIXED_RESIDUAL_LAYER_NORM_BACKEND,
    can_use_triton_mixed_residual_layer_norm,
    triton_mixed_residual_layer_norm,
    triton_mixed_residual_layer_norm_available,
)
from .triton_residual import (
    TRITON_RESIDUAL_LAYER_NORM_BACKEND,
    can_use_triton_residual_layer_norm,
    triton_residual_layer_norm,
    triton_residual_layer_norm_available,
)

__all__ = [
    "TRITON_INITIAL_FP16_LAYER_NORM_BACKEND",
    "TRITON_MIXED_RESIDUAL_LAYER_NORM_BACKEND",
    "TRITON_RESIDUAL_LAYER_NORM_BACKEND",
    "can_use_residual_add",
    "can_use_residual_layer_norm",
    "can_use_triton_initial_fp16_layer_norm",
    "can_use_triton_mixed_residual_layer_norm",
    "can_use_triton_residual_layer_norm",
    "residual_add",
    "residual_layer_norm",
    "triton_initial_fp16_layer_norm",
    "triton_initial_fp16_layer_norm_available",
    "triton_mixed_residual_layer_norm",
    "triton_mixed_residual_layer_norm_available",
    "triton_residual_layer_norm",
    "triton_residual_layer_norm_available",
]
