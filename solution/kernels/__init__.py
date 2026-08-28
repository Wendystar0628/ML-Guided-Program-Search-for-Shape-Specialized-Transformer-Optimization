"""Shape-independent primitives used by the submitted Transformer."""

from .causal_attention import (
    can_use_causal_sdpa,
    causal_sdpa,
    reference_causal_attention,
)
from .mixed_attention import (
    MIXED_FP16_CUDNN_BACKEND,
    MIXED_FP16_EFFICIENT_BACKEND,
    can_use_mixed_fp16_cudnn_attention,
    can_use_mixed_fp16_efficient_attention,
    mixed_fp16_cudnn_attention,
    mixed_fp16_efficient_attention,
    prevalidated_mixed_fp16_efficient_attention,
)
from .norm_residual import can_use_residual_add, residual_add
from .qkv import can_split_qkv, split_qkv
from .residual_norm import can_use_residual_layer_norm, residual_layer_norm
from .triton_residual_norm import (
    TRITON_RESIDUAL_LAYER_NORM_BACKEND,
    can_use_triton_residual_layer_norm,
    triton_residual_layer_norm,
    triton_residual_layer_norm_available,
)

__all__ = [
    "MIXED_FP16_CUDNN_BACKEND",
    "MIXED_FP16_EFFICIENT_BACKEND",
    "TRITON_RESIDUAL_LAYER_NORM_BACKEND",
    "can_split_qkv",
    "can_use_causal_sdpa",
    "can_use_mixed_fp16_cudnn_attention",
    "can_use_mixed_fp16_efficient_attention",
    "can_use_residual_add",
    "can_use_residual_layer_norm",
    "can_use_triton_residual_layer_norm",
    "causal_sdpa",
    "mixed_fp16_cudnn_attention",
    "mixed_fp16_efficient_attention",
    "prevalidated_mixed_fp16_efficient_attention",
    "reference_causal_attention",
    "residual_add",
    "residual_layer_norm",
    "split_qkv",
    "triton_residual_layer_norm",
    "triton_residual_layer_norm_available",
]
