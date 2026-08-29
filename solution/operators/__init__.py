"""Compositions built from mature PyTorch operators."""

from .attention import (
    MIXED_FP16_CUDNN_BACKEND,
    MIXED_FP16_EFFICIENT_BACKEND,
    can_use_causal_sdpa,
    can_use_mixed_fp16_cudnn_attention,
    can_use_mixed_fp16_efficient_attention,
    causal_sdpa,
    mixed_fp16_cudnn_attention,
    mixed_fp16_efficient_attention,
    prevalidated_mixed_fp16_efficient_attention,
    reference_causal_attention,
)
from .norm import (
    can_use_residual_add,
    can_use_residual_layer_norm,
    residual_add,
    residual_layer_norm,
)
from .projection import can_split_qkv, split_qkv

__all__ = [
    "MIXED_FP16_CUDNN_BACKEND",
    "MIXED_FP16_EFFICIENT_BACKEND",
    "can_split_qkv",
    "can_use_causal_sdpa",
    "can_use_mixed_fp16_cudnn_attention",
    "can_use_mixed_fp16_efficient_attention",
    "can_use_residual_add",
    "can_use_residual_layer_norm",
    "causal_sdpa",
    "mixed_fp16_cudnn_attention",
    "mixed_fp16_efficient_attention",
    "prevalidated_mixed_fp16_efficient_attention",
    "reference_causal_attention",
    "residual_add",
    "residual_layer_norm",
    "split_qkv",
]
