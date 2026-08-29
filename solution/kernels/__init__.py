"""Execution primitives used by the submitted Transformer."""

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
from .triton_dh8_attention import (
    TRITON_DH8_CAUSAL_ATTENTION_BSD_BACKEND,
    can_use_triton_dh8_causal_attention,
    prevalidated_triton_dh8_causal_attention_bsd,
    triton_dh8_causal_attention_available,
)
from .triton_initial_norm import (
    TRITON_INITIAL_FP16_LAYER_NORM_BACKEND,
    can_use_triton_initial_fp16_layer_norm,
    triton_initial_fp16_layer_norm,
    triton_initial_fp16_layer_norm_available,
)
from .triton_mixed_residual_norm import (
    TRITON_MIXED_RESIDUAL_LAYER_NORM_BACKEND,
    can_use_triton_mixed_residual_layer_norm,
    triton_mixed_residual_layer_norm,
    triton_mixed_residual_layer_norm_available,
)
from .triton_residual_norm import (
    TRITON_RESIDUAL_LAYER_NORM_BACKEND,
    can_use_triton_residual_layer_norm,
    triton_residual_layer_norm,
    triton_residual_layer_norm_available,
)
from .triton_shape13_attention import (
    TRITON_SHAPE13_CAUSAL_ATTENTION_BACKEND,
    can_use_triton_shape13_causal_attention,
    prevalidated_triton_shape13_causal_attention,
    triton_shape13_causal_attention,
    triton_shape13_causal_attention_available,
)

__all__ = [
    "MIXED_FP16_CUDNN_BACKEND",
    "MIXED_FP16_EFFICIENT_BACKEND",
    "TRITON_DH8_CAUSAL_ATTENTION_BSD_BACKEND",
    "TRITON_INITIAL_FP16_LAYER_NORM_BACKEND",
    "TRITON_MIXED_RESIDUAL_LAYER_NORM_BACKEND",
    "TRITON_RESIDUAL_LAYER_NORM_BACKEND",
    "TRITON_SHAPE13_CAUSAL_ATTENTION_BACKEND",
    "can_split_qkv",
    "can_use_causal_sdpa",
    "can_use_mixed_fp16_cudnn_attention",
    "can_use_mixed_fp16_efficient_attention",
    "can_use_residual_add",
    "can_use_residual_layer_norm",
    "can_use_triton_dh8_causal_attention",
    "can_use_triton_initial_fp16_layer_norm",
    "can_use_triton_mixed_residual_layer_norm",
    "can_use_triton_residual_layer_norm",
    "can_use_triton_shape13_causal_attention",
    "causal_sdpa",
    "mixed_fp16_cudnn_attention",
    "mixed_fp16_efficient_attention",
    "prevalidated_mixed_fp16_efficient_attention",
    "prevalidated_triton_dh8_causal_attention_bsd",
    "prevalidated_triton_shape13_causal_attention",
    "reference_causal_attention",
    "residual_add",
    "residual_layer_norm",
    "split_qkv",
    "triton_dh8_causal_attention_available",
    "triton_initial_fp16_layer_norm",
    "triton_initial_fp16_layer_norm_available",
    "triton_mixed_residual_layer_norm",
    "triton_mixed_residual_layer_norm_available",
    "triton_residual_layer_norm",
    "triton_residual_layer_norm_available",
    "triton_shape13_causal_attention",
    "triton_shape13_causal_attention_available",
]
