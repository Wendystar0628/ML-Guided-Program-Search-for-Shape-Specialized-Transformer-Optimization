"""Attention operator implementations."""

from .mixed_precision import (
    MIXED_FP16_CUDNN_BACKEND,
    MIXED_FP16_EFFICIENT_BACKEND,
    can_use_mixed_fp16_cudnn_attention,
    can_use_mixed_fp16_efficient_attention,
    mixed_fp16_cudnn_attention,
    mixed_fp16_efficient_attention,
    prevalidated_mixed_fp16_efficient_attention,
)
from .sdpa import can_use_causal_sdpa, causal_sdpa, reference_causal_attention
from .triton_dh8 import (
    TRITON_DH8_CAUSAL_ATTENTION_BSD_BACKEND,
    can_use_triton_dh8_causal_attention,
    prevalidated_triton_dh8_causal_attention_bsd,
    triton_dh8_causal_attention_available,
)
from .triton_s1024_dh32 import (
    TRITON_SHAPE13_CAUSAL_ATTENTION_BACKEND,
    TRITON_SHAPE13_CAUSAL_ATTENTION_BSD_BACKEND,
    can_use_triton_shape13_causal_attention,
    prevalidated_triton_shape13_causal_attention,
    prevalidated_triton_shape13_causal_attention_bsd,
    triton_shape13_causal_attention,
    triton_shape13_causal_attention_available,
    triton_shape13_causal_attention_bsd,
)

__all__ = [
    "MIXED_FP16_CUDNN_BACKEND",
    "MIXED_FP16_EFFICIENT_BACKEND",
    "TRITON_DH8_CAUSAL_ATTENTION_BSD_BACKEND",
    "TRITON_SHAPE13_CAUSAL_ATTENTION_BACKEND",
    "TRITON_SHAPE13_CAUSAL_ATTENTION_BSD_BACKEND",
    "can_use_causal_sdpa",
    "can_use_mixed_fp16_cudnn_attention",
    "can_use_mixed_fp16_efficient_attention",
    "can_use_triton_dh8_causal_attention",
    "can_use_triton_shape13_causal_attention",
    "causal_sdpa",
    "mixed_fp16_cudnn_attention",
    "mixed_fp16_efficient_attention",
    "prevalidated_mixed_fp16_efficient_attention",
    "prevalidated_triton_dh8_causal_attention_bsd",
    "prevalidated_triton_shape13_causal_attention",
    "prevalidated_triton_shape13_causal_attention_bsd",
    "reference_causal_attention",
    "triton_dh8_causal_attention_available",
    "triton_shape13_causal_attention",
    "triton_shape13_causal_attention_available",
    "triton_shape13_causal_attention_bsd",
]
