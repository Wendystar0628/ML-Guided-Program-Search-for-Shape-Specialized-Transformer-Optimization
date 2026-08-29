"""Handwritten Triton attention kernels."""

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
from .triton_streaming_dh64 import (
    TRITON_STREAMING_DH64_CAUSAL_ATTENTION_BSD_BACKEND,
    can_use_triton_streaming_dh64_causal_attention,
    prevalidated_triton_streaming_dh64_causal_attention_bsd,
    triton_streaming_dh64_causal_attention_available,
    triton_streaming_dh64_causal_attention_bsd,
)

__all__ = [
    "TRITON_DH8_CAUSAL_ATTENTION_BSD_BACKEND",
    "TRITON_SHAPE13_CAUSAL_ATTENTION_BACKEND",
    "TRITON_SHAPE13_CAUSAL_ATTENTION_BSD_BACKEND",
    "TRITON_STREAMING_DH64_CAUSAL_ATTENTION_BSD_BACKEND",
    "can_use_triton_dh8_causal_attention",
    "can_use_triton_shape13_causal_attention",
    "can_use_triton_streaming_dh64_causal_attention",
    "prevalidated_triton_dh8_causal_attention_bsd",
    "prevalidated_triton_shape13_causal_attention",
    "prevalidated_triton_shape13_causal_attention_bsd",
    "prevalidated_triton_streaming_dh64_causal_attention_bsd",
    "triton_dh8_causal_attention_available",
    "triton_shape13_causal_attention",
    "triton_shape13_causal_attention_available",
    "triton_shape13_causal_attention_bsd",
    "triton_streaming_dh64_causal_attention_available",
    "triton_streaming_dh64_causal_attention_bsd",
]
