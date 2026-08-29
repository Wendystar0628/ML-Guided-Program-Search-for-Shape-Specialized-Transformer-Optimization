"""Projection primitives that emit attention-native tensor layouts."""

from .triton_attention_output import (
    TRITON_ATTENTION_OUTPUT_PROJECTION_BACKEND,
    can_use_triton_attention_output_projection,
    prevalidated_triton_attention_output_projection,
    triton_attention_output_projection,
    triton_attention_output_projection_available,
)
from .triton_qkv_layout import (
    TRITON_QKV_NATIVE_BHSD_BACKEND,
    can_use_triton_qkv_native_bhsd,
    prevalidated_triton_qkv_native_bhsd,
    triton_qkv_native_bhsd,
    triton_qkv_native_bhsd_available,
)

__all__ = [
    "TRITON_ATTENTION_OUTPUT_PROJECTION_BACKEND",
    "TRITON_QKV_NATIVE_BHSD_BACKEND",
    "can_use_triton_attention_output_projection",
    "can_use_triton_qkv_native_bhsd",
    "prevalidated_triton_attention_output_projection",
    "prevalidated_triton_qkv_native_bhsd",
    "triton_attention_output_projection",
    "triton_attention_output_projection_available",
    "triton_qkv_native_bhsd",
    "triton_qkv_native_bhsd_available",
]
