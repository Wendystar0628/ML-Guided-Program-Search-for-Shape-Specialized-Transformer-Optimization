"""GPU kernels used by the submitted Transformer solution."""

from .attention_online import (
    can_use_triton_online_attention,
    supports_triton_online_attention,
    triton_online_attention,
)
from .attention_preprocess import (
    can_use_triton_attention_preprocess,
    supports_triton_attention_preprocess,
    triton_scale_mask_to_fp32,
)
from .attention_softmax import (
    can_use_s512_native_half_softmax,
    can_use_triton_attention_softmax,
    s512_scale_mask_native_half_softmax,
    supports_s512_native_half_softmax,
    supports_triton_attention_softmax,
    triton_scale_mask_softmax,
)
from .qkv_layout import (
    can_use_triton_qkv_layout,
    supports_triton_qkv_layout,
    triton_qkv_to_bhsd,
)
from .residual import (
    can_use_triton_residual,
    supports_triton_residual,
    triton_residual_add_padding,
)
from .wide_ffn import (
    can_use_wide_exact_gelu,
    supports_wide_exact_gelu,
    wide_linear_exact_gelu,
)

__all__ = [
    "can_use_s512_native_half_softmax",
    "can_use_triton_attention_preprocess",
    "can_use_triton_attention_softmax",
    "can_use_triton_online_attention",
    "can_use_triton_qkv_layout",
    "can_use_triton_residual",
    "can_use_wide_exact_gelu",
    "s512_scale_mask_native_half_softmax",
    "supports_s512_native_half_softmax",
    "supports_triton_attention_preprocess",
    "supports_triton_attention_softmax",
    "supports_triton_online_attention",
    "supports_triton_qkv_layout",
    "supports_triton_residual",
    "supports_wide_exact_gelu",
    "triton_online_attention",
    "triton_qkv_to_bhsd",
    "triton_residual_add_padding",
    "triton_scale_mask_softmax",
    "triton_scale_mask_to_fp32",
    "wide_linear_exact_gelu",
]
