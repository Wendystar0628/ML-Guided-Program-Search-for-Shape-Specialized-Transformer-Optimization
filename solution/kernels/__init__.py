"""GPU kernels used by the submitted Transformer solution."""

from .attention_online import (
    TRITON_ONLINE_ATTENTION_AVAILABLE,
    can_use_triton_online_attention,
    triton_online_attention,
)
from .attention_preprocess import (
    TRITON_ATTENTION_PREPROCESS_AVAILABLE,
    can_use_triton_attention_preprocess,
    triton_scale_mask_to_fp32,
)
from .attention_pv import (
    TRITON_ATTENTION_PV_AVAILABLE,
    can_use_triton_fp32_probability_value,
    triton_fp32_probability_value,
)
from .attention_softmax import (
    TRITON_ATTENTION_SOFTMAX_AVAILABLE,
    can_use_s512_native_half_softmax,
    can_use_triton_attention_softmax,
    s512_scale_mask_native_half_softmax,
    triton_scale_mask_softmax,
)
from .qkv_layout import (
    TRITON_QKV_LAYOUT_AVAILABLE,
    can_use_triton_qkv_layout,
    triton_qkv_to_bhsd,
)
from .residual import (
    TRITON_RESIDUAL_AVAILABLE,
    can_use_triton_residual,
    triton_residual_add_padding,
)
from .wide_ffn import (
    WIDE_FFN_EXACT_GELU_AVAILABLE,
    can_use_wide_exact_gelu,
    wide_linear_exact_gelu,
)

__all__ = [
    "TRITON_ATTENTION_PREPROCESS_AVAILABLE",
    "TRITON_ATTENTION_PV_AVAILABLE",
    "TRITON_ATTENTION_SOFTMAX_AVAILABLE",
    "TRITON_ONLINE_ATTENTION_AVAILABLE",
    "TRITON_QKV_LAYOUT_AVAILABLE",
    "TRITON_RESIDUAL_AVAILABLE",
    "WIDE_FFN_EXACT_GELU_AVAILABLE",
    "can_use_s512_native_half_softmax",
    "can_use_triton_attention_preprocess",
    "can_use_triton_attention_softmax",
    "can_use_triton_fp32_probability_value",
    "can_use_triton_online_attention",
    "can_use_triton_qkv_layout",
    "can_use_triton_residual",
    "can_use_wide_exact_gelu",
    "s512_scale_mask_native_half_softmax",
    "triton_fp32_probability_value",
    "triton_online_attention",
    "triton_qkv_to_bhsd",
    "triton_residual_add_padding",
    "triton_scale_mask_softmax",
    "triton_scale_mask_to_fp32",
    "wide_linear_exact_gelu",
]
