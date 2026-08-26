"""GPU kernels used by the submitted Transformer solution."""

from .attention_softmax import (
    TRITON_ATTENTION_SOFTMAX_AVAILABLE,
    can_use_triton_attention_softmax,
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

__all__ = [
    "TRITON_ATTENTION_SOFTMAX_AVAILABLE",
    "TRITON_QKV_LAYOUT_AVAILABLE",
    "TRITON_RESIDUAL_AVAILABLE",
    "can_use_triton_attention_softmax",
    "can_use_triton_qkv_layout",
    "can_use_triton_residual",
    "triton_qkv_to_bhsd",
    "triton_residual_add_padding",
    "triton_scale_mask_softmax",
]
