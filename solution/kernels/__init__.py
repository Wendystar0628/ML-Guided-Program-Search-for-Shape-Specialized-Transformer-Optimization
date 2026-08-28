"""Shape-independent primitives used by the submitted Transformer."""

from .causal_attention import (
    can_use_causal_sdpa,
    causal_attention,
    causal_sdpa,
    reference_causal_attention,
)
from .ffn import can_use_linear_exact_gelu, linear_exact_gelu
from .norm_residual import can_use_residual_add, residual_add
from .qkv import can_split_qkv, split_qkv

__all__ = [
    "can_split_qkv",
    "can_use_causal_sdpa",
    "can_use_linear_exact_gelu",
    "can_use_residual_add",
    "causal_attention",
    "causal_sdpa",
    "linear_exact_gelu",
    "reference_causal_attention",
    "residual_add",
    "split_qkv",
]
