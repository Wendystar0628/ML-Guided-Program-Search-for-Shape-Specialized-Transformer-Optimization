"""Strict mixed-precision attention for measured long and short-sequence families."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

MIXED_FP16_EFFICIENT_BACKEND = "mixed_fp16_efficient"
_MIN_SEQUENCE_LENGTH = 1024
_SUPPORTED_HEAD_DIM = 32
_SHORT_SEQUENCE_LENGTH = 128
_SHORT_BATCH_SIZES = frozenset({64, 128})
_SHORT_MODEL_DIMS = frozenset({32, 128})
_MIN_COMPUTE_CAPABILITY = (8, 0)


def can_use_mixed_fp16_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_token_mask: torch.Tensor | None = None,
    *,
    causal: bool = True,
    training: bool = False,
) -> bool:
    """Return whether the measured long-sequence mixed path is eligible.

    The guard describes only two CUDA-measured families: long requests with
    head dimension 32, and B64/B128 short requests with model width 32 or 128.
    Unsupported requests stay on the regular FP32 path instead of silently
    changing precision.
    """

    if training or torch.is_grad_enabled() or valid_token_mask is not None:
        return False
    if not causal or query.device.type != "cuda":
        return False
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False
    if query.shape != key.shape or query.shape != value.shape:
        return False
    if query.dtype != torch.float32:
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if query.device != key.device or query.device != value.device:
        return False
    if query.requires_grad or key.requires_grad or value.requires_grad:
        return False
    sequence_length = query.shape[-2]
    head_dim = query.shape[-1]
    long_family = (
        sequence_length >= _MIN_SEQUENCE_LENGTH and head_dim == _SUPPORTED_HEAD_DIM
    )
    short_family = (
        query.shape[0] in _SHORT_BATCH_SIZES
        and sequence_length == _SHORT_SEQUENCE_LENGTH
        and query.shape[1] * head_dim in _SHORT_MODEL_DIMS
    )
    if not (long_family or short_family):
        return False
    if not all(tensor.stride(-1) == 1 for tensor in (query, key, value)):
        return False
    if not torch.backends.cuda.mem_efficient_sdp_enabled():
        return False
    return torch.cuda.get_device_capability(query.device) >= _MIN_COMPUTE_CAPABILITY


def mixed_fp16_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_token_mask: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    causal: bool = True,
    training: bool = False,
) -> tuple[torch.Tensor, str]:
    """Run forced Efficient SDPA on temporary FP16 Q/K/V tensors.

    The returned tensor is converted back to FP32. Forcing the backend makes a
    successful backend marker truthful; an unavailable kernel is reported as
    an error and never falls through to a different implementation.
    """

    if not can_use_mixed_fp16_efficient_attention(
        query,
        key,
        value,
        valid_token_mask,
        causal=causal,
        training=training,
    ):
        raise ValueError(
            "attention tensors are incompatible with mixed FP16 Efficient SDPA"
        )

    try:
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            output = F.scaled_dot_product_attention(
                query.to(torch.float16),
                key.to(torch.float16),
                value.to(torch.float16),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=scale,
            )
    except RuntimeError as exc:
        raise RuntimeError(
            "forced FP16 Efficient SDPA is unavailable for this CUDA request"
        ) from exc
    return output.to(torch.float32), MIXED_FP16_EFFICIENT_BACKEND


__all__ = [
    "MIXED_FP16_EFFICIENT_BACKEND",
    "can_use_mixed_fp16_efficient_attention",
    "mixed_fp16_efficient_attention",
]
