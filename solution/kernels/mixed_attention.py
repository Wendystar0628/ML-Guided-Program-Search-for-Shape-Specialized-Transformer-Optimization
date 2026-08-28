"""Strict mixed-precision attention for measured long and short-sequence families."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..shape_families import is_mixed_fp16_core_efficient_runtime_family

MIXED_FP16_EFFICIENT_BACKEND = "mixed_fp16_efficient"
MIXED_FP16_CUDNN_BACKEND = "mixed_fp16_cudnn"
_MIN_SEQUENCE_LENGTH = 1024
_SUPPORTED_HEAD_DIMS = frozenset({32, 64})
_CUDNN_HEAD_DIM = 64
_SHORT_SEQUENCE_LENGTH = 128
_SHORT_BATCH_SIZES = frozenset({64, 128})
_SHORT_MODEL_DIMS = frozenset({32, 128})
_MIN_COMPUTE_CAPABILITY = (8, 0)


def _mixed_fp16_inputs_are_legal(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_token_mask: torch.Tensor | None,
    *,
    causal: bool,
    training: bool,
) -> bool:
    """Validate invariants shared by forced mixed-FP16 SDPA backends."""

    if training or torch.is_grad_enabled() or valid_token_mask is not None:
        return False
    if not causal or query.device.type != "cuda":
        return False
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False
    if query.shape != key.shape or query.shape != value.shape:
        return False
    if query.dtype not in {torch.float16, torch.float32}:
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if query.device != key.device or query.device != value.device:
        return False
    if query.requires_grad or key.requires_grad or value.requires_grad:
        return False
    return all(tensor.stride(-1) == 1 for tensor in (query, key, value))


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

    The guard describes CUDA-measured long, short-graph, and mixed-core
    families. The mixed-core family also includes the measured wide and
    extreme-batch requests.
    Unsupported requests stay on the regular FP32 path instead of silently
    changing precision.
    """

    if not _mixed_fp16_inputs_are_legal(
        query,
        key,
        value,
        valid_token_mask,
        causal=causal,
        training=training,
    ):
        return False
    sequence_length = query.shape[-2]
    head_dim = query.shape[-1]
    long_family = (
        sequence_length >= _MIN_SEQUENCE_LENGTH and head_dim in _SUPPORTED_HEAD_DIMS
    )
    short_family = (
        query.shape[0] in _SHORT_BATCH_SIZES
        and sequence_length == _SHORT_SEQUENCE_LENGTH
        and query.shape[1] * head_dim in _SHORT_MODEL_DIMS
    )
    mixed_core_runtime_family = is_mixed_fp16_core_efficient_runtime_family(
        batch_size=query.shape[0],
        seq_len=sequence_length,
        num_heads=query.shape[1],
        head_dim=head_dim,
    )
    if not (long_family or short_family or mixed_core_runtime_family):
        return False
    if not torch.backends.cuda.mem_efficient_sdp_enabled():
        return False
    return torch.cuda.get_device_capability(query.device) >= _MIN_COMPUTE_CAPABILITY


def can_use_mixed_fp16_cudnn_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_token_mask: torch.Tensor | None = None,
    *,
    causal: bool = True,
    training: bool = False,
) -> bool:
    """Return whether the strict long-sequence cuDNN path is eligible."""

    if not _mixed_fp16_inputs_are_legal(
        query,
        key,
        value,
        valid_token_mask,
        causal=causal,
        training=training,
    ):
        return False
    if query.shape[-2] < _MIN_SEQUENCE_LENGTH or query.shape[-1] != _CUDNN_HEAD_DIM:
        return False
    if not torch.backends.cuda.cudnn_sdp_enabled():
        return False
    if not torch.backends.cudnn.is_available():
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

    The returned tensor preserves the caller's dtype. The regular mixed
    attention policy therefore returns FP32, while the mixed-core policy keeps
    the attention and surrounding linear layers inside one FP16 autocast
    region. Forcing the backend makes a successful backend marker truthful;
    an unavailable kernel is reported as an error and never falls through.
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

    return prevalidated_mixed_fp16_efficient_attention(
        query,
        key,
        value,
        scale=scale,
    )


def prevalidated_mixed_fp16_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, str]:
    """Run the forced kernel after an immutable ExecutionPlan validated it."""

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
    return output.to(query.dtype), MIXED_FP16_EFFICIENT_BACKEND


def mixed_fp16_cudnn_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_token_mask: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    causal: bool = True,
    training: bool = False,
) -> tuple[torch.Tensor, str]:
    """Run forced cuDNN SDPA on FP16 Q/K/V tensors.

    The backend is forced so an unavailable cuDNN kernel fails explicitly;
    this policy never falls through to Efficient, Flash, or Math SDPA.
    """

    if not can_use_mixed_fp16_cudnn_attention(
        query,
        key,
        value,
        valid_token_mask,
        causal=causal,
        training=training,
    ):
        raise ValueError(
            "attention tensors are incompatible with mixed FP16 cuDNN SDPA"
        )

    try:
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
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
            "forced FP16 cuDNN SDPA is unavailable for this CUDA request"
        ) from exc
    return output.to(query.dtype), MIXED_FP16_CUDNN_BACKEND


__all__ = [
    "MIXED_FP16_CUDNN_BACKEND",
    "MIXED_FP16_EFFICIENT_BACKEND",
    "can_use_mixed_fp16_cudnn_attention",
    "can_use_mixed_fp16_efficient_attention",
    "mixed_fp16_cudnn_attention",
    "mixed_fp16_efficient_attention",
    "prevalidated_mixed_fp16_efficient_attention",
]
