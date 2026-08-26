"""Triton softmax for the explicit reference-order attention path."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except (ImportError, OSError):
    triton = None
    tl = None


TRITON_ATTENTION_SOFTMAX_AVAILABLE = triton is not None and tl is not None


if TRITON_ATTENTION_SOFTMAX_AVAILABLE:

    @triton.jit
    def _scale_mask_inplace_fp16_kernel(
        scores_ptr,
        valid_mask_ptr,
        scale,
        sequence_length: tl.constexpr,
        num_heads: tl.constexpr,
        HAS_VALID_MASK: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_index = tl.program_id(axis=0)
        column_offsets = tl.arange(0, BLOCK_SIZE)
        column_mask = column_offsets < sequence_length
        row_offsets = row_index * sequence_length + column_offsets

        scores = tl.load(
            scores_ptr + row_offsets,
            mask=column_mask,
            other=0.0,
        )
        # Preserve the official FP16 scaling boundary before softmax promotion.
        scores = (scores.to(tl.float32) * scale).to(tl.float16)
        if IS_CAUSAL:
            query_index = row_index % sequence_length
            scores = tl.where(column_offsets <= query_index, scores, -float("inf"))
        if HAS_VALID_MASK:
            batch_head_index = row_index // sequence_length
            batch_index = batch_head_index // num_heads
            valid_keys = tl.load(
                valid_mask_ptr + batch_index * sequence_length + column_offsets,
                mask=column_mask,
                other=False,
            )
            scores = tl.where(valid_keys, scores, -float("inf"))

        tl.store(scores_ptr + row_offsets, scores, mask=column_mask)

    @triton.jit
    def _scale_mask_softmax_kernel(
        scores_ptr,
        valid_mask_ptr,
        probabilities_ptr,
        scale,
        sequence_length: tl.constexpr,
        num_heads: tl.constexpr,
        HAS_VALID_MASK: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_index = tl.program_id(axis=0)
        column_offsets = tl.arange(0, BLOCK_SIZE)
        column_mask = column_offsets < sequence_length

        query_index = row_index % sequence_length
        batch_head_index = row_index // sequence_length
        batch_index = batch_head_index // num_heads
        row_offsets = row_index * sequence_length + column_offsets

        scores = tl.load(scores_ptr + row_offsets, mask=column_mask, other=-float("inf"))
        # Match the reference boundary: scale in FP16, then promote for softmax.
        scores = (scores.to(tl.float32) * scale).to(tl.float16).to(tl.float32)
        if IS_CAUSAL:
            scores = tl.where(column_offsets <= query_index, scores, -float("inf"))
        if HAS_VALID_MASK:
            valid_keys = tl.load(
                valid_mask_ptr + batch_index * sequence_length + column_offsets,
                mask=column_mask,
                other=False,
            )
            scores = tl.where(valid_keys, scores, -float("inf"))

        row_max = tl.max(scores, axis=0)
        numerator = tl.exp(scores - row_max)
        denominator = tl.sum(numerator, axis=0)
        probabilities = numerator / denominator
        tl.store(
            probabilities_ptr + row_offsets,
            probabilities,
            mask=column_mask,
        )


def can_use_triton_attention_softmax(
    scores: torch.Tensor,
    valid_token_mask: torch.Tensor | None,
    head_dim: int,
) -> bool:
    """Return whether scores match the deliberately narrow tuned shape family."""

    if not TRITON_ATTENTION_SOFTMAX_AVAILABLE:
        return False
    if not scores.is_cuda or not scores.is_contiguous():
        return False
    if scores.dtype != torch.float16 or scores.ndim != 4:
        return False
    if scores.shape[-1] != scores.shape[-2] or scores.shape[-1] not in (512, 2048):
        return False
    if head_dim != 64:
        return False
    if valid_token_mask is None:
        return True
    return (
        valid_token_mask.is_cuda
        and valid_token_mask.device == scores.device
        and valid_token_mask.dtype == torch.bool
        and valid_token_mask.is_contiguous()
        and valid_token_mask.shape
        == (scores.shape[0], scores.shape[-1])
    )


def triton_scale_mask_softmax(
    scores: torch.Tensor,
    valid_token_mask: torch.Tensor | None,
    *,
    head_dim: int,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    """Apply scale, optional masks and FP32 softmax, then store in FP16."""

    if not can_use_triton_attention_softmax(
        scores,
        valid_token_mask,
        head_dim,
    ):
        raise ValueError("scores are not supported by the Triton attention softmax")
    assert triton is not None

    batch_size, num_heads, sequence_length, _ = scores.shape
    probabilities = torch.empty_like(scores)
    block_size = triton.next_power_of_2(sequence_length)
    grid = (batch_size * num_heads * sequence_length,)
    mask_pointer = scores if valid_token_mask is None else valid_token_mask
    _scale_mask_softmax_kernel[grid](
        scores,
        mask_pointer,
        probabilities,
        scale,
        sequence_length=sequence_length,
        num_heads=num_heads,
        HAS_VALID_MASK=valid_token_mask is not None,
        IS_CAUSAL=causal,
        BLOCK_SIZE=block_size,
        num_warps=8,
        num_stages=1,
    )
    return probabilities


def can_use_s512_native_half_softmax(
    scores: torch.Tensor,
    valid_token_mask: torch.Tensor | None,
    head_dim: int,
) -> bool:
    """Return whether inputs match the deliberately narrow S512 candidate."""

    if not TRITON_ATTENTION_SOFTMAX_AVAILABLE:
        return False
    if not scores.is_cuda or scores.dtype != torch.float16:
        return False
    if not scores.is_contiguous() or scores.shape != (8, 8, 512, 512):
        return False
    if head_dim != 64:
        return False
    if valid_token_mask is None:
        return True
    return bool(
        valid_token_mask.is_cuda
        and valid_token_mask.device == scores.device
        and valid_token_mask.dtype == torch.bool
        and valid_token_mask.is_contiguous()
        and valid_token_mask.shape == (8, 512)
    )


def s512_scale_mask_native_half_softmax(
    scores: torch.Tensor,
    valid_token_mask: torch.Tensor | None,
    *,
    head_dim: int,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    """Fuse S512 FP16 scale/masks before native FP32-accumulating softmax."""

    if not can_use_s512_native_half_softmax(
        scores,
        valid_token_mask,
        head_dim,
    ):
        raise ValueError("scores are not supported by the S512 softmax candidate")
    assert triton is not None

    batch_size, num_heads, sequence_length, _ = scores.shape
    mask_pointer = scores if valid_token_mask is None else valid_token_mask
    grid = (batch_size * num_heads * sequence_length,)
    _scale_mask_inplace_fp16_kernel[grid](
        scores,
        mask_pointer,
        scale,
        sequence_length=sequence_length,
        num_heads=num_heads,
        HAS_VALID_MASK=valid_token_mask is not None,
        IS_CAUSAL=causal,
        BLOCK_SIZE=512,
        num_warps=8,
        num_stages=1,
    )

    # Native CUDA softmax reads FP16, accumulates in FP32 and writes FP16.
    return torch.ops.aten._softmax.default(scores, -1, False)
