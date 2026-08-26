"""Bounded two-pass streaming attention candidate for the long FP16 case.

The kernel deliberately targets only ``[1, 8, 2048, 64]`` inference. It
recomputes QK tiles instead of materializing either the score or probability
tensor. Unlike a conventional one-pass FlashAttention kernel, the second pass
normalizes each probability and rounds it to FP16 before the PV dot product.
That boundary mirrors the official reference more closely and makes this a
plausible, though still experimental, comparator candidate.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except (ImportError, OSError):
    triton = None
    tl = None


TRITON_ONLINE_ATTENTION_AVAILABLE = triton is not None and tl is not None

_BATCH_SIZE = 1
_NUM_HEADS = 8
_SEQUENCE_LENGTH = 2048
_HEAD_DIM = 64
_REFERENCE_SCALE = _HEAD_DIM**-0.5


if TRITON_ONLINE_ATTENTION_AVAILABLE:

    @triton.jit
    def _two_pass_streaming_attention_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        valid_mask_ptr,
        output_ptr,
        query_stride_batch,
        query_stride_head,
        query_stride_sequence,
        query_stride_feature,
        key_stride_batch,
        key_stride_head,
        key_stride_sequence,
        key_stride_feature,
        value_stride_batch,
        value_stride_head,
        value_stride_sequence,
        value_stride_feature,
        output_stride_batch,
        output_stride_sequence,
        output_stride_head,
        output_stride_feature,
        scale,
        HAS_VALID_MASK: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        SEQUENCE_LENGTH: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        query_block = tl.program_id(axis=0)
        batch_head_index = tl.program_id(axis=1)
        batch_index = batch_head_index // NUM_HEADS
        head_index = batch_head_index % NUM_HEADS

        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        feature_offsets = tl.arange(0, HEAD_DIM)
        key_block_offsets = tl.arange(0, BLOCK_N)
        query_mask = query_offsets < SEQUENCE_LENGTH

        query_pointers = (
            query_ptr
            + batch_index * query_stride_batch
            + head_index * query_stride_head
            + query_offsets[:, None] * query_stride_sequence
            + feature_offsets[None, :] * query_stride_feature
        )
        query = tl.load(query_pointers, mask=query_mask[:, None], other=0.0)

        row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # First pass computes only stable row-wise softmax statistics. QK is
        # explicitly rounded before and after scaling to preserve the two FP16
        # boundaries in the official implementation.
        for key_start in range(0, SEQUENCE_LENGTH, BLOCK_N):
            key_offsets = key_start + key_block_offsets
            key_mask = key_offsets < SEQUENCE_LENGTH
            key_pointers = (
                key_ptr
                + batch_index * key_stride_batch
                + head_index * key_stride_head
                + key_offsets[:, None] * key_stride_sequence
                + feature_offsets[None, :] * key_stride_feature
            )
            key = tl.load(key_pointers, mask=key_mask[:, None], other=0.0)
            scores = tl.dot(query, key.T)
            scores = scores.to(tl.float16).to(tl.float32)
            scores = (scores * scale).to(tl.float16).to(tl.float32)

            score_mask = query_mask[:, None] & key_mask[None, :]
            if IS_CAUSAL:
                score_mask &= key_offsets[None, :] <= query_offsets[:, None]
            if HAS_VALID_MASK:
                valid_keys = tl.load(
                    valid_mask_ptr
                    + batch_index * SEQUENCE_LENGTH
                    + key_offsets,
                    mask=key_mask,
                    other=False,
                )
                score_mask &= valid_keys[None, :]
            scores = tl.where(score_mask, scores, -float("inf"))

            block_max = tl.max(scores, axis=1)
            next_max = tl.maximum(row_max, block_max)
            correction = tl.exp(row_max - next_max)
            numerator = tl.exp(scores - next_max[:, None])
            row_sum = row_sum * correction + tl.sum(numerator, axis=1)
            row_max = next_max

        accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

        # The second pass recomputes QK, applies the final normalization, then
        # rounds probabilities before PV just like ``softmax(...).to(fp16)``.
        for key_start in range(0, SEQUENCE_LENGTH, BLOCK_N):
            key_offsets = key_start + key_block_offsets
            key_mask = key_offsets < SEQUENCE_LENGTH
            key_pointers = (
                key_ptr
                + batch_index * key_stride_batch
                + head_index * key_stride_head
                + key_offsets[:, None] * key_stride_sequence
                + feature_offsets[None, :] * key_stride_feature
            )
            key = tl.load(key_pointers, mask=key_mask[:, None], other=0.0)
            scores = tl.dot(query, key.T)
            scores = scores.to(tl.float16).to(tl.float32)
            scores = (scores * scale).to(tl.float16).to(tl.float32)

            score_mask = query_mask[:, None] & key_mask[None, :]
            if IS_CAUSAL:
                score_mask &= key_offsets[None, :] <= query_offsets[:, None]
            if HAS_VALID_MASK:
                valid_keys = tl.load(
                    valid_mask_ptr
                    + batch_index * SEQUENCE_LENGTH
                    + key_offsets,
                    mask=key_mask,
                    other=False,
                )
                score_mask &= valid_keys[None, :]
            scores = tl.where(score_mask, scores, -float("inf"))
            probabilities = tl.exp(scores - row_max[:, None]) / row_sum[:, None]
            probabilities = probabilities.to(tl.float16)

            value_pointers = (
                value_ptr
                + batch_index * value_stride_batch
                + head_index * value_stride_head
                + key_offsets[:, None] * value_stride_sequence
                + feature_offsets[None, :] * value_stride_feature
            )
            values = tl.load(value_pointers, mask=key_mask[:, None], other=0.0)
            accumulator = tl.dot(probabilities, values, accumulator)

        output_pointers = (
            output_ptr
            + batch_index * output_stride_batch
            + query_offsets[:, None] * output_stride_sequence
            + head_index * output_stride_head
            + feature_offsets[None, :] * output_stride_feature
        )
        tl.store(
            output_pointers,
            accumulator.to(tl.float16),
            mask=query_mask[:, None],
        )


def can_use_triton_online_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_token_mask: torch.Tensor | None,
) -> bool:
    """Return whether tensors match the single bounded experimental route."""

    if not TRITON_ONLINE_ATTENTION_AVAILABLE:
        return False
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        return False
    if query.device != key.device or query.device != value.device:
        return False
    if query.dtype != torch.float16 or key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    expected_shape = (_BATCH_SIZE, _NUM_HEADS, _SEQUENCE_LENGTH, _HEAD_DIM)
    if query.shape != expected_shape or key.shape != expected_shape:
        return False
    if value.shape != expected_shape:
        return False
    if query.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1:
        return False
    if query.requires_grad or key.requires_grad or value.requires_grad:
        return False
    if torch.is_grad_enabled():
        return False
    if torch.version.cuda is None:
        return False
    if torch.cuda.get_device_capability(query.device)[0] < 8:
        return False
    if valid_token_mask is None:
        return True
    return (
        valid_token_mask.is_cuda
        and valid_token_mask.device == query.device
        and valid_token_mask.dtype == torch.bool
        and valid_token_mask.is_contiguous()
        and valid_token_mask.shape == (_BATCH_SIZE, _SEQUENCE_LENGTH)
    )


def triton_online_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_token_mask: torch.Tensor | None,
    *,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    """Run the bounded candidate and return a BHSD view over BSHD storage."""

    if not can_use_triton_online_attention(
        query,
        key,
        value,
        valid_token_mask,
    ):
        raise ValueError("tensors are not supported by Triton online attention")
    if not math.isclose(scale, _REFERENCE_SCALE, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f"scale must be exactly {_REFERENCE_SCALE} for head_dim=64")
    assert triton is not None

    output_bshd = torch.empty(
        (_BATCH_SIZE, _SEQUENCE_LENGTH, _NUM_HEADS, _HEAD_DIM),
        device=query.device,
        dtype=query.dtype,
    )
    block_m = 64
    block_n = 64
    grid = (triton.cdiv(_SEQUENCE_LENGTH, block_m), _BATCH_SIZE * _NUM_HEADS)
    mask_pointer = query if valid_token_mask is None else valid_token_mask
    _two_pass_streaming_attention_kernel[grid](
        query,
        key,
        value,
        mask_pointer,
        output_bshd,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *output_bshd.stride(),
        scale,
        HAS_VALID_MASK=valid_token_mask is not None,
        IS_CAUSAL=causal,
        SEQUENCE_LENGTH=_SEQUENCE_LENGTH,
        NUM_HEADS=_NUM_HEADS,
        HEAD_DIM=_HEAD_DIM,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )
    return output_bshd.transpose(1, 2)
