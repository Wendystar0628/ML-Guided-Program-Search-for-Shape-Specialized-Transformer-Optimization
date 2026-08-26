"""Triton fusion for FP32 attention probabilities and FP16 values."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except (ImportError, OSError):
    triton = None
    tl = None


TRITON_ATTENTION_PV_AVAILABLE = triton is not None and tl is not None


if TRITON_ATTENTION_PV_AVAILABLE:

    @triton.jit
    def _fp32_probability_value_kernel(
        probabilities_ptr,
        value_ptr,
        output_ptr,
        probability_stride_batch,
        probability_stride_head,
        probability_stride_query,
        probability_stride_key,
        value_stride_batch,
        value_stride_head,
        value_stride_key,
        value_stride_feature,
        output_stride_batch,
        output_stride_query,
        output_stride_head,
        output_stride_feature,
        sequence_length: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        query_block = tl.program_id(axis=0)
        batch_head_index = tl.program_id(axis=1)
        batch_index = batch_head_index // num_heads
        head_index = batch_head_index % num_heads

        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        feature_offsets = tl.arange(0, head_dim)
        query_mask = query_offsets < sequence_length

        accumulator = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)
        for key_start in range(0, sequence_length, BLOCK_K):
            key_offsets = key_start + tl.arange(0, BLOCK_K)
            key_mask = key_offsets < sequence_length

            probability_offsets = (
                batch_index * probability_stride_batch
                + head_index * probability_stride_head
                + query_offsets[:, None] * probability_stride_query
                + key_offsets[None, :] * probability_stride_key
            )
            probabilities = tl.load(
                probabilities_ptr + probability_offsets,
                mask=query_mask[:, None] & key_mask[None, :],
                other=0.0,
            ).to(tl.float16)

            value_offsets = (
                batch_index * value_stride_batch
                + head_index * value_stride_head
                + key_offsets[:, None] * value_stride_key
                + feature_offsets[None, :] * value_stride_feature
            )
            values = tl.load(
                value_ptr + value_offsets,
                mask=key_mask[:, None],
                other=0.0,
            )
            accumulator = tl.dot(probabilities, values, accumulator)

        output_offsets = (
            batch_index * output_stride_batch
            + query_offsets[:, None] * output_stride_query
            + head_index * output_stride_head
            + feature_offsets[None, :] * output_stride_feature
        )
        tl.store(
            output_ptr + output_offsets,
            accumulator.to(tl.float16),
            mask=query_mask[:, None],
        )


def can_use_triton_fp32_probability_value(
    probabilities: torch.Tensor,
    value: torch.Tensor,
) -> bool:
    """Return whether tensors match the deliberately narrow S=2048 path."""

    if not TRITON_ATTENTION_PV_AVAILABLE:
        return False
    if not probabilities.is_cuda or not value.is_cuda:
        return False
    if probabilities.device != value.device:
        return False
    if probabilities.dtype != torch.float32 or value.dtype != torch.float16:
        return False
    if probabilities.ndim != 4 or value.ndim != 4:
        return False
    if probabilities.shape != (1, 8, 2048, 2048):
        return False
    if value.shape != (1, 8, 2048, 64):
        return False
    if not probabilities.is_contiguous():
        return False
    return value.stride(-1) == 1


def triton_fp32_probability_value(
    probabilities: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Multiply probabilities by values and return a BHSD view over BSHD storage."""

    if not can_use_triton_fp32_probability_value(probabilities, value):
        raise ValueError(
            "probabilities and value are not supported by the Triton PV path"
        )
    assert triton is not None

    batch_size, num_heads, sequence_length, _ = probabilities.shape
    head_dim = value.shape[-1]
    output_bshd = torch.empty(
        (batch_size, sequence_length, num_heads, head_dim),
        device=value.device,
        dtype=value.dtype,
    )

    block_m = 128
    block_k = 32
    grid = (triton.cdiv(sequence_length, block_m), batch_size * num_heads)
    _fp32_probability_value_kernel[grid](
        probabilities,
        value,
        output_bshd,
        *probabilities.stride(),
        *value.stride(),
        *output_bshd.stride(),
        sequence_length=sequence_length,
        num_heads=num_heads,
        head_dim=head_dim,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=8,
        num_stages=2,
    )

    return output_bshd.transpose(1, 2)
