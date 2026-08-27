"""Triton preprocessing for the native FP32 attention softmax path."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except (ImportError, OSError):
    triton = None
    tl = None


TRITON_ATTENTION_PREPROCESS_AVAILABLE = triton is not None and tl is not None
_SUPPORTED_PREPROCESS_SHAPES = frozenset({(64, 32), (2048, 64)})


def supports_triton_attention_preprocess(
    *,
    device_type: str,
    dtype: torch.dtype,
    sequence_length: int,
    head_dim: int,
    mask_compatible: bool,
) -> bool:
    """Return whether static context can use the preprocessing kernel."""

    return bool(
        TRITON_ATTENTION_PREPROCESS_AVAILABLE
        and device_type == "cuda"
        and dtype == torch.float16
        and (sequence_length, head_dim) in _SUPPORTED_PREPROCESS_SHAPES
        and mask_compatible
    )


if TRITON_ATTENTION_PREPROCESS_AVAILABLE:

    @triton.jit
    def _scale_mask_to_fp32_kernel(
        scores_ptr,
        valid_mask_ptr,
        output_ptr,
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

        values = tl.load(scores_ptr + row_offsets, mask=column_mask, other=0.0)
        # Preserve the reference boundary: scale in FP16 before promotion.
        values = (values.to(tl.float32) * scale).to(tl.float16).to(tl.float32)

        if IS_CAUSAL:
            query_index = row_index % sequence_length
            values = tl.where(column_offsets <= query_index, values, -float("inf"))

        if HAS_VALID_MASK:
            batch_head_index = row_index // sequence_length
            batch_index = batch_head_index // num_heads
            valid_keys = tl.load(
                valid_mask_ptr + batch_index * sequence_length + column_offsets,
                mask=column_mask,
                other=False,
            )
            values = tl.where(valid_keys, values, -float("inf"))

        tl.store(output_ptr + row_offsets, values, mask=column_mask)


def can_use_triton_attention_preprocess(
    scores: torch.Tensor,
    valid_token_mask: torch.Tensor | None,
    head_dim: int,
) -> bool:
    """Return whether the long-sequence FP16 route is supported."""

    if not scores.is_cuda or not scores.is_contiguous():
        return False
    if scores.dtype != torch.float16 or scores.ndim != 4:
        return False
    sequence_length = scores.shape[-1]
    if scores.shape[-2] != sequence_length:
        return False
    mask_compatible = valid_token_mask is None or (
        valid_token_mask.is_cuda
        and valid_token_mask.device == scores.device
        and valid_token_mask.dtype == torch.bool
        and valid_token_mask.is_contiguous()
        and valid_token_mask.shape == (scores.shape[0], scores.shape[-1])
    )
    return supports_triton_attention_preprocess(
        device_type=scores.device.type,
        dtype=scores.dtype,
        sequence_length=sequence_length,
        head_dim=head_dim,
        mask_compatible=mask_compatible,
    )


def triton_scale_mask_to_fp32(
    scores: torch.Tensor,
    valid_token_mask: torch.Tensor | None,
    *,
    head_dim: int,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    """Scale and mask FP16 scores while writing native-softmax FP32 input."""

    if not can_use_triton_attention_preprocess(
        scores,
        valid_token_mask,
        head_dim,
    ):
        raise ValueError("scores are not supported by the Triton preprocessor")
    assert triton is not None

    batch_size, num_heads, sequence_length, _ = scores.shape
    output = torch.empty_like(scores, dtype=torch.float32)
    block_size = triton.next_power_of_2(sequence_length)
    num_warps = 4 if sequence_length <= 128 else 8
    grid = (batch_size * num_heads * sequence_length,)
    mask_pointer = scores if valid_token_mask is None else valid_token_mask
    _scale_mask_to_fp32_kernel[grid](
        scores,
        mask_pointer,
        output,
        scale,
        sequence_length=sequence_length,
        num_heads=num_heads,
        HAS_VALID_MASK=valid_token_mask is not None,
        IS_CAUSAL=causal,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return output
