"""Triton layout conversion for one packed QKV projection.

The input follows the output layout of ``Linear(D, 3 * D)``:
``[batch, sequence, 3, heads, head_dim]``. The output is a single contiguous
allocation laid out as ``[3, batch, heads, sequence, head_dim]``. Unbinding the
leading dimension therefore gives contiguous Q, K and V tensors without three
separate ``transpose().contiguous()`` operations.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except (ImportError, OSError):
    triton = None
    tl = None


TRITON_QKV_LAYOUT_AVAILABLE = triton is not None and tl is not None


if TRITON_QKV_LAYOUT_AVAILABLE:

    @triton.jit
    def _packed_qkv_to_bhsd_kernel(
        packed_ptr,
        output_ptr,
        sequence_length: tl.constexpr,
        model_width: tl.constexpr,
        head_dim: tl.constexpr,
        projection_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        token_index = tl.program_id(axis=0)
        feature_offsets = (
            tl.program_id(axis=1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        )
        feature_mask = feature_offsets < model_width

        batch_index = token_index // sequence_length
        sequence_index = token_index - batch_index * sequence_length
        head_index = feature_offsets // head_dim
        head_feature = feature_offsets - head_index * head_dim

        source_base = token_index * 3 * model_width
        destination_offsets = (
            ((batch_index * (model_width // head_dim) + head_index)
            * sequence_length + sequence_index)
            * head_dim
            + head_feature
        )

        query = tl.load(packed_ptr + source_base + feature_offsets, mask=feature_mask)
        key = tl.load(
            packed_ptr + source_base + model_width + feature_offsets,
            mask=feature_mask,
        )
        value = tl.load(
            packed_ptr + source_base + 2 * model_width + feature_offsets,
            mask=feature_mask,
        )
        tl.store(output_ptr + destination_offsets, query, mask=feature_mask)
        tl.store(
            output_ptr + projection_size + destination_offsets,
            key,
            mask=feature_mask,
        )
        tl.store(
            output_ptr + 2 * projection_size + destination_offsets,
            value,
            mask=feature_mask,
        )


def can_use_triton_qkv_layout(
    packed_qkv: torch.Tensor,
    num_heads: int,
) -> bool:
    """Return whether the project-specialized Triton layout path is applicable."""

    if not TRITON_QKV_LAYOUT_AVAILABLE:
        return False
    if not packed_qkv.is_cuda or not packed_qkv.is_contiguous():
        return False
    if packed_qkv.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if packed_qkv.ndim != 3 or packed_qkv.shape[-1] % 3:
        return False

    model_width = packed_qkv.shape[-1] // 3
    if model_width % num_heads:
        return False
    head_dim = model_width // num_heads
    return 16 <= head_dim <= 128 and head_dim & (head_dim - 1) == 0


def triton_qkv_to_bhsd(
    packed_qkv: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Convert packed QKV to one contiguous ``[3, B, H, S, Dh]`` tensor."""

    if not can_use_triton_qkv_layout(packed_qkv, num_heads):
        raise ValueError("packed QKV tensor is not supported by the Triton layout path")
    assert triton is not None

    batch_size, sequence_length, packed_width = packed_qkv.shape
    model_width = packed_width // 3
    head_dim = model_width // num_heads
    output = torch.empty(
        (3, batch_size, num_heads, sequence_length, head_dim),
        device=packed_qkv.device,
        dtype=packed_qkv.dtype,
    )
    projection_size = batch_size * num_heads * sequence_length * head_dim
    block_size = 256
    grid = (batch_size * sequence_length, triton.cdiv(model_width, block_size))
    _packed_qkv_to_bhsd_kernel[grid](
        packed_qkv,
        output,
        sequence_length=sequence_length,
        model_width=model_width,
        head_dim=head_dim,
        projection_size=projection_size,
        BLOCK_SIZE=block_size,
    )
    return output
