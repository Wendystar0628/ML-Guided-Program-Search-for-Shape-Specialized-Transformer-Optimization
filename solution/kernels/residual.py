"""Low-risk Triton fusion for residual addition and padding zeroing."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except (ImportError, OSError):
    triton = None
    tl = None


TRITON_RESIDUAL_AVAILABLE = triton is not None and tl is not None


if TRITON_RESIDUAL_AVAILABLE:

    @triton.jit
    def _residual_add_padding_kernel(
        value_ptr,
        update_ptr,
        valid_mask_ptr,
        output_ptr,
        element_count,
        model_width: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        element_mask = offsets < element_count
        token_offsets = offsets // model_width
        valid_tokens = tl.load(valid_mask_ptr + token_offsets, mask=element_mask)
        value = tl.load(value_ptr + offsets, mask=element_mask)
        update = tl.load(update_ptr + offsets, mask=element_mask)
        output = tl.where(valid_tokens, value + update, 0.0)
        tl.store(output_ptr + offsets, output, mask=element_mask)


def can_use_triton_residual(
    value: torch.Tensor,
    update: torch.Tensor,
    valid_token_mask: torch.Tensor,
) -> bool:
    """Return whether the residual tensors support the fused pointwise path."""

    return bool(
        TRITON_RESIDUAL_AVAILABLE
        and value.is_cuda
        and update.is_cuda
        and valid_token_mask.is_cuda
        and value.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and update.dtype == value.dtype
        and valid_token_mask.dtype == torch.bool
        and value.shape == update.shape
        and value.ndim == 3
        and valid_token_mask.shape == value.shape[:2]
        and value.is_contiguous()
        and update.is_contiguous()
        and valid_token_mask.is_contiguous()
    )


def triton_residual_add_padding(
    value: torch.Tensor,
    update: torch.Tensor,
    valid_token_mask: torch.Tensor,
) -> torch.Tensor:
    """Add one residual update and write exact zeros for invalid tokens."""

    if not can_use_triton_residual(value, update, valid_token_mask):
        raise ValueError("residual tensors are not supported by the Triton fusion")
    assert triton is not None

    output = torch.empty_like(value)
    block_size = 256
    grid = (triton.cdiv(value.numel(), block_size),)
    _residual_add_padding_kernel[grid](
        value,
        update,
        valid_token_mask,
        output,
        value.numel(),
        model_width=value.shape[-1],
        BLOCK_SIZE=block_size,
    )
    return output
