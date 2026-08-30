"""Outer microbatch execution for Shape 14 streamed plans."""

from __future__ import annotations

from collections.abc import Callable

import torch

ChunkForward = Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]


def execute_streamed(
    value: torch.Tensor,
    valid_token_mask: torch.Tensor | None,
    *,
    microbatch_size: int,
    forward_chunk: ChunkForward,
) -> torch.Tensor:
    """Execute one logical batch as ordered, preallocated microbatches."""

    if (
        isinstance(microbatch_size, bool)
        or not isinstance(microbatch_size, int)
        or microbatch_size <= 0
    ):
        raise ValueError("microbatch_size must be a positive integer")
    batch_size = value.shape[0]
    if batch_size <= 0:
        raise ValueError("streamed execution requires a non-empty batch")
    if batch_size % microbatch_size:
        raise ValueError("microbatch_size must divide the batch size")
    if valid_token_mask is not None and valid_token_mask.shape[0] != batch_size:
        raise ValueError("valid_token_mask batch dimension must match the input")

    output: torch.Tensor | None = None
    for start in range(0, batch_size, microbatch_size):
        end = start + microbatch_size
        mask_slice = None if valid_token_mask is None else valid_token_mask[start:end]
        chunk = forward_chunk(value[start:end], mask_slice)
        if output is None:
            output = torch.empty(
                (batch_size, *chunk.shape[1:]),
                dtype=chunk.dtype,
                device=chunk.device,
            )
        output[start:end].copy_(chunk)

    assert output is not None
    return output


__all__ = ["execute_streamed"]
