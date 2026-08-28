"""Small, shape-independent residual helpers."""

from __future__ import annotations

import torch


def can_use_residual_add(
    value: torch.Tensor,
    update: torch.Tensor,
    valid_token_mask: torch.Tensor | None = None,
) -> bool:
    """Return whether the native residual path preserves official semantics."""

    if value.shape != update.shape or value.device != update.device:
        return False
    if value.dtype != update.dtype or value.ndim != 3:
        return False
    if valid_token_mask is None:
        return True
    return bool(
        valid_token_mask.device == value.device
        and valid_token_mask.dtype == torch.bool
        and valid_token_mask.shape == value.shape[:2]
    )


def residual_add(
    value: torch.Tensor,
    update: torch.Tensor,
    valid_token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add a residual update and zero invalid query tokens when requested."""

    if not can_use_residual_add(value, update, valid_token_mask):
        raise ValueError("residual tensors have incompatible shape, dtype, or device")
    output = value + update
    if valid_token_mask is not None:
        output.masked_fill_(~valid_token_mask[..., None], 0)
    return output


__all__ = ["can_use_residual_add", "residual_add"]
