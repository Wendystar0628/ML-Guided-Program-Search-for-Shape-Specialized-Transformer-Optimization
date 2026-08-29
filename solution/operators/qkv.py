"""Packed QKV layout helpers shared by every official shape."""

from __future__ import annotations

import torch


def can_split_qkv(packed_qkv: torch.Tensor, num_heads: int) -> bool:
    """Return whether ``packed_qkv`` can be viewed as three BHSD tensors."""

    if packed_qkv.ndim != 3 or num_heads <= 0:
        return False
    if packed_qkv.stride(-1) != 1:
        return False
    packed_width = packed_qkv.shape[-1]
    if packed_width <= 0 or packed_width % 3:
        return False
    model_width = packed_width // 3
    return model_width % num_heads == 0


def split_qkv(
    packed_qkv: torch.Tensor,
    num_heads: int,
    *,
    contiguous: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a ``[B, S, 3D]`` projection into ``[B, H, S, Dh]`` tensors.

    The default path is a view and therefore performs no layout kernel or
    allocation. ``contiguous=True`` remains an explicit experiment knob for
    backends that benefit from materialized BHSD inputs.
    """

    if not can_split_qkv(packed_qkv, num_heads):
        raise ValueError("packed QKV shape is incompatible with num_heads")

    packed_width = packed_qkv.shape[-1]
    model_width = packed_width // 3
    head_dim = model_width // num_heads
    packed_heads = packed_qkv.unflatten(
        -1,
        (3, num_heads, head_dim),
    )
    query, key, value = (
        component.transpose(1, 2) for component in packed_heads.unbind(dim=2)
    )
    if contiguous:
        return query.contiguous(), key.contiguous(), value.contiguous()
    return query, key, value


__all__ = ["can_split_qkv", "split_qkv"]
