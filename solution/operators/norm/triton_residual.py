"""Triton residual-plus-LayerNorm for contiguous official model widths."""

from __future__ import annotations

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by environments without Triton.
    triton = None
    tl = None


TRITON_RESIDUAL_LAYER_NORM_BACKEND = "triton_residual_layer_norm"
_BLOCK_ROWS = 2
_NUM_WARPS = 2
_SUPPORTED_WIDTHS = frozenset({32, 128, 1024})
_SUPPORTED_BLOCK_ROWS = frozenset({1, 2, 4, 8})
_SUPPORTED_NUM_WARPS = frozenset({1, 2, 4, 8})


def _validate_launch_config(block_rows: int, num_warps: int) -> None:
    if (
        isinstance(block_rows, bool)
        or not isinstance(block_rows, int)
        or block_rows not in _SUPPORTED_BLOCK_ROWS
        or isinstance(num_warps, bool)
        or not isinstance(num_warps, int)
        or num_warps not in _SUPPORTED_NUM_WARPS
    ):
        raise ValueError("unsupported Triton residual LayerNorm launch configuration")


if triton is not None and tl is not None:

    @triton.jit
    def _residual_layer_norm_kernel(
        value,
        update,
        weight,
        bias,
        residual,
        normalized,
        row_count,
        eps: tl.constexpr,
        width: tl.constexpr,
        block_rows: tl.constexpr,
    ) -> None:
        row_start = tl.program_id(0) * block_rows
        rows = row_start + tl.arange(0, block_rows)[:, None]
        columns = tl.arange(0, width)[None, :]
        offsets = rows * width + columns
        mask = rows < row_count

        summed = tl.load(value + offsets, mask=mask, other=0.0) + tl.load(
            update + offsets,
            mask=mask,
            other=0.0,
        )
        mean = tl.sum(summed, axis=1) / width
        centered = summed - mean[:, None]
        variance = tl.sum(centered * centered, axis=1) / width
        reciprocal_std = tl.rsqrt(variance + eps)
        scale = tl.load(weight + columns)
        shift = tl.load(bias + columns)
        output = centered * reciprocal_std[:, None] * scale + shift

        tl.store(residual + offsets, summed, mask=mask)
        tl.store(normalized + offsets, output, mask=mask)


def triton_residual_layer_norm_available() -> bool:
    """Return whether the optional Triton runtime can load this specialization."""

    return triton is not None and tl is not None


def can_use_triton_residual_layer_norm(
    value: torch.Tensor,
    update: torch.Tensor,
    layer_norm: nn.LayerNorm,
) -> bool:
    """Return whether the strictly bounded Triton specialization is eligible."""

    if triton is None or tl is None or not isinstance(layer_norm, nn.LayerNorm):
        return False
    if torch.is_grad_enabled() or value.device.type != "cuda":
        return False
    if value.dtype != torch.float32 or update.dtype != value.dtype:
        return False
    if value.shape != update.shape or value.device != update.device:
        return False
    if value.ndim < 2 or not value.is_contiguous() or not update.is_contiguous():
        return False
    if value.shape[-1] not in _SUPPORTED_WIDTHS:
        return False
    width = value.shape[-1]
    if tuple(layer_norm.normalized_shape) != (width,):
        return False
    if layer_norm.weight is None or layer_norm.bias is None:
        return False
    return all(
        parameter.device == value.device and parameter.dtype == value.dtype
        for parameter in (layer_norm.weight, layer_norm.bias)
    )


def triton_residual_layer_norm(
    value: torch.Tensor,
    update: torch.Tensor,
    layer_norm: nn.LayerNorm,
    *,
    block_rows: int = _BLOCK_ROWS,
    num_warps: int = _NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Run the measured specialization without a silent implementation fallback."""

    _validate_launch_config(block_rows, num_warps)
    if not can_use_triton_residual_layer_norm(value, update, layer_norm):
        raise RuntimeError(
            "Triton residual LayerNorm is ineligible for the requested inputs"
        )
    assert triton is not None
    assert layer_norm.weight is not None
    assert layer_norm.bias is not None
    width = value.shape[-1]
    row_count = value.numel() // width
    residual = torch.empty_like(value)
    normalized = torch.empty_like(value)
    try:
        _residual_layer_norm_kernel[(triton.cdiv(row_count, block_rows),)](
            value,
            update,
            layer_norm.weight,
            layer_norm.bias,
            residual,
            normalized,
            row_count,
            eps=layer_norm.eps,
            width=width,
            block_rows=block_rows,
            num_warps=num_warps,
        )
    except Exception as exc:
        raise RuntimeError("Triton residual LayerNorm execution failed") from exc
    return residual, normalized, TRITON_RESIDUAL_LAYER_NORM_BACKEND


__all__ = [
    "TRITON_RESIDUAL_LAYER_NORM_BACKEND",
    "can_use_triton_residual_layer_norm",
    "triton_residual_layer_norm",
    "triton_residual_layer_norm_available",
]
