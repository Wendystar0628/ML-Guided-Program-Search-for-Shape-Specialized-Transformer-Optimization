"""Mixed-precision residual-plus-LayerNorm for width-128 tensors."""

from __future__ import annotations

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised without the optional runtime.
    triton = None
    tl = None


TRITON_MIXED_RESIDUAL_LAYER_NORM_BACKEND = "triton_mixed_residual_layer_norm"
_WIDTH = 128
_BLOCK_ROWS = 2
_NUM_WARPS = 2
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
        raise ValueError(
            "unsupported Triton mixed residual LayerNorm launch configuration"
        )


if triton is not None and tl is not None:

    @triton.jit
    def _mixed_residual_layer_norm_kernel(
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
        final_boundary: tl.constexpr,
    ) -> None:
        row_start = tl.program_id(0) * block_rows
        rows = row_start + tl.arange(0, block_rows)[:, None]
        columns = tl.arange(0, width)[None, :]
        offsets = rows * width + columns
        mask = rows < row_count

        residual_value = tl.load(value + offsets, mask=mask, other=0.0).to(tl.float32)
        branch_update = tl.load(update + offsets, mask=mask, other=0.0).to(tl.float32)
        summed = residual_value + branch_update
        mean = tl.sum(summed, axis=1) / width
        centered = summed - mean[:, None]
        variance = tl.sum(centered * centered, axis=1) / width
        reciprocal_std = tl.rsqrt(variance + eps)
        scale = tl.load(weight + columns).to(tl.float32)
        shift = tl.load(bias + columns).to(tl.float32)
        output = centered * reciprocal_std[:, None] * scale + shift

        tl.store(residual + offsets, summed, mask=mask)
        if final_boundary:
            tl.store(normalized + offsets, output.to(tl.float32), mask=mask)
        else:
            tl.store(normalized + offsets, output.to(tl.float16), mask=mask)


def triton_mixed_residual_layer_norm_available() -> bool:
    """Return whether the optional Triton runtime can load this specialization."""

    return triton is not None and tl is not None


def can_use_triton_mixed_residual_layer_norm(
    value: torch.Tensor,
    update: torch.Tensor,
    layer_norm: nn.LayerNorm,
) -> bool:
    """Return whether the exact mixed residual-stream specialization can run."""

    if triton is None or tl is None or not isinstance(layer_norm, nn.LayerNorm):
        return False
    if torch.is_grad_enabled() or value.device.type != "cuda":
        return False
    if value.dtype != torch.float32 or update.dtype != torch.float16:
        return False
    if value.shape != update.shape or value.device != update.device:
        return False
    if value.ndim < 2 or value.shape[-1] != _WIDTH:
        return False
    if not value.is_contiguous() or not update.is_contiguous():
        return False
    if tuple(layer_norm.normalized_shape) != (_WIDTH,):
        return False
    if layer_norm.weight is None or layer_norm.bias is None:
        return False
    return all(
        parameter.device == value.device and parameter.dtype == torch.float32
        for parameter in (layer_norm.weight, layer_norm.bias)
    )


def triton_mixed_residual_layer_norm(
    value: torch.Tensor,
    update: torch.Tensor,
    layer_norm: nn.LayerNorm,
    *,
    final_boundary: bool,
    block_rows: int = _BLOCK_ROWS,
    num_warps: int = _NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Keep residuals in FP32 while feeding intermediate branches with FP16."""

    _validate_launch_config(block_rows, num_warps)
    if not can_use_triton_mixed_residual_layer_norm(value, update, layer_norm):
        raise RuntimeError(
            "Triton mixed residual LayerNorm is ineligible for the requested inputs"
        )
    assert triton is not None
    assert layer_norm.weight is not None
    assert layer_norm.bias is not None
    row_count = value.numel() // _WIDTH
    residual = torch.empty_like(value)
    normalized = torch.empty_like(
        value,
        dtype=torch.float32 if final_boundary else torch.float16,
    )
    try:
        _mixed_residual_layer_norm_kernel[(triton.cdiv(row_count, block_rows),)](
            value,
            update,
            layer_norm.weight,
            layer_norm.bias,
            residual,
            normalized,
            row_count,
            eps=layer_norm.eps,
            width=_WIDTH,
            block_rows=block_rows,
            final_boundary=final_boundary,
            num_warps=num_warps,
        )
    except Exception as exc:
        raise RuntimeError("Triton mixed residual LayerNorm execution failed") from exc
    return residual, normalized, TRITON_MIXED_RESIDUAL_LAYER_NORM_BACKEND


__all__ = [
    "TRITON_MIXED_RESIDUAL_LAYER_NORM_BACKEND",
    "can_use_triton_mixed_residual_layer_norm",
    "triton_mixed_residual_layer_norm",
    "triton_mixed_residual_layer_norm_available",
]
