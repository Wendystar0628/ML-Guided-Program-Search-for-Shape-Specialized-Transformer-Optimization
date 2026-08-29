"""Mask-aware Triton LayerNorm primitives for padded official workloads."""

from __future__ import annotations

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised without the optional runtime.
    triton = None
    tl = None


TRITON_MASKED_LAYER_NORM_BACKEND = "triton_masked_layer_norm"
TRITON_MASKED_RESIDUAL_LAYER_NORM_BACKEND = "triton_masked_residual_layer_norm"
_DEFAULT_BLOCK_ROWS = 2
_DEFAULT_NUM_WARPS = 2
_SUPPORTED_WIDTHS = frozenset({32, 128, 1024})
_SUPPORTED_OUTPUT_DTYPES = frozenset({torch.float16, torch.float32})
_SUPPORTED_UPDATE_DTYPES = frozenset({torch.float16, torch.float32})
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
        raise ValueError("unsupported Triton masked LayerNorm launch configuration")


def _normalize_output_dtype(output_dtype: torch.dtype) -> torch.dtype:
    if output_dtype not in _SUPPORTED_OUTPUT_DTYPES:
        raise ValueError("masked LayerNorm output must be float16 or float32")
    return output_dtype


if triton is not None and tl is not None:

    @triton.jit
    def _masked_layer_norm_kernel(
        value,
        valid_token_mask,
        weight,
        bias,
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
        row_in_bounds = rows < row_count
        valid_rows = tl.load(valid_token_mask + rows, mask=row_in_bounds, other=0) != 0
        active = row_in_bounds & valid_rows

        input_value = tl.load(value + offsets, mask=active, other=0.0).to(tl.float32)
        mean = tl.sum(input_value, axis=1) / width
        centered = input_value - mean[:, None]
        variance = tl.sum(centered * centered, axis=1) / width
        reciprocal_std = tl.rsqrt(variance + eps)
        scale = tl.load(weight + columns).to(tl.float32)
        shift = tl.load(bias + columns).to(tl.float32)
        output = centered * reciprocal_std[:, None] * scale + shift
        output = tl.where(valid_rows, output, 0.0)
        tl.store(normalized + offsets, output, mask=row_in_bounds)

    @triton.jit
    def _masked_residual_layer_norm_kernel(
        value,
        update,
        valid_token_mask,
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
        row_in_bounds = rows < row_count
        valid_rows = tl.load(valid_token_mask + rows, mask=row_in_bounds, other=0) != 0
        active = row_in_bounds & valid_rows

        residual_value = tl.load(value + offsets, mask=active, other=0.0).to(tl.float32)
        branch_update = tl.load(update + offsets, mask=active, other=0.0).to(tl.float32)
        summed = residual_value + branch_update
        mean = tl.sum(summed, axis=1) / width
        centered = summed - mean[:, None]
        variance = tl.sum(centered * centered, axis=1) / width
        reciprocal_std = tl.rsqrt(variance + eps)
        scale = tl.load(weight + columns).to(tl.float32)
        shift = tl.load(bias + columns).to(tl.float32)
        output = centered * reciprocal_std[:, None] * scale + shift

        summed = tl.where(valid_rows, summed, 0.0)
        output = tl.where(valid_rows, output, 0.0)
        tl.store(residual + offsets, summed, mask=row_in_bounds)
        tl.store(normalized + offsets, output, mask=row_in_bounds)


def triton_masked_layer_norm_available() -> bool:
    """Return whether the optional Triton implementation can be loaded."""

    return triton is not None and tl is not None


def can_use_triton_masked_layer_norm(
    value: torch.Tensor,
    valid_token_mask: torch.Tensor,
    layer_norm: nn.LayerNorm,
    *,
    output_dtype: torch.dtype = torch.float32,
) -> bool:
    """Validate a contiguous FP32 ``[B, S, D]`` masked LayerNorm request."""

    if triton is None or tl is None or not isinstance(layer_norm, nn.LayerNorm):
        return False
    if torch.is_grad_enabled() or value.device.type != "cuda":
        return False
    if (
        value.dtype != torch.float32
        or value.ndim != 3
        or value.shape[-1] not in _SUPPORTED_WIDTHS
        or not value.is_contiguous()
    ):
        return False
    if (
        valid_token_mask.device != value.device
        or valid_token_mask.dtype != torch.bool
        or valid_token_mask.shape != value.shape[:2]
        or not valid_token_mask.is_contiguous()
    ):
        return False
    width = value.shape[-1]
    if tuple(layer_norm.normalized_shape) != (width,):
        return False
    if layer_norm.weight is None or layer_norm.bias is None:
        return False
    if not all(
        parameter.device == value.device and parameter.dtype == torch.float32
        for parameter in (layer_norm.weight, layer_norm.bias)
    ):
        return False
    try:
        _normalize_output_dtype(output_dtype)
    except ValueError:
        return False
    return True


def prevalidated_triton_masked_layer_norm(
    value: torch.Tensor,
    valid_token_mask: torch.Tensor,
    layer_norm: nn.LayerNorm,
    *,
    output_dtype: torch.dtype = torch.float32,
    block_rows: int = _DEFAULT_BLOCK_ROWS,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, str]:
    """Execute masked LayerNorm after immutable plan validation."""

    _validate_launch_config(block_rows, num_warps)
    normalized_dtype = _normalize_output_dtype(output_dtype)
    if triton is None or tl is None:
        raise RuntimeError("Triton masked LayerNorm is unavailable")
    assert layer_norm.weight is not None
    assert layer_norm.bias is not None
    width = value.shape[-1]
    row_count = value.numel() // width
    normalized = torch.empty_like(value, dtype=normalized_dtype)
    try:
        _masked_layer_norm_kernel[(triton.cdiv(row_count, block_rows),)](
            value,
            valid_token_mask,
            layer_norm.weight,
            layer_norm.bias,
            normalized,
            row_count,
            eps=layer_norm.eps,
            width=width,
            block_rows=block_rows,
            num_warps=num_warps,
        )
    except Exception as exc:
        raise RuntimeError("Triton masked LayerNorm execution failed") from exc
    return normalized, TRITON_MASKED_LAYER_NORM_BACKEND


def triton_masked_layer_norm(
    value: torch.Tensor,
    valid_token_mask: torch.Tensor,
    layer_norm: nn.LayerNorm,
    *,
    output_dtype: torch.dtype = torch.float32,
    block_rows: int = _DEFAULT_BLOCK_ROWS,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, str]:
    """Normalize valid tokens and zero padded-token outputs in one kernel."""

    _validate_launch_config(block_rows, num_warps)
    if not can_use_triton_masked_layer_norm(
        value,
        valid_token_mask,
        layer_norm,
        output_dtype=output_dtype,
    ):
        raise RuntimeError("Triton masked LayerNorm is ineligible for this request")
    return prevalidated_triton_masked_layer_norm(
        value,
        valid_token_mask,
        layer_norm,
        output_dtype=output_dtype,
        block_rows=block_rows,
        num_warps=num_warps,
    )


def can_use_triton_masked_residual_layer_norm(
    value: torch.Tensor,
    update: torch.Tensor,
    valid_token_mask: torch.Tensor,
    layer_norm: nn.LayerNorm,
    *,
    output_dtype: torch.dtype = torch.float32,
) -> bool:
    """Validate residual add plus masked LayerNorm eligibility."""

    if not can_use_triton_masked_layer_norm(
        value,
        valid_token_mask,
        layer_norm,
        output_dtype=output_dtype,
    ):
        return False
    return bool(
        update.device == value.device
        and update.dtype in _SUPPORTED_UPDATE_DTYPES
        and update.shape == value.shape
        and update.is_contiguous()
    )


def prevalidated_triton_masked_residual_layer_norm(
    value: torch.Tensor,
    update: torch.Tensor,
    valid_token_mask: torch.Tensor,
    layer_norm: nn.LayerNorm,
    *,
    output_dtype: torch.dtype = torch.float32,
    block_rows: int = _DEFAULT_BLOCK_ROWS,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Execute residual add, masked LayerNorm and padded-row zeroing."""

    _validate_launch_config(block_rows, num_warps)
    normalized_dtype = _normalize_output_dtype(output_dtype)
    if triton is None or tl is None:
        raise RuntimeError("Triton masked residual LayerNorm is unavailable")
    assert layer_norm.weight is not None
    assert layer_norm.bias is not None
    width = value.shape[-1]
    row_count = value.numel() // width
    residual = torch.empty_like(value)
    normalized = torch.empty_like(value, dtype=normalized_dtype)
    try:
        _masked_residual_layer_norm_kernel[(triton.cdiv(row_count, block_rows),)](
            value,
            update,
            valid_token_mask,
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
        raise RuntimeError("Triton masked residual LayerNorm execution failed") from exc
    return residual, normalized, TRITON_MASKED_RESIDUAL_LAYER_NORM_BACKEND


def triton_masked_residual_layer_norm(
    value: torch.Tensor,
    update: torch.Tensor,
    valid_token_mask: torch.Tensor,
    layer_norm: nn.LayerNorm,
    *,
    output_dtype: torch.dtype = torch.float32,
    block_rows: int = _DEFAULT_BLOCK_ROWS,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Run the strictly guarded masked residual LayerNorm primitive."""

    _validate_launch_config(block_rows, num_warps)
    if not can_use_triton_masked_residual_layer_norm(
        value,
        update,
        valid_token_mask,
        layer_norm,
        output_dtype=output_dtype,
    ):
        raise RuntimeError(
            "Triton masked residual LayerNorm is ineligible for this request"
        )
    return prevalidated_triton_masked_residual_layer_norm(
        value,
        update,
        valid_token_mask,
        layer_norm,
        output_dtype=output_dtype,
        block_rows=block_rows,
        num_warps=num_warps,
    )


__all__ = [
    "TRITON_MASKED_LAYER_NORM_BACKEND",
    "TRITON_MASKED_RESIDUAL_LAYER_NORM_BACKEND",
    "can_use_triton_masked_layer_norm",
    "can_use_triton_masked_residual_layer_norm",
    "prevalidated_triton_masked_layer_norm",
    "prevalidated_triton_masked_residual_layer_norm",
    "triton_masked_layer_norm",
    "triton_masked_layer_norm_available",
    "triton_masked_residual_layer_norm",
]
