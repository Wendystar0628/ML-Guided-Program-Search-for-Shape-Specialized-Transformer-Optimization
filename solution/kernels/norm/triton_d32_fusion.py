"""D32-specialized residual add, LayerNorm, and output-cast fusion."""

from __future__ import annotations

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised without the optional runtime.
    triton = None
    tl = None


TRITON_D32_RESIDUAL_LAYER_NORM_BACKEND = "triton_d32_residual_layer_norm"
_BATCH_SIZE = 64
_SEQUENCE_LENGTH = 128
_WIDTH = 32
_DEFAULT_ROWS_PER_PROGRAM = 4
_DEFAULT_NUM_WARPS = 4
_SUPPORTED_ROWS_PER_PROGRAM = frozenset({4, 8, 16})
_SUPPORTED_NUM_WARPS = frozenset({2, 4, 8})


def _validate_launch_config(rows_per_program: int, num_warps: int) -> None:
    if (
        isinstance(rows_per_program, bool)
        or not isinstance(rows_per_program, int)
        or rows_per_program not in _SUPPORTED_ROWS_PER_PROGRAM
        or isinstance(num_warps, bool)
        or not isinstance(num_warps, int)
        or num_warps not in _SUPPORTED_NUM_WARPS
    ):
        raise ValueError("unsupported Triton D32 fusion launch configuration")


if triton is not None and tl is not None:

    @triton.jit
    def _d32_residual_layer_norm_kernel(
        value,
        update,
        weight,
        bias,
        residual,
        normalized,
        eps: tl.constexpr,
        rows_per_program: tl.constexpr,
        normalized_is_fp16: tl.constexpr,
    ) -> None:
        row_start = tl.program_id(0) * rows_per_program
        rows = row_start + tl.arange(0, rows_per_program)[:, None]
        columns = tl.arange(0, 32)[None, :]
        offsets = rows * 32 + columns

        residual_input = tl.load(value + offsets).to(tl.float32)
        branch_update = tl.load(update + offsets).to(tl.float32)
        summed = residual_input + branch_update

        mean = tl.sum(summed, axis=1) * (1.0 / 32.0)
        centered = summed - mean[:, None]
        variance = tl.sum(centered * centered, axis=1) * (1.0 / 32.0)
        reciprocal_std = tl.rsqrt(variance + eps)
        normalized_value = centered * reciprocal_std[:, None]
        scale = tl.load(weight + columns).to(tl.float32)
        shift = tl.load(bias + columns).to(tl.float32)
        normalized_value = normalized_value * scale + shift

        tl.store(residual + offsets, summed)
        if normalized_is_fp16:
            tl.store(normalized + offsets, normalized_value.to(tl.float16))
        else:
            tl.store(normalized + offsets, normalized_value)


def triton_d32_residual_layer_norm_available() -> bool:
    """Return whether the optional Triton implementation can be loaded."""

    return triton is not None and tl is not None


def can_use_triton_d32_residual_layer_norm(
    value: torch.Tensor,
    update: torch.Tensor,
    layer_norm: nn.LayerNorm,
    *,
    output_dtype: torch.dtype,
) -> bool:
    """Return whether the exact official Shape-07 specialization can run."""

    if triton is None or tl is None or not isinstance(layer_norm, nn.LayerNorm):
        return False
    if torch.is_grad_enabled() or value.device.type != "cuda":
        return False
    if value.dtype != torch.float32 or update.dtype not in {
        torch.float16,
        torch.float32,
    }:
        return False
    if output_dtype not in {torch.float16, torch.float32}:
        return False
    if value.shape != (_BATCH_SIZE, _SEQUENCE_LENGTH, _WIDTH):
        return False
    if update.shape != value.shape or update.device != value.device:
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


def prevalidated_triton_d32_residual_layer_norm(
    value: torch.Tensor,
    update: torch.Tensor,
    layer_norm: nn.LayerNorm,
    *,
    output_dtype: torch.dtype,
    rows_per_program: int = _DEFAULT_ROWS_PER_PROGRAM,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Execute the specialization after the caller established eligibility."""

    _validate_launch_config(rows_per_program, num_warps)
    if triton is None or tl is None:
        raise RuntimeError("Triton D32 residual LayerNorm is unavailable")
    if output_dtype not in {torch.float16, torch.float32}:
        raise ValueError("D32 fusion output must be float16 or float32")
    assert layer_norm.weight is not None
    assert layer_norm.bias is not None

    residual = torch.empty_like(value)
    normalized = torch.empty_like(value, dtype=output_dtype)
    row_count = _BATCH_SIZE * _SEQUENCE_LENGTH
    try:
        _d32_residual_layer_norm_kernel[(triton.cdiv(row_count, rows_per_program),)](
            value,
            update,
            layer_norm.weight,
            layer_norm.bias,
            residual,
            normalized,
            eps=layer_norm.eps,
            rows_per_program=rows_per_program,
            normalized_is_fp16=output_dtype == torch.float16,
            num_warps=num_warps,
        )
    except Exception as exc:
        raise RuntimeError("Triton D32 residual LayerNorm execution failed") from exc
    return residual, normalized, TRITON_D32_RESIDUAL_LAYER_NORM_BACKEND


def triton_d32_residual_layer_norm(
    value: torch.Tensor,
    update: torch.Tensor,
    layer_norm: nn.LayerNorm,
    *,
    output_dtype: torch.dtype,
    rows_per_program: int = _DEFAULT_ROWS_PER_PROGRAM,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Fuse the Shape-07 residual boundary without a silent fallback."""

    _validate_launch_config(rows_per_program, num_warps)
    if not can_use_triton_d32_residual_layer_norm(
        value,
        update,
        layer_norm,
        output_dtype=output_dtype,
    ):
        raise RuntimeError(
            "Triton D32 residual LayerNorm is ineligible for the requested inputs"
        )
    return prevalidated_triton_d32_residual_layer_norm(
        value,
        update,
        layer_norm,
        output_dtype=output_dtype,
        rows_per_program=rows_per_program,
        num_warps=num_warps,
    )


__all__ = [
    "TRITON_D32_RESIDUAL_LAYER_NORM_BACKEND",
    "can_use_triton_d32_residual_layer_norm",
    "prevalidated_triton_d32_residual_layer_norm",
    "triton_d32_residual_layer_norm",
    "triton_d32_residual_layer_norm_available",
]
