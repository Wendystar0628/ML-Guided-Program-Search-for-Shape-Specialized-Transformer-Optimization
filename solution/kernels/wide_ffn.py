"""Bounded exact-GELU candidate for the fixed Wide FFN projection."""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    _ATEN_GELU_INPLACE = torch.ops.aten.gelu_.default
except (AttributeError, RuntimeError):
    _ATEN_GELU_INPLACE = None


WIDE_FFN_EXACT_GELU_AVAILABLE = _ATEN_GELU_INPLACE is not None

_WIDE_INPUT_SHAPE = (16, 256, 1024)
_WIDE_WEIGHT_SHAPE = (4096, 1024)
_WIDE_BIAS_SHAPE = (4096,)


def supports_wide_exact_gelu(
    *,
    device_type: str,
    dtype: torch.dtype,
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    bias_shape: tuple[int, ...] | None,
    grad_enabled: bool,
) -> bool:
    """Return whether static shapes can use the bounded Wide FFN path."""

    return bool(
        WIDE_FFN_EXACT_GELU_AVAILABLE
        and device_type == "cuda"
        and dtype == torch.bfloat16
        and input_shape == _WIDE_INPUT_SHAPE
        and weight_shape == _WIDE_WEIGHT_SHAPE
        and bias_shape == _WIDE_BIAS_SHAPE
        and not grad_enabled
    )


def can_use_wide_exact_gelu(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> bool:
    """Return whether tensors match the inference-only Wide BF16 candidate."""

    if bias is None:
        return False
    if not value.is_cuda or not weight.is_cuda or not bias.is_cuda:
        return False
    if value.device != weight.device or value.device != bias.device:
        return False
    if weight.dtype != value.dtype or bias.dtype != value.dtype:
        return False
    if not supports_wide_exact_gelu(
        device_type=value.device.type,
        dtype=value.dtype,
        input_shape=tuple(value.shape),
        weight_shape=tuple(weight.shape),
        bias_shape=tuple(bias.shape),
        grad_enabled=torch.is_grad_enabled(),
    ):
        return False
    return value.is_contiguous() and weight.is_contiguous() and bias.is_contiguous()


def wide_linear_exact_gelu(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Run the Wide projection and reuse its fresh output for exact GELU.

    The matrix multiplication remains on PyTorch's tuned CUDA library path. The
    only changed boundary is that exact GELU writes back into the newly-created
    linear output instead of allocating another 16 x 256 x 4096 BF16 tensor.
    """

    if not can_use_wide_exact_gelu(value, weight, bias):
        raise ValueError("tensors are not supported by the Wide exact-GELU path")
    assert bias is not None
    assert _ATEN_GELU_INPLACE is not None

    flattened = value.view(-1, _WIDE_INPUT_SHAPE[-1])
    hidden = F.linear(flattened, weight, bias)
    _ATEN_GELU_INPLACE(hidden, approximate="none")
    return hidden.view(*value.shape[:-1], _WIDE_WEIGHT_SHAPE[0])


__all__ = [
    "WIDE_FFN_EXACT_GELU_AVAILABLE",
    "can_use_wide_exact_gelu",
    "supports_wide_exact_gelu",
    "wide_linear_exact_gelu",
]
