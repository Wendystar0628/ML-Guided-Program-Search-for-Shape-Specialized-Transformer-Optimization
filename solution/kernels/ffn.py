"""Shape-independent exact-GELU FFN primitives."""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    _ATEN_GELU_INPLACE = torch.ops.aten.gelu_.default
except (AttributeError, RuntimeError):
    _ATEN_GELU_INPLACE = None


def supports_inplace_exact_gelu() -> bool:
    """Return whether this PyTorch runtime exposes exact in-place GELU."""

    return _ATEN_GELU_INPLACE is not None


def can_use_linear_exact_gelu(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> bool:
    """Return whether linear plus in-place exact GELU is safe for inference."""

    return bool(
        supports_inplace_exact_gelu()
        and not torch.is_grad_enabled()
        and value.ndim >= 2
        and weight.ndim == 2
        and value.shape[-1] == weight.shape[-1]
        and value.device == weight.device
        and value.dtype == weight.dtype
        and (
            bias is None
            or (
                bias.ndim == 1
                and bias.shape[0] == weight.shape[0]
                and bias.device == value.device
                and bias.dtype == value.dtype
            )
        )
    )


def linear_exact_gelu(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, str]:
    """Run exact GELU and return both its output and actual backend.

    GEMM selection stays in PyTorch/cuBLAS and no workload shape is embedded in
    this helper. The in-place operation is restricted to inference; ordinary
    exact GELU remains the correctness-preserving fallback. Returning the
    backend keeps execution evidence tied to the branch that actually ran.
    """

    hidden = F.linear(value, weight, bias)
    if can_use_linear_exact_gelu(value, weight, bias):
        assert _ATEN_GELU_INPLACE is not None
        _ATEN_GELU_INPLACE(hidden, approximate="none")
        return hidden, "inplace_exact_gelu"
    return F.gelu(hidden, approximate="none"), "torch"


__all__ = [
    "can_use_linear_exact_gelu",
    "linear_exact_gelu",
    "supports_inplace_exact_gelu",
]
