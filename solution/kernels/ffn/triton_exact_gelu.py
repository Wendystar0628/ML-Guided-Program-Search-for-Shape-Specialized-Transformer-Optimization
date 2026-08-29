"""Forward-only contiguous Exact-GELU with an optional output cast."""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised without the optional runtime.
    triton = None
    tl = None


TRITON_EXACT_GELU_BACKEND = "triton_exact_gelu"
_DEFAULT_BLOCK_SIZE = 256
_DEFAULT_NUM_WARPS = 4
_SUPPORTED_DTYPES = frozenset({torch.float16, torch.float32})
_SUPPORTED_BLOCK_SIZES = frozenset({128, 256, 512, 1024})
_SUPPORTED_NUM_WARPS = frozenset({1, 2, 4, 8})


def _validate_launch_config(block_size: int, num_warps: int) -> None:
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size not in _SUPPORTED_BLOCK_SIZES
        or isinstance(num_warps, bool)
        or not isinstance(num_warps, int)
        or num_warps not in _SUPPORTED_NUM_WARPS
    ):
        raise ValueError("unsupported Triton Exact-GELU launch configuration")


def _normalize_output_dtype(
    value: torch.Tensor,
    output_dtype: torch.dtype | None,
) -> torch.dtype:
    normalized = value.dtype if output_dtype is None else output_dtype
    if normalized not in _SUPPORTED_DTYPES:
        raise ValueError("Triton Exact-GELU output must be float16 or float32")
    return normalized


if triton is not None and tl is not None:

    @triton.jit
    def _exact_gelu_kernel(
        value,
        output,
        element_count,
        BLOCK_SIZE: tl.constexpr,
    ) -> None:
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < element_count
        input_value = tl.load(value + offsets, mask=mask, other=0.0).to(tl.float32)
        exact_gelu = (
            0.5 * input_value * (1.0 + tl.erf(input_value * (1.0 / math.sqrt(2.0))))
        )
        tl.store(output + offsets, exact_gelu, mask=mask)


def triton_exact_gelu_available() -> bool:
    """Return whether the optional Triton implementation can be loaded."""

    return triton is not None and tl is not None


def can_use_triton_exact_gelu(
    value: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> bool:
    """Return whether the Exact-GELU specialization accepts this tensor."""

    if triton is None or tl is None or not isinstance(value, torch.Tensor):
        return False
    if torch.is_grad_enabled() or value.device.type != "cuda":
        return False
    if value.dtype not in _SUPPORTED_DTYPES or not value.is_contiguous():
        return False
    if value.numel() == 0:
        return False
    try:
        _normalize_output_dtype(value, output_dtype)
    except ValueError:
        return False
    return True


def prevalidated_triton_exact_gelu(
    value: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, str]:
    """Execute Exact-GELU after the caller has established tensor eligibility."""

    _validate_launch_config(block_size, num_warps)
    normalized_dtype = _normalize_output_dtype(value, output_dtype)
    if triton is None or tl is None:
        raise RuntimeError("Triton Exact-GELU is unavailable")

    output = torch.empty_like(value, dtype=normalized_dtype)
    try:
        _exact_gelu_kernel[(triton.cdiv(value.numel(), block_size),)](
            value,
            output,
            value.numel(),
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    except Exception as exc:
        raise RuntimeError("Triton Exact-GELU execution failed") from exc
    return output, TRITON_EXACT_GELU_BACKEND


def triton_exact_gelu(
    value: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, str]:
    """Run contiguous Exact-GELU and optionally cast its output in one kernel."""

    _validate_launch_config(block_size, num_warps)
    if not can_use_triton_exact_gelu(value, output_dtype=output_dtype):
        raise RuntimeError("Triton Exact-GELU is ineligible for the requested input")
    return prevalidated_triton_exact_gelu(
        value,
        output_dtype=output_dtype,
        block_size=block_size,
        num_warps=num_warps,
    )


__all__ = [
    "TRITON_EXACT_GELU_BACKEND",
    "can_use_triton_exact_gelu",
    "prevalidated_triton_exact_gelu",
    "triton_exact_gelu",
    "triton_exact_gelu_available",
]
