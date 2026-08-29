"""Forward-only fused Linear + bias + exact GELU with an FP16 output."""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised without the optional runtime.
    triton = None
    tl = None


TRITON_LINEAR_EXACT_GELU_BACKEND = "triton_linear_exact_gelu"
_DEFAULT_BLOCK_M = 64
_DEFAULT_BLOCK_N = 64
_DEFAULT_BLOCK_K = 32
_DEFAULT_NUM_WARPS = 4
_DEFAULT_NUM_STAGES = 2
_SUPPORTED_FEATURE_SIZES = frozenset({32, 128, 1024})
_SUPPORTED_BLOCK_SIZES = frozenset({16, 32, 64, 128})
_SUPPORTED_NUM_WARPS = frozenset({1, 2, 4, 8})
_SUPPORTED_NUM_STAGES = frozenset({1, 2, 3, 4})
_SUPPORTED_DTYPES = frozenset({torch.float16, torch.float32})


def _launch_config_is_valid(
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> bool:
    values = (block_m, block_n, block_k, num_warps, num_stages)
    return bool(
        not any(isinstance(value, bool) for value in values)
        and all(isinstance(value, int) for value in values)
        and block_m in _SUPPORTED_BLOCK_SIZES
        and block_n in _SUPPORTED_BLOCK_SIZES
        and block_k in _SUPPORTED_BLOCK_SIZES
        and num_warps in _SUPPORTED_NUM_WARPS
        and num_stages in _SUPPORTED_NUM_STAGES
    )


def _validate_launch_config(
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> None:
    if not _launch_config_is_valid(
        block_m,
        block_n,
        block_k,
        num_warps,
        num_stages,
    ):
        raise ValueError("unsupported Triton Linear + Exact-GELU launch configuration")


if triton is not None and tl is not None:

    @triton.jit
    def _linear_exact_gelu_kernel(
        value,
        weight,
        bias,
        output,
        row_count,
        input_features: tl.constexpr,
        output_features: tl.constexpr,
        stride_vm,
        stride_vk,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        program_m = tl.program_id(0)
        program_n = tl.program_id(1)
        offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)

        value_ptrs = (
            value + offsets_m[:, None] * stride_vm + offsets_k[None, :] * stride_vk
        )
        weight_ptrs = (
            weight + offsets_n[None, :] * stride_wn + offsets_k[:, None] * stride_wk
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for start_k in range(0, input_features, BLOCK_K):
            remaining_k = input_features - start_k
            value_tile = tl.load(
                value_ptrs,
                mask=(offsets_m[:, None] < row_count)
                & (offsets_k[None, :] < remaining_k),
                other=0.0,
            )
            weight_tile = tl.load(
                weight_ptrs,
                mask=(offsets_n[None, :] < output_features)
                & (offsets_k[:, None] < remaining_k),
                other=0.0,
            )
            accumulator = tl.dot(value_tile, weight_tile, accumulator)
            value_ptrs += BLOCK_K * stride_vk
            weight_ptrs += BLOCK_K * stride_wk

        bias_value = tl.load(
            bias + offsets_n,
            mask=offsets_n < output_features,
            other=0.0,
        ).to(tl.float32)
        projected = accumulator + bias_value[None, :]
        exact_gelu = (
            0.5 * projected * (1.0 + tl.erf(projected * (1.0 / math.sqrt(2.0))))
        )
        output_ptrs = (
            output + offsets_m[:, None] * stride_om + offsets_n[None, :] * stride_on
        )
        tl.store(
            output_ptrs,
            exact_gelu,
            mask=(offsets_m[:, None] < row_count)
            & (offsets_n[None, :] < output_features),
        )


def triton_linear_exact_gelu_available() -> bool:
    """Return whether the optional Triton implementation can be loaded."""

    return triton is not None and tl is not None


def can_use_triton_linear_exact_gelu(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> bool:
    """Return whether the specialization accepts these Linear operands."""

    if triton is None or tl is None:
        return False
    if not all(isinstance(item, torch.Tensor) for item in (value, weight, bias)):
        return False
    if torch.is_grad_enabled() or value.device.type != "cuda":
        return False
    if weight.device != value.device or bias.device != value.device:
        return False
    if value.ndim < 2 or weight.ndim != 2 or bias.ndim != 1:
        return False
    if value.numel() == 0 or value.shape[-1] != weight.shape[1]:
        return False
    if weight.shape[0] != bias.shape[0]:
        return False
    if value.shape[-1] not in _SUPPORTED_FEATURE_SIZES:
        return False
    if weight.shape[0] not in _SUPPORTED_FEATURE_SIZES:
        return False
    if value.dtype not in _SUPPORTED_DTYPES:
        return False
    if weight.dtype != value.dtype or bias.dtype != value.dtype:
        return False
    return bool(
        value.is_contiguous() and weight.is_contiguous() and bias.is_contiguous()
    )


def prevalidated_triton_linear_exact_gelu(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
    block_k: int = _DEFAULT_BLOCK_K,
    num_warps: int = _DEFAULT_NUM_WARPS,
    num_stages: int = _DEFAULT_NUM_STAGES,
) -> tuple[torch.Tensor, str]:
    """Execute the fusion after the caller has established tensor eligibility."""

    _validate_launch_config(block_m, block_n, block_k, num_warps, num_stages)
    if triton is None or tl is None:
        raise RuntimeError("Triton Linear + Exact-GELU is unavailable")

    input_features = value.shape[-1]
    output_features = weight.shape[0]
    row_count = value.numel() // input_features
    matrix = value.view(row_count, input_features)
    output_matrix = torch.empty(
        (row_count, output_features),
        device=value.device,
        dtype=torch.float16,
    )
    grid = (triton.cdiv(row_count, block_m), triton.cdiv(output_features, block_n))
    try:
        _linear_exact_gelu_kernel[grid](
            matrix,
            weight,
            bias,
            output_matrix,
            row_count,
            input_features=input_features,
            output_features=output_features,
            stride_vm=matrix.stride(0),
            stride_vk=matrix.stride(1),
            stride_wn=weight.stride(0),
            stride_wk=weight.stride(1),
            stride_om=output_matrix.stride(0),
            stride_on=output_matrix.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    except Exception as exc:
        raise RuntimeError("Triton Linear + Exact-GELU execution failed") from exc
    return (
        output_matrix.view(*value.shape[:-1], output_features),
        TRITON_LINEAR_EXACT_GELU_BACKEND,
    )


def triton_linear_exact_gelu(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
    block_k: int = _DEFAULT_BLOCK_K,
    num_warps: int = _DEFAULT_NUM_WARPS,
    num_stages: int = _DEFAULT_NUM_STAGES,
) -> tuple[torch.Tensor, str]:
    """Run fused Linear + bias + exact GELU and store a contiguous FP16 output."""

    _validate_launch_config(block_m, block_n, block_k, num_warps, num_stages)
    if not can_use_triton_linear_exact_gelu(value, weight, bias):
        raise RuntimeError(
            "Triton Linear + Exact-GELU is ineligible for the requested operands"
        )
    return prevalidated_triton_linear_exact_gelu(
        value,
        weight,
        bias,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )


__all__ = [
    "TRITON_LINEAR_EXACT_GELU_BACKEND",
    "can_use_triton_linear_exact_gelu",
    "prevalidated_triton_linear_exact_gelu",
    "triton_linear_exact_gelu",
    "triton_linear_exact_gelu_available",
]
