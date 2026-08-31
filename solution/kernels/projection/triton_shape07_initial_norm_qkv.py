"""Shape-07 initial LayerNorm and packed QKV projection fusion."""

from __future__ import annotations

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised without the optional runtime.
    triton = None
    tl = None


TRITON_INITIAL_NORM_QKV_NATIVE_BHSD_BACKEND = (
    "triton_initial_norm_qkv_native_bhsd"
)
_BATCH_SIZE = 64
_SEQUENCE_LENGTH = 128
_WIDTH = 32
_NUM_HEADS = 4
_HEAD_DIM = 8
_PACKED_WIDTH = 3 * _WIDTH
_ROW_COUNT = _BATCH_SIZE * _SEQUENCE_LENGTH
_DEFAULT_BLOCK_M = 16
_BLOCK_N = 128
_BLOCK_K = 32
_DEFAULT_NUM_WARPS = 4
_SUPPORTED_BLOCK_M = frozenset({16, 32, 64})
_SUPPORTED_NUM_WARPS = frozenset({4, 8})


def _launch_config_is_valid(
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
) -> bool:
    return bool(
        not isinstance(block_m, bool)
        and isinstance(block_m, int)
        and block_m in _SUPPORTED_BLOCK_M
        and block_n == _BLOCK_N
        and block_k == _BLOCK_K
        and not isinstance(num_warps, bool)
        and isinstance(num_warps, int)
        and num_warps in _SUPPORTED_NUM_WARPS
    )


def _validate_launch_config(
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
) -> None:
    if not _launch_config_is_valid(block_m, block_n, block_k, num_warps):
        raise ValueError(
            "unsupported Triton Shape-07 initial-norm QKV launch configuration"
        )


if triton is not None and tl is not None:

    @triton.jit
    def _shape07_initial_norm_qkv_bhsd_kernel(
        value,
        norm_weight,
        norm_bias,
        qkv_weight,
        qkv_bias,
        query,
        key,
        projected_value,
        EPS: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ) -> None:
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        input_columns = tl.arange(0, 32)
        input_offsets = rows[:, None] * 32 + input_columns[None, :]

        input_fp32 = tl.load(value + input_offsets).to(tl.float32)
        mean = tl.sum(input_fp32, axis=1) * (1.0 / 32.0)
        centered = input_fp32 - mean[:, None]
        variance = tl.sum(centered * centered, axis=1) * (1.0 / 32.0)
        reciprocal_std = tl.rsqrt(variance + EPS)
        scale = tl.load(norm_weight + input_columns).to(tl.float32)
        shift = tl.load(norm_bias + input_columns).to(tl.float32)
        normalized_fp32 = centered * reciprocal_std[:, None] * scale + shift

        # This cast is part of the primitive contract: the projection observes
        # exactly the FP16 stream produced by the FP32 LayerNorm boundary.
        normalized_fp16 = normalized_fp32.to(tl.float16)
        output_columns = tl.arange(0, 128)
        projection_weight = tl.load(
            qkv_weight
            + input_columns[:, None]
            + output_columns[None, :] * 32,
            mask=output_columns[None, :] < 96,
            other=0.0,
        )
        accumulator = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
        accumulator = tl.dot(normalized_fp16, projection_weight, accumulator)
        projected = accumulator + tl.load(
            qkv_bias + output_columns[None, :],
            mask=output_columns[None, :] < 96,
            other=0.0,
        )

        model_columns = output_columns % 32
        batch_offsets = rows // 128
        sequence_offsets = rows % 128
        head_offsets = model_columns // 8
        head_dim_offsets = model_columns % 8
        output_offsets = (
            batch_offsets[:, None]
            * (4 * 128 * 8)
            + head_offsets[None, :] * (128 * 8)
            + sequence_offsets[:, None] * 8
            + head_dim_offsets[None, :]
        )
        tl.store(
            query + output_offsets,
            projected,
            mask=output_columns[None, :] < 32,
        )
        tl.store(
            key + output_offsets,
            projected,
            mask=(32 <= output_columns[None, :])
            & (output_columns[None, :] < 64),
        )
        tl.store(
            projected_value + output_offsets,
            projected,
            mask=(64 <= output_columns[None, :])
            & (output_columns[None, :] < 96),
        )


def triton_initial_norm_qkv_native_bhsd_available() -> bool:
    """Return whether the optional Triton implementation can be loaded."""

    return triton is not None and tl is not None


def can_use_triton_initial_norm_qkv_native_bhsd(
    value: torch.Tensor,
    layer_norm: nn.LayerNorm,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    *,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _BLOCK_N,
    block_k: int = _BLOCK_K,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> bool:
    """Validate the exact inference-only Shape-07 fusion contract."""

    if not triton_initial_norm_qkv_native_bhsd_available():
        return False
    if not _launch_config_is_valid(block_m, block_n, block_k, num_warps):
        return False
    if not isinstance(value, torch.Tensor):
        return False
    if not isinstance(layer_norm, nn.LayerNorm):
        return False
    if not isinstance(qkv_weight, torch.Tensor) or not isinstance(
        qkv_bias, torch.Tensor
    ):
        return False
    if torch.is_grad_enabled() or value.device.type != "cuda":
        return False
    if value.shape != (_BATCH_SIZE, _SEQUENCE_LENGTH, _WIDTH):
        return False
    if value.dtype != torch.float32 or not value.is_contiguous():
        return False
    if tuple(layer_norm.normalized_shape) != (_WIDTH,):
        return False
    if layer_norm.weight is None or layer_norm.bias is None:
        return False
    if qkv_weight.shape != (_PACKED_WIDTH, _WIDTH):
        return False
    if qkv_bias.shape != (_PACKED_WIDTH,):
        return False
    tensors = (layer_norm.weight, layer_norm.bias, qkv_weight, qkv_bias)
    if any(tensor.device != value.device for tensor in tensors):
        return False
    if layer_norm.weight.dtype != torch.float32:
        return False
    if layer_norm.bias.dtype != torch.float32:
        return False
    if qkv_weight.dtype != torch.float16 or qkv_bias.dtype != torch.float16:
        return False
    if not all(tensor.is_contiguous() for tensor in tensors):
        return False
    return not (
        value.requires_grad or qkv_weight.requires_grad or qkv_bias.requires_grad
    )


def prevalidated_triton_initial_norm_qkv_native_bhsd(
    value: torch.Tensor,
    layer_norm: nn.LayerNorm,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    *,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _BLOCK_N,
    block_k: int = _BLOCK_K,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Run the fusion after the caller has established tensor eligibility."""

    _validate_launch_config(block_m, block_n, block_k, num_warps)
    if triton is None or tl is None:
        raise RuntimeError("Triton Shape-07 initial-norm QKV is unavailable")
    assert layer_norm.weight is not None
    assert layer_norm.bias is not None

    output_shape = (_BATCH_SIZE, _NUM_HEADS, _SEQUENCE_LENGTH, _HEAD_DIM)
    query = torch.empty(output_shape, dtype=torch.float16, device=value.device)
    key = torch.empty_like(query)
    projected_value = torch.empty_like(query)
    grid = (triton.cdiv(_ROW_COUNT, block_m),)
    try:
        _shape07_initial_norm_qkv_bhsd_kernel[grid](
            value,
            layer_norm.weight,
            layer_norm.bias,
            qkv_weight,
            qkv_bias,
            query,
            key,
            projected_value,
            EPS=layer_norm.eps,
            BLOCK_M=block_m,
            num_warps=num_warps,
            num_stages=2,
        )
    except Exception as exc:
        raise RuntimeError(
            "Triton Shape-07 initial-norm QKV execution failed"
        ) from exc
    return (
        query,
        key,
        projected_value,
        TRITON_INITIAL_NORM_QKV_NATIVE_BHSD_BACKEND,
    )


def triton_initial_norm_qkv_native_bhsd(
    value: torch.Tensor,
    layer_norm: nn.LayerNorm,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    *,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _BLOCK_N,
    block_k: int = _BLOCK_K,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Fuse initial LayerNorm and packed QKV without a fallback path."""

    _validate_launch_config(block_m, block_n, block_k, num_warps)
    if not can_use_triton_initial_norm_qkv_native_bhsd(
        value,
        layer_norm,
        qkv_weight,
        qkv_bias,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
    ):
        raise RuntimeError(
            "Triton Shape-07 initial-norm QKV is ineligible for the requested inputs"
        )
    return prevalidated_triton_initial_norm_qkv_native_bhsd(
        value,
        layer_norm,
        qkv_weight,
        qkv_bias,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
    )


__all__ = [
    "TRITON_INITIAL_NORM_QKV_NATIVE_BHSD_BACKEND",
    "can_use_triton_initial_norm_qkv_native_bhsd",
    "prevalidated_triton_initial_norm_qkv_native_bhsd",
    "triton_initial_norm_qkv_native_bhsd",
    "triton_initial_norm_qkv_native_bhsd_available",
]
