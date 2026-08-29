"""Forward-only FP16 QKV projection with direct contiguous BHSD outputs."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised without the optional runtime.
    triton = None
    tl = None


TRITON_QKV_NATIVE_BHSD_BACKEND = "triton_qkv_native_bhsd"
_DEFAULT_BLOCK_M = 64
_DEFAULT_BLOCK_N = 32
_DEFAULT_BLOCK_K = 32
_DEFAULT_NUM_WARPS = 4
_SUPPORTED_DIMENSIONS = frozenset({32, 128, 1024})
_SUPPORTED_HEADS = frozenset({1, 2, 4, 16})
_SUPPORTED_SEQUENCE_LENGTHS = frozenset({32, 128, 1024})
_SUPPORTED_BLOCK_M = frozenset({16, 32, 64, 128})
_SUPPORTED_BLOCK_N = frozenset({16, 32, 64, 128})
_SUPPORTED_BLOCK_K = frozenset({16, 32, 64})
_SUPPORTED_NUM_WARPS = frozenset({2, 4, 8})
_MIN_COMPUTE_CAPABILITY = (8, 0)


def _launch_config_is_valid(
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
) -> bool:
    values = (block_m, block_n, block_k, num_warps)
    return bool(
        not any(isinstance(value, bool) for value in values)
        and all(isinstance(value, int) for value in values)
        and block_m in _SUPPORTED_BLOCK_M
        and block_n in _SUPPORTED_BLOCK_N
        and block_k in _SUPPORTED_BLOCK_K
        and num_warps in _SUPPORTED_NUM_WARPS
    )


def _validate_launch_config(
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
) -> None:
    if not _launch_config_is_valid(block_m, block_n, block_k, num_warps):
        raise ValueError("unsupported Triton QKV-native-layout launch configuration")


if triton is not None and tl is not None:

    @triton.jit
    def _qkv_native_bhsd_kernel(
        hidden_states,
        weight,
        bias,
        query,
        key,
        value,
        row_count,
        DIMENSION: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        SEQUENCE_LENGTH: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        column_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        qkv_index = tl.program_id(2)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start_k in range(0, DIMENSION, BLOCK_K):
            k_offsets = start_k + tl.arange(0, BLOCK_K)
            input_tile = tl.load(
                hidden_states + row_offsets[:, None] * DIMENSION + k_offsets[None, :],
                mask=(row_offsets[:, None] < row_count)
                & (k_offsets[None, :] < DIMENSION),
                other=0.0,
            )
            weight_rows = qkv_index * DIMENSION + column_offsets
            weight_tile = tl.load(
                weight + weight_rows[None, :] * DIMENSION + k_offsets[:, None],
                mask=(column_offsets[None, :] < DIMENSION)
                & (k_offsets[:, None] < DIMENSION),
                other=0.0,
            )
            accumulator = tl.dot(input_tile, weight_tile, accumulator)

        result = accumulator + tl.load(
            bias + qkv_index * DIMENSION + column_offsets[None, :],
            mask=column_offsets[None, :] < DIMENSION,
            other=0.0,
        )
        batch_offsets = row_offsets // SEQUENCE_LENGTH
        sequence_offsets = row_offsets % SEQUENCE_LENGTH
        head_offsets = column_offsets // HEAD_DIM
        head_dim_offsets = column_offsets % HEAD_DIM
        output_offsets = (
            batch_offsets[:, None] * (NUM_HEADS * SEQUENCE_LENGTH * HEAD_DIM)
            + head_offsets[None, :] * (SEQUENCE_LENGTH * HEAD_DIM)
            + sequence_offsets[:, None] * HEAD_DIM
            + head_dim_offsets[None, :]
        )
        output_mask = (row_offsets[:, None] < row_count) & (
            column_offsets[None, :] < DIMENSION
        )
        tl.store(query + output_offsets, result, mask=output_mask & (qkv_index == 0))
        tl.store(key + output_offsets, result, mask=output_mask & (qkv_index == 1))
        tl.store(value + output_offsets, result, mask=output_mask & (qkv_index == 2))


def triton_qkv_native_bhsd_available() -> bool:
    """Return whether the optional Triton projection implementation is available."""

    return triton is not None and tl is not None


def can_use_triton_qkv_native_bhsd(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    num_heads: int,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
    block_k: int = _DEFAULT_BLOCK_K,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> bool:
    """Validate the fixed-shape FP16 inference contract for this primitive."""

    if not triton_qkv_native_bhsd_available():
        return False
    if not _launch_config_is_valid(block_m, block_n, block_k, num_warps):
        return False
    if not isinstance(hidden_states, torch.Tensor):
        return False
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        return False
    if torch.is_grad_enabled() or hidden_states.device.type != "cuda":
        return False
    if hidden_states.ndim != 3 or hidden_states.dtype != torch.float16:
        return False
    batch_size, sequence_length, dimension = hidden_states.shape
    if batch_size <= 0 or sequence_length not in _SUPPORTED_SEQUENCE_LENGTHS:
        return False
    if dimension not in _SUPPORTED_DIMENSIONS or num_heads not in _SUPPORTED_HEADS:
        return False
    if dimension % num_heads != 0:
        return False
    if weight.shape != (3 * dimension, dimension) or bias.shape != (3 * dimension,):
        return False
    if weight.dtype != hidden_states.dtype or bias.dtype != hidden_states.dtype:
        return False
    if weight.device != hidden_states.device or bias.device != hidden_states.device:
        return False
    if any(tensor.requires_grad for tensor in (hidden_states, weight, bias)):
        return False
    if not all(tensor.is_contiguous() for tensor in (hidden_states, weight, bias)):
        return False
    return torch.cuda.get_device_capability(hidden_states.device) >= (
        _MIN_COMPUTE_CAPABILITY
    )


def prevalidated_triton_qkv_native_bhsd(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    num_heads: int,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
    block_k: int = _DEFAULT_BLOCK_K,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Execute projection after the caller has established tensor eligibility."""

    _validate_launch_config(block_m, block_n, block_k, num_warps)
    if triton is None or tl is None:
        raise RuntimeError("Triton QKV native-layout projection is unavailable")

    batch_size, sequence_length, dimension = hidden_states.shape
    head_dim = dimension // num_heads
    output_shape = (batch_size, num_heads, sequence_length, head_dim)
    query = torch.empty(
        output_shape, dtype=hidden_states.dtype, device=hidden_states.device
    )
    key = torch.empty_like(query)
    value = torch.empty_like(query)
    row_count = batch_size * sequence_length
    grid = (
        triton.cdiv(row_count, block_m),
        triton.cdiv(dimension, block_n),
        3,
    )
    try:
        _qkv_native_bhsd_kernel[grid](
            hidden_states,
            weight,
            bias,
            query,
            key,
            value,
            row_count,
            DIMENSION=dimension,
            NUM_HEADS=num_heads,
            SEQUENCE_LENGTH=sequence_length,
            HEAD_DIM=head_dim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=2,
        )
    except Exception as exc:
        raise RuntimeError(
            "Triton QKV native-layout projection execution failed"
        ) from exc
    return query, key, value, TRITON_QKV_NATIVE_BHSD_BACKEND


def triton_qkv_native_bhsd(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    num_heads: int,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
    block_k: int = _DEFAULT_BLOCK_K,
    num_warps: int = _DEFAULT_NUM_WARPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Project QKV and directly materialize three contiguous BHSD tensors."""

    _validate_launch_config(block_m, block_n, block_k, num_warps)
    if not can_use_triton_qkv_native_bhsd(
        hidden_states,
        weight,
        bias,
        num_heads=num_heads,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
    ):
        raise RuntimeError("Triton QKV native-layout projection is ineligible")
    return prevalidated_triton_qkv_native_bhsd(
        hidden_states,
        weight,
        bias,
        num_heads=num_heads,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
    )


__all__ = [
    "TRITON_QKV_NATIVE_BHSD_BACKEND",
    "can_use_triton_qkv_native_bhsd",
    "prevalidated_triton_qkv_native_bhsd",
    "triton_qkv_native_bhsd",
    "triton_qkv_native_bhsd_available",
]
