"""Forward-only causal FP16 attention for the exact Shape 11 Dh8 family."""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton
except ImportError:  # pragma: no cover - exercised without the optional runtime.
    triton = None
    tl = None
    triton_op = None
    wrap_triton = None


TRITON_DH8_CAUSAL_ATTENTION_BSD_BACKEND = "triton_dh8_causal_attention_bsd"
_BATCH_SIZE = 64
_NUM_HEADS = 16
_SEQUENCE_LENGTH = 128
_HEAD_DIM = 8
_PADDED_HEAD_DIM = 16
_BLOCK_M = 32
_BLOCK_N = 32
_NUM_WARPS = 4
_NUM_STAGES = 2
_SUPPORTED_BLOCK_SIZES = frozenset({16, 32, 64, 128})
_SUPPORTED_NUM_WARPS = frozenset({2, 4, 8})
_SUPPORTED_NUM_STAGES = frozenset({1, 2, 3, 4})
_MIN_COMPUTE_CAPABILITY = (8, 0)


def _launch_config_is_valid(
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> bool:
    values = (block_m, block_n, num_warps, num_stages)
    return bool(
        not any(isinstance(value, bool) for value in values)
        and all(isinstance(value, int) for value in values)
        and block_m in _SUPPORTED_BLOCK_SIZES
        and block_n in _SUPPORTED_BLOCK_SIZES
        and num_warps in _SUPPORTED_NUM_WARPS
        and num_stages in _SUPPORTED_NUM_STAGES
        and _SEQUENCE_LENGTH % block_m == 0
        and _SEQUENCE_LENGTH % block_n == 0
        and block_m % block_n == 0
    )


def _validate_launch_config(
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> None:
    if not _launch_config_is_valid(block_m, block_n, num_warps, num_stages):
        raise ValueError(
            "invalid Dh8 Triton launch configuration; block sizes must be "
            "supported divisors with block_m divisible by block_n, and warps/stages "
            "must use supported values"
        )


if triton is not None and tl is not None:

    @triton.jit
    def _dh8_causal_attention_kernel(
        query,
        key,
        value,
        output,
        qk_scale,
        stride_qb,
        stride_qh,
        stride_qs,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vs,
        stride_vd,
        stride_ob,
        stride_os,
        stride_od,
        NUM_HEADS: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        PADDED_HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ) -> None:
        tl.static_assert(SEQ_LEN % BLOCK_M == 0)
        tl.static_assert(SEQ_LEN % BLOCK_N == 0)
        tl.static_assert(BLOCK_M % BLOCK_N == 0)

        start_m = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch_index = batch_head // NUM_HEADS
        head_index = batch_head % NUM_HEADS

        offsets_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = tl.arange(0, BLOCK_N)
        offsets_d = tl.arange(0, PADDED_HEAD_DIM)
        dim_mask = offsets_d < HEAD_DIM

        query_base = (
            batch_index.to(tl.int64) * stride_qb
            + head_index.to(tl.int64) * stride_qh
        )
        key_base = (
            batch_index.to(tl.int64) * stride_kb
            + head_index.to(tl.int64) * stride_kh
        )
        value_base = (
            batch_index.to(tl.int64) * stride_vb
            + head_index.to(tl.int64) * stride_vh
        )

        query_offsets = (
            query_base
            + offsets_m[:, None] * stride_qs
            + offsets_d[None, :] * stride_qd
        )
        query_tile = tl.load(
            query + query_offsets,
            mask=dim_mask[None, :],
            other=0.0,
        )

        row_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
        row_sum = tl.full([BLOCK_M], 1.0, tl.float32)
        accumulator = tl.zeros([BLOCK_M, PADDED_HEAD_DIM], tl.float32)
        qk_scale = qk_scale.to(tl.float32) * 1.4426950408889634
        high = (start_m + 1) * BLOCK_M

        for start_n in range(0, high, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            key_rows = start_n + offsets_n
            key_offsets = (
                key_base
                + offsets_d[:, None] * stride_kd
                + key_rows[None, :] * stride_ks
            )
            key_tile = tl.load(
                key + key_offsets,
                mask=dim_mask[:, None],
                other=0.0,
            )
            scores = tl.dot(query_tile, key_tile) * qk_scale
            causal_mask = offsets_m[:, None] >= key_rows[None, :]
            scores = tl.where(causal_mask, scores, -float("inf"))
            next_max = tl.maximum(row_max, tl.max(scores, axis=1))
            probabilities = tl.math.exp2(scores - next_max[:, None])
            correction = tl.math.exp2(row_max - next_max)
            accumulator *= correction[:, None]

            value_offsets = (
                value_base
                + key_rows[:, None] * stride_vs
                + offsets_d[None, :] * stride_vd
            )
            value_tile = tl.load(
                value + value_offsets,
                mask=dim_mask[None, :],
                other=0.0,
            )
            accumulator = tl.dot(
                probabilities.to(tl.float16),
                value_tile,
                accumulator,
            )
            row_sum = row_sum * correction + tl.sum(probabilities, axis=1)
            row_max = next_max

        accumulator /= row_sum[:, None]
        output_offsets = (
            batch_index.to(tl.int64) * stride_ob
            + offsets_m[:, None] * stride_os
            + (head_index * HEAD_DIM + offsets_d[None, :]) * stride_od
        )
        tl.store(
            output + output_offsets,
            accumulator.to(tl.float16),
            mask=dim_mask[None, :],
        )


if (
    triton is not None
    and tl is not None
    and triton_op is not None
    and wrap_triton is not None
):

    @triton_op(
        "shape_aware_transformer::dh8_causal_attention_bsd",
        mutates_args={},
    )
    def _dh8_causal_attention_bsd_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        block_m: int,
        block_n: int,
        num_warps: int,
        num_stages: int,
    ) -> torch.Tensor:
        output = torch.empty(
            (_BATCH_SIZE, _SEQUENCE_LENGTH, _NUM_HEADS * _HEAD_DIM),
            dtype=query.dtype,
            device=query.device,
        )
        grid = (
            _SEQUENCE_LENGTH // block_m,
            _BATCH_SIZE * _NUM_HEADS,
        )
        wrap_triton(_dh8_causal_attention_kernel)[grid](
            query,
            key,
            value,
            output,
            scale,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *output.stride(),
            NUM_HEADS=_NUM_HEADS,
            SEQ_LEN=_SEQUENCE_LENGTH,
            HEAD_DIM=_HEAD_DIM,
            PADDED_HEAD_DIM=_PADDED_HEAD_DIM,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return output

else:

    def _dh8_causal_attention_bsd_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        block_m: int,
        block_n: int,
        num_warps: int,
        num_stages: int,
    ) -> torch.Tensor:
        del query, key, value, scale, block_m, block_n, num_warps, num_stages
        raise RuntimeError("Triton Dh8 attention is unavailable")


def triton_dh8_causal_attention_available() -> bool:
    """Return whether the optional Triton custom-op stack can be loaded."""

    return all(
        dependency is not None for dependency in (triton, tl, triton_op, wrap_triton)
    )


def can_use_triton_dh8_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_token_mask: torch.Tensor | None = None,
    *,
    causal: bool = True,
    training: bool = False,
    block_m: int = _BLOCK_M,
    block_n: int = _BLOCK_N,
    num_warps: int = _NUM_WARPS,
    num_stages: int = _NUM_STAGES,
) -> bool:
    """Validate the exact Shape 11 tensor contract."""

    if not triton_dh8_causal_attention_available():
        return False
    if not _launch_config_is_valid(block_m, block_n, num_warps, num_stages):
        return False
    if training or torch.is_grad_enabled() or not causal:
        return False
    if valid_token_mask is not None or query.device.type != "cuda":
        return False
    if query.ndim != 4 or tuple(query.shape) != (
        _BATCH_SIZE,
        _NUM_HEADS,
        _SEQUENCE_LENGTH,
        _HEAD_DIM,
    ):
        return False
    if query.shape != key.shape or query.shape != value.shape:
        return False
    if query.dtype != torch.float16:
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if query.device != key.device or query.device != value.device:
        return False
    if query.requires_grad or key.requires_grad or value.requires_grad:
        return False
    if any(tensor.stride(-1) != 1 for tensor in (query, key, value)):
        return False
    return torch.cuda.get_device_capability(query.device) >= _MIN_COMPUTE_CAPABILITY


def prevalidated_triton_dh8_causal_attention_bsd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
    block_m: int = _BLOCK_M,
    block_n: int = _BLOCK_N,
    num_warps: int = _NUM_WARPS,
    num_stages: int = _NUM_STAGES,
) -> tuple[torch.Tensor, str]:
    """Run the exact specialization after immutable plan validation."""

    _validate_launch_config(block_m, block_n, num_warps, num_stages)
    resolved_scale = 1.0 / math.sqrt(_HEAD_DIM) if scale is None else float(scale)
    try:
        output = _dh8_causal_attention_bsd_op(
            query,
            key,
            value,
            resolved_scale,
            block_m,
            block_n,
            num_warps,
            num_stages,
        )
    except Exception as exc:
        raise RuntimeError("Triton Dh8 attention execution failed") from exc
    return output, TRITON_DH8_CAUSAL_ATTENTION_BSD_BACKEND


__all__ = [
    "TRITON_DH8_CAUSAL_ATTENTION_BSD_BACKEND",
    "can_use_triton_dh8_causal_attention",
    "prevalidated_triton_dh8_causal_attention_bsd",
    "triton_dh8_causal_attention_available",
]
