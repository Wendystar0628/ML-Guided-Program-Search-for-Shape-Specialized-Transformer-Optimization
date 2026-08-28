"""Forward-only causal FP16 attention for the exact resident Shape 13."""

from __future__ import annotations

import math

import torch

from ..shape_families import is_shape13_triton_attention_tensor_family

try:
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton
except ImportError:  # pragma: no cover - exercised without the optional runtime.
    triton = None
    tl = None
    triton_op = None
    wrap_triton = None


TRITON_SHAPE13_CAUSAL_ATTENTION_BACKEND = "triton_shape13_causal_attention"
_BATCH_SIZE = 64
_NUM_HEADS = 4
_SEQUENCE_LENGTH = 1024
_HEAD_DIM = 32
_BLOCK_M = 64
_BLOCK_N = 64
_NUM_WARPS = 4
_NUM_STAGES = 2
_MIN_COMPUTE_CAPABILITY = (8, 0)


if triton is not None and tl is not None:

    @triton.jit
    def _attention_inner(
        accumulator,
        row_sum,
        row_max,
        query,
        key_block,
        value_block,
        start_m,
        qk_scale,
        BLOCK_M: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
        STAGE: tl.constexpr,
        offsets_m: tl.constexpr,
        offsets_n: tl.constexpr,
    ):
        if STAGE == 1:
            low, high = 0, start_m * BLOCK_M
        else:
            low, high = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
            low = tl.multiple_of(low, BLOCK_M)

        key_block = tl.advance(key_block, (0, low))
        value_block = tl.advance(value_block, (low, 0))
        for start_n in range(low, high, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            key = tl.load(key_block)
            scores = tl.dot(query, key)
            if STAGE == 2:
                causal_mask = offsets_m[:, None] >= (start_n + offsets_n[None, :])
                scaled_scores = scores * qk_scale
                scores = tl.where(
                    causal_mask,
                    scaled_scores,
                    -float("inf"),
                )
                next_max = tl.maximum(row_max, tl.max(scores, axis=1))
                scores -= next_max[:, None]
            else:
                next_max = tl.maximum(
                    row_max,
                    tl.max(scores, axis=1) * qk_scale,
                )
                scores = scores * qk_scale - next_max[:, None]

            probabilities = tl.math.exp2(scores)
            correction = tl.math.exp2(row_max - next_max)
            accumulator *= correction[:, None]
            value = tl.load(value_block)
            accumulator = tl.dot(
                probabilities.to(tl.float16),
                value,
                accumulator,
            )
            row_sum = row_sum * correction + tl.sum(probabilities, axis=1)
            row_max = next_max
            key_block = tl.advance(key_block, (0, BLOCK_N))
            value_block = tl.advance(value_block, (BLOCK_N, 0))

        return accumulator, row_sum, row_max

    @triton.jit
    def _shape13_causal_attention_kernel(
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
        stride_oh,
        stride_os,
        stride_od,
        NUM_HEADS: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ) -> None:
        tl.static_assert(NUM_HEADS == 4)
        tl.static_assert(SEQ_LEN == 1024)
        tl.static_assert(HEAD_DIM == 32)
        tl.static_assert(BLOCK_M == 64)
        tl.static_assert(BLOCK_N == 64)

        start_m = tl.program_id(axis=0)
        batch_head = tl.program_id(axis=1)
        batch_index = batch_head // NUM_HEADS
        head_index = batch_head % NUM_HEADS

        query_base = (
            batch_index.to(tl.int64) * stride_qb + head_index.to(tl.int64) * stride_qh
        )
        key_base = (
            batch_index.to(tl.int64) * stride_kb + head_index.to(tl.int64) * stride_kh
        )
        value_base = (
            batch_index.to(tl.int64) * stride_vb + head_index.to(tl.int64) * stride_vh
        )
        output_base = (
            batch_index.to(tl.int64) * stride_ob + head_index.to(tl.int64) * stride_oh
        )

        query_block = tl.make_block_ptr(
            base=query + query_base,
            shape=(SEQ_LEN, HEAD_DIM),
            strides=(stride_qs, stride_qd),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        key_block = tl.make_block_ptr(
            base=key + key_base,
            shape=(HEAD_DIM, SEQ_LEN),
            strides=(stride_kd, stride_ks),
            offsets=(0, 0),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(0, 1),
        )
        value_block = tl.make_block_ptr(
            base=value + value_base,
            shape=(SEQ_LEN, HEAD_DIM),
            strides=(stride_vs, stride_vd),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        output_block = tl.make_block_ptr(
            base=output + output_base,
            shape=(SEQ_LEN, HEAD_DIM),
            strides=(stride_os, stride_od),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        offsets_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = tl.arange(0, BLOCK_N)
        row_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
        row_sum = tl.full([BLOCK_M], 1.0, tl.float32)
        accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
        query_tile = tl.load(query_block)
        # Inductor represents Python scalar arguments as fp64. Keep the
        # online-softmax state in fp32 so the accumulator remains compatible
        # with Triton's fp32 dot output.
        qk_scale = qk_scale.to(tl.float32) * 1.4426950408889634

        accumulator, row_sum, row_max = _attention_inner(
            accumulator,
            row_sum,
            row_max,
            query_tile,
            key_block,
            value_block,
            start_m,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            1,
            offsets_m,
            offsets_n,
        )
        accumulator, row_sum, row_max = _attention_inner(
            accumulator,
            row_sum,
            row_max,
            query_tile,
            key_block,
            value_block,
            start_m,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            2,
            offsets_m,
            offsets_n,
        )
        accumulator /= row_sum[:, None]
        tl.store(output_block, accumulator.to(tl.float16))


if (
    triton is not None
    and tl is not None
    and triton_op is not None
    and wrap_triton is not None
):

    @triton_op(
        "shape_aware_transformer::shape13_causal_attention",
        mutates_args={},
    )
    def _shape13_causal_attention_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        output = torch.empty(
            query.shape,
            dtype=query.dtype,
            device=query.device,
        )
        grid = (
            _SEQUENCE_LENGTH // _BLOCK_M,
            _BATCH_SIZE * _NUM_HEADS,
        )
        wrap_triton(_shape13_causal_attention_kernel)[grid](
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
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )
        return output

else:

    def _shape13_causal_attention_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        del query, key, value, scale
        raise RuntimeError("Triton Shape 13 attention is unavailable")


def triton_shape13_causal_attention_available() -> bool:
    """Return whether the optional Triton custom-op stack can be loaded."""

    return all(
        dependency is not None for dependency in (triton, tl, triton_op, wrap_triton)
    )


def can_use_triton_shape13_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_token_mask: torch.Tensor | None = None,
    *,
    causal: bool = True,
    training: bool = False,
) -> bool:
    """Validate the exact tensor contract used by the measured specialization."""

    if not triton_shape13_causal_attention_available():
        return False
    if training or torch.is_grad_enabled() or not causal:
        return False
    if valid_token_mask is not None or query.device.type != "cuda":
        return False
    if query.ndim != 4:
        return False
    if not is_shape13_triton_attention_tensor_family(
        batch_size=query.shape[0],
        seq_len=query.shape[2],
        num_heads=query.shape[1],
        head_dim=query.shape[3],
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


def prevalidated_triton_shape13_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, str]:
    """Run the immutable-plan specialization without re-evaluating Python guards."""

    resolved_scale = 1.0 / math.sqrt(_HEAD_DIM) if scale is None else float(scale)
    try:
        output = _shape13_causal_attention_op(
            query,
            key,
            value,
            resolved_scale,
        )
    except Exception as exc:
        raise RuntimeError("Triton Shape 13 attention execution failed") from exc
    return output, TRITON_SHAPE13_CAUSAL_ATTENTION_BACKEND


def triton_shape13_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_token_mask: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    causal: bool = True,
    training: bool = False,
) -> tuple[torch.Tensor, str]:
    """Run the specialization with an explicit failure instead of fallback."""

    if not can_use_triton_shape13_causal_attention(
        query,
        key,
        value,
        valid_token_mask,
        causal=causal,
        training=training,
    ):
        raise RuntimeError(
            "Triton Shape 13 attention is ineligible for the requested inputs"
        )
    return prevalidated_triton_shape13_causal_attention(
        query,
        key,
        value,
        scale=scale,
    )


__all__ = [
    "TRITON_SHAPE13_CAUSAL_ATTENTION_BACKEND",
    "can_use_triton_shape13_causal_attention",
    "prevalidated_triton_shape13_causal_attention",
    "triton_shape13_causal_attention",
    "triton_shape13_causal_attention_available",
]
