"""Shape 14-only plan validation with lazy kernel capability detection."""

from __future__ import annotations

from collections.abc import Callable

from ..config import ConfigSpec
from ..plan import ExecutionContext

Reject = Callable[[str, str, str], None]


def validate_streamed_runtime(
    config: ConfigSpec,
    context: ExecutionContext,
    reject: Reject,
) -> ExecutionContext:
    """Validate the streamed outer schedule and return its inner context."""

    microbatch_size = config.schedule.microbatch_size
    if microbatch_size is None:
        raise RuntimeError("streamed configuration is missing its microbatch size")
    if context.has_valid_token_mask:
        reject(
            "mask_not_supported",
            "schedule.runtime",
            "streamed execution currently requires an all-valid workload",
        )
    if microbatch_size > context.batch_size:
        reject(
            "microbatch_exceeds_batch",
            "schedule.microbatch_size",
            "microbatch cannot exceed the logical batch",
        )
    elif context.batch_size % microbatch_size:
        reject(
            "microbatch_not_divisor",
            "schedule.microbatch_size",
            "microbatch must divide the logical batch",
        )
    return context.with_batch_size(microbatch_size)


def _streaming_attention_available(capability: bool | None) -> bool:
    if capability is not None:
        return capability
    from .triton_streaming_dh64 import (
        triton_streaming_dh64_causal_attention_available,
    )

    return triton_streaming_dh64_causal_attention_available()


def validate_streaming_attention(
    *,
    config: ConfigSpec,
    context: ExecutionContext,
    capability: bool | None,
    qkv_dtype: str,
    reject: Reject,
) -> None:
    """Validate the Shape 14 streaming attention template."""

    if not _streaming_attention_available(capability):
        reject(
            "backend_unavailable",
            "program.attention",
            "Shape 14 Triton streaming attention is unavailable",
        )
    if (
        context.batch_size not in {1, 2, 4}
        or context.num_layers != 2
        or context.ffn_dim != 1024
        or context.num_heads != 16
        or context.seq_len != 100000
        or context.d_model != 1024
        or context.causal is not True
        or context.has_valid_token_mask
        or context.head_dim != 64
    ):
        reject(
            "unsupported_shape",
            "program.attention",
            "Shape 14 Triton attention requires a B1/2/4 microbatch "
            "with L2/H16/S100000/Dh64",
        )
    if qkv_dtype != "float16":
        reject(
            "requires_fp16_qkv",
            "program.qkv_projection",
            "Shape 14 Triton attention requires FP16 QKV",
        )

    launch = config.schedule.attention_launch
    if launch is None:
        raise RuntimeError("Shape 14 Triton attention is missing launch parameters")
    if (
        launch.block_m not in {16, 32, 64}
        or launch.block_n not in {16, 32, 64, 128}
        or launch.num_warps not in {2, 4, 8}
        or launch.num_stages not in {1, 2, 3, 4}
        or (launch.block_n == 128 and launch.num_stages == 4)
    ):
        reject(
            "unsupported_launch_value",
            "schedule.attention_launch",
            "Shape 14 streaming launch is outside the implemented template",
        )


__all__ = ["validate_streamed_runtime", "validate_streaming_attention"]
