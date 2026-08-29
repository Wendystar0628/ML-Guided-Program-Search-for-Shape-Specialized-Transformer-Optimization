"""Forward-only FP16 attention-output projection without a BHSD-to-BSD copy."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised without the optional runtime.
    triton = None
    tl = None


TRITON_ATTENTION_OUTPUT_PROJECTION_BACKEND = "triton_attention_output_projection"
_DEFAULT_BLOCK_M = 32
_DEFAULT_BLOCK_N = 32
_DEFAULT_BLOCK_K = 32
_DEFAULT_NUM_WARPS = 4
_DEFAULT_NUM_STAGES = 2
_SUPPORTED_WIDTHS = frozenset({32, 128, 1024})
_SUPPORTED_SEQUENCE_LENGTHS = frozenset({32, 128, 1024})
_SUPPORTED_BLOCK_M = frozenset({16, 32, 64, 128})
_SUPPORTED_BLOCK_N = frozenset({16, 32, 64, 128})
_SUPPORTED_BLOCK_K = frozenset({16, 32, 64})
_SUPPORTED_NUM_WARPS = frozenset({2, 4, 8})
_SUPPORTED_NUM_STAGES = frozenset({1, 2, 3, 4})
_MIN_COMPUTE_CAPABILITY = (8, 0)


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
        and block_m in _SUPPORTED_BLOCK_M
        and block_n in _SUPPORTED_BLOCK_N
        and block_k in _SUPPORTED_BLOCK_K
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
        raise ValueError("unsupported Triton attention-output projection launch config")


if triton is not None and tl is not None:

    @triton.jit
    def _attention_output_projection_kernel(
        attention_output,
        weight,
        bias,
        output,
        stride_ab,
        stride_ah,
        stride_as,
        stride_ad,
        stride_wo,
        stride_wi,
        stride_ob,
        stride_os,
        stride_od,
        BATCH_SIZE: tl.constexpr,
        SEQUENCE_LENGTH: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        INPUT_WIDTH: tl.constexpr,
        OUTPUT_WIDTH: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        output_columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        batch_indices = rows // SEQUENCE_LENGTH
        sequence_indices = rows % SEQUENCE_LENGTH
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, INPUT_WIDTH, BLOCK_K):
            input_columns = k_start + tl.arange(0, BLOCK_K)
            head_indices = input_columns // HEAD_DIM
            head_columns = input_columns % HEAD_DIM
            input_mask = (rows[:, None] < BATCH_SIZE * SEQUENCE_LENGTH) & (
                input_columns[None, :] < INPUT_WIDTH
            )
            input_offsets = (
                batch_indices[:, None].to(tl.int64) * stride_ab
                + head_indices[None, :].to(tl.int64) * stride_ah
                + sequence_indices[:, None].to(tl.int64) * stride_as
                + head_columns[None, :].to(tl.int64) * stride_ad
            )
            input_tile = tl.load(
                attention_output + input_offsets,
                mask=input_mask,
                other=0.0,
            )
            weight_offsets = (
                output_columns[:, None].to(tl.int64) * stride_wo
                + input_columns[None, :].to(tl.int64) * stride_wi
            )
            weight_tile = tl.load(
                weight + weight_offsets,
                mask=(output_columns[:, None] < OUTPUT_WIDTH)
                & (input_columns[None, :] < INPUT_WIDTH),
                other=0.0,
            )
            accumulator = tl.dot(input_tile, tl.trans(weight_tile), accumulator)

        if HAS_BIAS:
            accumulator += tl.load(
                bias + output_columns,
                mask=output_columns < OUTPUT_WIDTH,
                other=0.0,
            )[None, :]

        output_mask = (rows[:, None] < BATCH_SIZE * SEQUENCE_LENGTH) & (
            output_columns[None, :] < OUTPUT_WIDTH
        )
        output_offsets = (
            batch_indices[:, None].to(tl.int64) * stride_ob
            + sequence_indices[:, None].to(tl.int64) * stride_os
            + output_columns[None, :].to(tl.int64) * stride_od
        )
        tl.store(output + output_offsets, accumulator, mask=output_mask)


def triton_attention_output_projection_available() -> bool:
    """Return whether the optional Triton implementation can be loaded."""

    return triton is not None and tl is not None


def can_use_triton_attention_output_projection(
    attention_output: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
    block_k: int = _DEFAULT_BLOCK_K,
    num_warps: int = _DEFAULT_NUM_WARPS,
    num_stages: int = _DEFAULT_NUM_STAGES,
) -> bool:
    """Validate a contiguous BHSD attention output and Linear parameter pair."""

    if not triton_attention_output_projection_available():
        return False
    if not _launch_config_is_valid(
        block_m,
        block_n,
        block_k,
        num_warps,
        num_stages,
    ):
        return False
    if torch.is_grad_enabled() or attention_output.device.type != "cuda":
        return False
    if attention_output.ndim != 4 or weight.ndim != 2:
        return False
    batch_size, num_heads, sequence_length, head_dim = attention_output.shape
    input_width = num_heads * head_dim
    output_width, weight_input_width = weight.shape
    if batch_size <= 0 or num_heads <= 0 or head_dim <= 0:
        return False
    if sequence_length not in _SUPPORTED_SEQUENCE_LENGTHS:
        return False
    if input_width not in _SUPPORTED_WIDTHS or output_width not in _SUPPORTED_WIDTHS:
        return False
    if weight_input_width != input_width:
        return False
    if attention_output.dtype != torch.float16 or weight.dtype != torch.float16:
        return False
    if attention_output.device != weight.device:
        return False
    if not attention_output.is_contiguous() or not weight.is_contiguous():
        return False
    if attention_output.requires_grad or weight.requires_grad:
        return False
    if bias is not None:
        if bias.shape != (output_width,) or bias.dtype != torch.float16:
            return False
        if bias.device != attention_output.device or not bias.is_contiguous():
            return False
        if bias.requires_grad:
            return False
    return (
        torch.cuda.get_device_capability(attention_output.device)
        >= _MIN_COMPUTE_CAPABILITY
    )


def prevalidated_triton_attention_output_projection(
    attention_output: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
    block_k: int = _DEFAULT_BLOCK_K,
    num_warps: int = _DEFAULT_NUM_WARPS,
    num_stages: int = _DEFAULT_NUM_STAGES,
) -> tuple[torch.Tensor, str]:
    """Project BHSD directly into a contiguous BSD tensor after plan validation."""

    _validate_launch_config(block_m, block_n, block_k, num_warps, num_stages)
    if triton is None or tl is None:
        raise RuntimeError("Triton attention-output projection is unavailable")

    batch_size, num_heads, sequence_length, head_dim = attention_output.shape
    input_width = num_heads * head_dim
    output_width = weight.shape[0]
    output = torch.empty(
        (batch_size, sequence_length, output_width),
        dtype=attention_output.dtype,
        device=attention_output.device,
    )
    bias_pointer = weight if bias is None else bias
    grid = (
        triton.cdiv(batch_size * sequence_length, block_m),
        triton.cdiv(output_width, block_n),
    )
    try:
        _attention_output_projection_kernel[grid](
            attention_output,
            weight,
            bias_pointer,
            output,
            *attention_output.stride(),
            *weight.stride(),
            *output.stride(),
            BATCH_SIZE=batch_size,
            SEQUENCE_LENGTH=sequence_length,
            HEAD_DIM=head_dim,
            INPUT_WIDTH=input_width,
            OUTPUT_WIDTH=output_width,
            HAS_BIAS=bias is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    except Exception as exc:
        raise RuntimeError("Triton attention-output projection failed") from exc
    return output, TRITON_ATTENTION_OUTPUT_PROJECTION_BACKEND


def triton_attention_output_projection(
    attention_output: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
    block_k: int = _DEFAULT_BLOCK_K,
    num_warps: int = _DEFAULT_NUM_WARPS,
    num_stages: int = _DEFAULT_NUM_STAGES,
) -> tuple[torch.Tensor, str]:
    """Run the guarded BHSD-to-BSD output projection specialization."""

    if not can_use_triton_attention_output_projection(
        attention_output,
        weight,
        bias,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    ):
        raise RuntimeError("Triton attention-output projection is ineligible")
    return prevalidated_triton_attention_output_projection(
        attention_output,
        weight,
        bias,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )


__all__ = [
    "TRITON_ATTENTION_OUTPUT_PROJECTION_BACKEND",
    "can_use_triton_attention_output_projection",
    "prevalidated_triton_attention_output_projection",
    "triton_attention_output_projection",
    "triton_attention_output_projection_available",
]
