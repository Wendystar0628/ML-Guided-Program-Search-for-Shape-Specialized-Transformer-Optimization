"""D128 FFN fused through its FP32 residual and LayerNorm boundary."""

from __future__ import annotations

import math

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional GPU runtime.
    triton = None
    tl = None


TRITON_FFN_RESIDUAL_NORM_BACKEND = "triton_ffn_residual_norm"
_WIDTH = 128
_BLOCK_K = 32
_DEFAULT_BLOCK_ROWS = 16
_DEFAULT_NUM_WARPS = 4
_DEFAULT_NUM_STAGES = 2
_SUPPORTED_BLOCK_ROWS = frozenset({_DEFAULT_BLOCK_ROWS})
_SUPPORTED_NUM_WARPS = frozenset({_DEFAULT_NUM_WARPS})
_SUPPORTED_NUM_STAGES = frozenset({1, 2})
_MIN_COMPUTE_CAPABILITY = (8, 0)


def _launch_is_valid(block_rows: int, num_warps: int, num_stages: int) -> bool:
    values = (block_rows, num_warps, num_stages)
    return bool(
        not any(isinstance(value, bool) for value in values)
        and all(isinstance(value, int) for value in values)
        and block_rows in _SUPPORTED_BLOCK_ROWS
        and num_warps in _SUPPORTED_NUM_WARPS
        and num_stages in _SUPPORTED_NUM_STAGES
    )


if triton is not None and tl is not None:

    @triton.jit
    def _ffn_residual_norm_kernel(
        source,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
        value,
        norm_weight,
        norm_bias,
        residual,
        normalized,
        row_count,
        WIDTH: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        FINAL_BOUNDARY: tl.constexpr,
    ) -> None:
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        input_columns = tl.arange(0, BLOCK_K)
        output_columns = tl.arange(0, WIDTH)
        row_mask = rows[:, None] < row_count
        output_accumulator = tl.zeros((BLOCK_M, WIDTH), dtype=tl.float32)

        # Stream a complete D=128 hidden row through W2 in four K=32 tiles.
        # This preserves the current FP16 hidden boundary without writing it to
        # global memory.
        for hidden_start in range(0, WIDTH, BLOCK_K):
            hidden_columns = hidden_start + tl.arange(0, BLOCK_K)
            hidden_accumulator = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

            for input_start in range(0, WIDTH, BLOCK_K):
                source_columns = input_start + input_columns
                source_tile = tl.load(
                    source + rows[:, None] * WIDTH + source_columns[None, :],
                    mask=row_mask,
                    other=0.0,
                )
                input_weight_tile = tl.load(
                    input_weight
                    + source_columns[:, None] * 1
                    + hidden_columns[None, :] * WIDTH
                )
                hidden_accumulator = tl.dot(
                    source_tile,
                    input_weight_tile,
                    hidden_accumulator,
                )

            hidden_projected = hidden_accumulator + tl.load(
                input_bias + hidden_columns
            )[None, :].to(tl.float32)
            hidden_gelu = 0.5 * hidden_projected * (
                1.0
                + tl.erf(hidden_projected * (1.0 / math.sqrt(2.0)))
            )
            # The unfused program stores the Exact-GELU result to an FP16
            # tensor before W2 consumes it.  Keep that rounding point explicit.
            hidden_fp16 = hidden_gelu.to(tl.float16)
            output_weight_tile = tl.load(
                output_weight
                + hidden_columns[:, None] * 1
                + output_columns[None, :] * WIDTH
            )
            output_accumulator = tl.dot(
                hidden_fp16,
                output_weight_tile,
                output_accumulator,
            )

        output_accumulator += tl.load(output_bias + output_columns)[None, :]
        # Match the mixed residual stream: W2 is rounded to FP16 before being
        # promoted back to FP32 for the residual addition.
        update = output_accumulator.to(tl.float16).to(tl.float32)
        offsets = rows[:, None] * WIDTH + output_columns[None, :]
        residual_input = tl.load(value + offsets, mask=row_mask, other=0.0)
        summed = residual_input + update
        mean = tl.sum(summed, axis=1) / WIDTH
        centered = summed - mean[:, None]
        variance = tl.sum(centered * centered, axis=1) / WIDTH
        reciprocal_std = tl.rsqrt(variance + eps)
        scale = tl.load(norm_weight + output_columns).to(tl.float32)
        shift = tl.load(norm_bias + output_columns).to(tl.float32)
        output = centered * reciprocal_std[:, None] * scale + shift

        tl.store(residual + offsets, summed, mask=row_mask)
        if FINAL_BOUNDARY:
            tl.store(normalized + offsets, output.to(tl.float32), mask=row_mask)
        else:
            tl.store(normalized + offsets, output.to(tl.float16), mask=row_mask)


def triton_ffn_residual_norm_available() -> bool:
    """Return whether the optional Triton implementation can be loaded."""

    return triton is not None and tl is not None


def can_use_triton_ffn_residual_norm(
    source: torch.Tensor,
    input_weight: torch.Tensor,
    input_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    value: torch.Tensor,
    layer_norm: nn.LayerNorm,
    *,
    block_rows: int = _DEFAULT_BLOCK_ROWS,
    num_warps: int = _DEFAULT_NUM_WARPS,
    num_stages: int = _DEFAULT_NUM_STAGES,
) -> bool:
    """Validate the narrow inference-only fused D=F=128 FFN contract."""

    if not triton_ffn_residual_norm_available() or torch.is_grad_enabled():
        return False
    if not _launch_is_valid(block_rows, num_warps, num_stages):
        return False
    tensors = (
        source,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
        value,
    )
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return False
    if source.device.type != "cuda" or source.ndim != 3 or value.ndim != 3:
        return False
    if source.numel() == 0:
        return False
    if tuple(source.shape) != tuple(value.shape) or source.shape[-1] != _WIDTH:
        return False
    if input_weight.shape != (_WIDTH, _WIDTH):
        return False
    if output_weight.shape != (_WIDTH, _WIDTH):
        return False
    if input_bias.shape != (_WIDTH,) or output_bias.shape != (_WIDTH,):
        return False
    if source.dtype != torch.float16 or value.dtype != torch.float32:
        return False
    if any(
        tensor.dtype != torch.float16
        for tensor in (input_weight, input_bias, output_weight, output_bias)
    ):
        return False
    if any(tensor.device != source.device for tensor in tensors):
        return False
    if any(tensor.requires_grad for tensor in tensors):
        return False
    if any(not tensor.is_contiguous() for tensor in tensors):
        return False
    if not isinstance(layer_norm, nn.LayerNorm):
        return False
    if tuple(layer_norm.normalized_shape) != (_WIDTH,):
        return False
    if layer_norm.weight is None or layer_norm.bias is None:
        return False
    if any(
        parameter.device != source.device
        or parameter.dtype != torch.float32
        or not parameter.is_contiguous()
        for parameter in (layer_norm.weight, layer_norm.bias)
    ):
        return False
    return torch.cuda.get_device_capability(source.device) >= _MIN_COMPUTE_CAPABILITY


def triton_ffn_residual_norm(
    source: torch.Tensor,
    input_weight: torch.Tensor,
    input_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    value: torch.Tensor,
    layer_norm: nn.LayerNorm,
    *,
    final_boundary: bool,
    block_rows: int = _DEFAULT_BLOCK_ROWS,
    num_warps: int = _DEFAULT_NUM_WARPS,
    num_stages: int = _DEFAULT_NUM_STAGES,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Execute the fused FFN without materializing hidden or branch update."""

    if not isinstance(final_boundary, bool):
        raise TypeError("final_boundary must be a bool")
    if not can_use_triton_ffn_residual_norm(
        source,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
        value,
        layer_norm,
        block_rows=block_rows,
        num_warps=num_warps,
        num_stages=num_stages,
    ):
        raise RuntimeError("Triton fused FFN residual-norm boundary is ineligible")
    assert triton is not None
    assert layer_norm.weight is not None and layer_norm.bias is not None
    row_count = source.numel() // _WIDTH
    residual = torch.empty_like(value)
    normalized = torch.empty_like(
        value,
        dtype=torch.float32 if final_boundary else torch.float16,
    )
    try:
        _ffn_residual_norm_kernel[(triton.cdiv(row_count, block_rows),)](
            source,
            input_weight,
            input_bias,
            output_weight,
            output_bias,
            value,
            layer_norm.weight,
            layer_norm.bias,
            residual,
            normalized,
            row_count,
            WIDTH=_WIDTH,
            eps=float(layer_norm.eps),
            BLOCK_M=block_rows,
            BLOCK_K=_BLOCK_K,
            FINAL_BOUNDARY=final_boundary,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    except Exception as exc:
        raise RuntimeError("Triton fused FFN residual-norm execution failed") from exc
    return residual, normalized, TRITON_FFN_RESIDUAL_NORM_BACKEND


__all__ = [
    "TRITON_FFN_RESIDUAL_NORM_BACKEND",
    "can_use_triton_ffn_residual_norm",
    "triton_ffn_residual_norm",
    "triton_ffn_residual_norm_available",
]
