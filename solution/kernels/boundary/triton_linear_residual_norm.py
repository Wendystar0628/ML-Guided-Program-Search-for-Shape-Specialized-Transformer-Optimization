"""D32/128 Linear epilogue fused with FP32 residual and LayerNorm."""

from __future__ import annotations

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional GPU runtime.
    triton = None
    tl = None


TRITON_LINEAR_RESIDUAL_NORM_BACKEND = "triton_linear_residual_norm"
_SUPPORTED_WIDTHS = frozenset({32, 128})
_SUPPORTED_BLOCK_ROWS = {
    32: frozenset({16, 32, 64}),
    128: frozenset({16, 32}),
}
_SUPPORTED_NUM_WARPS = {
    32: frozenset({2, 4}),
    128: frozenset({4, 8}),
}
_BLOCK_K = 32
_NUM_STAGES = 2
_MIN_COMPUTE_CAPABILITY = (8, 0)


if triton is not None and tl is not None:

    @triton.jit
    def _linear_residual_norm_kernel(
        source,
        weight,
        linear_bias,
        value,
        norm_weight,
        norm_bias,
        residual,
        normalized,
        stride_sb,
        stride_sh,
        stride_ss,
        stride_sd,
        stride_wo,
        stride_wi,
        row_count,
        sequence_length: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        width: tl.constexpr,
        block_k: tl.constexpr,
        eps: tl.constexpr,
        block_rows: tl.constexpr,
        final_boundary: tl.constexpr,
    ) -> None:
        rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
        output_columns = tl.arange(0, width)
        batch_indices = rows // sequence_length
        sequence_indices = rows % sequence_length
        accumulator = tl.zeros((block_rows, width), dtype=tl.float32)

        for k_start in range(0, width, block_k):
            input_columns = k_start + tl.arange(0, block_k)
            head_indices = input_columns // head_dim
            head_columns = input_columns % head_dim
            source_offsets = (
                batch_indices[:, None].to(tl.int64) * stride_sb
                + head_indices[None, :].to(tl.int64) * stride_sh
                + sequence_indices[:, None].to(tl.int64) * stride_ss
                + head_columns[None, :].to(tl.int64) * stride_sd
            )
            source_tile = tl.load(
                source + source_offsets,
                mask=(rows[:, None] < row_count)
                & (head_indices[None, :] < num_heads),
                other=0.0,
            )
            weight_offsets = (
                input_columns[:, None].to(tl.int64) * stride_wi
                + output_columns[None, :].to(tl.int64) * stride_wo
            )
            weight_tile = tl.load(weight + weight_offsets)
            accumulator = tl.dot(source_tile, weight_tile, accumulator)

        accumulator += tl.load(linear_bias + output_columns)[None, :]
        # Preserve the existing program: the branch update is rounded to FP16
        # before it is added to the FP32 residual stream.
        update = accumulator.to(tl.float16).to(tl.float32)
        offsets = rows[:, None] * width + output_columns[None, :]
        row_mask = rows[:, None] < row_count
        residual_input = tl.load(value + offsets, mask=row_mask, other=0.0)
        summed = residual_input + update
        mean = tl.sum(summed, axis=1) / width
        centered = summed - mean[:, None]
        variance = tl.sum(centered * centered, axis=1) / width
        reciprocal_std = tl.rsqrt(variance + eps)
        scale = tl.load(norm_weight + output_columns).to(tl.float32)
        shift = tl.load(norm_bias + output_columns).to(tl.float32)
        output = centered * reciprocal_std[:, None] * scale + shift

        tl.store(residual + offsets, summed, mask=row_mask)
        if final_boundary:
            tl.store(normalized + offsets, output.to(tl.float32), mask=row_mask)
        else:
            tl.store(normalized + offsets, output.to(tl.float16), mask=row_mask)


def triton_linear_residual_norm_available() -> bool:
    return triton is not None and tl is not None


def _launch_is_valid(width: int, block_rows: int, num_warps: int) -> bool:
    return bool(
        width in _SUPPORTED_WIDTHS
        and block_rows in _SUPPORTED_BLOCK_ROWS[width]
        and num_warps in _SUPPORTED_NUM_WARPS[width]
    )


def can_use_triton_linear_residual_norm(
    source: torch.Tensor,
    weight: torch.Tensor,
    linear_bias: torch.Tensor | None,
    value: torch.Tensor,
    layer_norm: nn.LayerNorm,
    *,
    block_rows: int,
    num_warps: int,
) -> bool:
    """Validate the exact inference-only fused boundary contract."""

    if not triton_linear_residual_norm_available() or torch.is_grad_enabled():
        return False
    if source.device.type != "cuda" or source.ndim not in {3, 4}:
        return False
    if weight.ndim != 2 or value.ndim != 3 or linear_bias is None:
        return False
    width = value.shape[-1]
    if not _launch_is_valid(width, block_rows, num_warps):
        return False
    if weight.shape != (width, width) or linear_bias.shape != (width,):
        return False
    if tuple(source.shape[:1]) != tuple(value.shape[:1]):
        return False
    if source.ndim == 3:
        if tuple(source.shape) != tuple(value.shape):
            return False
    else:
        batch, heads, sequence, head_dim = source.shape
        if (batch, sequence, heads * head_dim) != tuple(value.shape):
            return False
    if source.dtype != torch.float16 or weight.dtype != torch.float16:
        return False
    if linear_bias.dtype != torch.float16 or value.dtype != torch.float32:
        return False
    tensors = (source, weight, linear_bias, value)
    if any(tensor.device != source.device for tensor in tensors):
        return False
    if any(tensor.requires_grad for tensor in tensors):
        return False
    # SDPA commonly returns a stride-only BHSD view over BSD storage.  The
    # kernel consumes the explicit source strides, so requiring physical BHSD
    # contiguity would reject the intended zero-copy boundary.
    if source.stride(-1) != 1 or not weight.is_contiguous():
        return False
    if not linear_bias.is_contiguous() or not value.is_contiguous():
        return False
    if tuple(layer_norm.normalized_shape) != (width,):
        return False
    if layer_norm.weight is None or layer_norm.bias is None:
        return False
    if any(
        parameter.device != value.device or parameter.dtype != torch.float32
        for parameter in (layer_norm.weight, layer_norm.bias)
    ):
        return False
    return torch.cuda.get_device_capability(source.device) >= _MIN_COMPUTE_CAPABILITY


def triton_linear_residual_norm(
    source: torch.Tensor,
    weight: torch.Tensor,
    linear_bias: torch.Tensor,
    value: torch.Tensor,
    layer_norm: nn.LayerNorm,
    *,
    final_boundary: bool,
    block_rows: int,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Execute the fused boundary without materializing the Linear update."""

    if not can_use_triton_linear_residual_norm(
        source,
        weight,
        linear_bias,
        value,
        layer_norm,
        block_rows=block_rows,
        num_warps=num_warps,
    ):
        raise RuntimeError("Triton Linear residual-norm boundary is ineligible")
    assert triton is not None
    assert layer_norm.weight is not None and layer_norm.bias is not None
    batch_size, sequence_length, width = value.shape
    if source.ndim == 4:
        _, num_heads, _, head_dim = source.shape
        strides = source.stride()
    else:
        num_heads = 1
        head_dim = width
        stride_b, stride_s, stride_d = source.stride()
        strides = (stride_b, 0, stride_s, stride_d)
    row_count = batch_size * sequence_length
    residual = torch.empty_like(value)
    normalized = torch.empty_like(
        value,
        dtype=torch.float32 if final_boundary else torch.float16,
    )
    try:
        _linear_residual_norm_kernel[(triton.cdiv(row_count, block_rows),)](
            source,
            weight,
            linear_bias,
            value,
            layer_norm.weight,
            layer_norm.bias,
            residual,
            normalized,
            *strides,
            *weight.stride(),
            row_count,
            sequence_length=sequence_length,
            num_heads=num_heads,
            head_dim=head_dim,
            width=width,
            block_k=_BLOCK_K,
            eps=layer_norm.eps,
            block_rows=block_rows,
            final_boundary=final_boundary,
            num_warps=num_warps,
            num_stages=_NUM_STAGES,
        )
    except Exception as exc:
        raise RuntimeError("Triton Linear residual-norm boundary failed") from exc
    return residual, normalized, TRITON_LINEAR_RESIDUAL_NORM_BACKEND


__all__ = [
    "TRITON_LINEAR_RESIDUAL_NORM_BACKEND",
    "can_use_triton_linear_residual_norm",
    "triton_linear_residual_norm",
    "triton_linear_residual_norm_available",
]
