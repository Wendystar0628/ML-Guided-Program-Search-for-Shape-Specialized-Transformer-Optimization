"""Pure shape-family predicates shared by planning and candidate gating."""

from __future__ import annotations


def is_mixed_fp16_core_efficient_attention_family(
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> bool:
    """Return whether one attention request belongs to a measured core family."""

    extreme_batch = (
        batch_size >= 1024 and seq_len == 128 and num_heads == 4 and head_dim == 32
    )
    wide_projection = (
        batch_size == 64 and seq_len == 128 and num_heads == 4 and head_dim == 256
    )
    long_sequence = (
        batch_size == 64 and seq_len == 1024 and num_heads == 4 and head_dim == 32
    )
    return extreme_batch or wide_projection or long_sequence


def is_streamed_mixed_fp16_core_cudnn_slice(
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    ffn_dim: int,
    num_layers: int,
) -> bool:
    """Return whether one resident slice belongs to the streamed long case."""

    return bool(
        0 < batch_size <= 32
        and seq_len == 100_000
        and num_heads == 16
        and head_dim == 64
        and ffn_dim == 1024
        and num_layers == 2
    )


def is_measured_mixed_fp16_core_efficient_workload(
    *,
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    ffn_dim: int,
    num_layers: int,
) -> bool:
    """Limit deployment to the three full workloads measured on the GPU."""

    if num_heads != 4 or num_layers != 4 or ffn_dim != d_model:
        return False
    return bool(
        (batch_size == 10_000 and seq_len == 128 and d_model == 128)
        or (batch_size == 64 and seq_len == 128 and d_model == 1024)
        or (batch_size == 64 and seq_len == 1024 and d_model == 128)
    )


def is_measured_streamed_mixed_fp16_core_cudnn_workload(
    *,
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    ffn_dim: int,
    num_layers: int,
) -> bool:
    """Limit deployment to the measured logical streamed workload."""

    return bool(
        batch_size == 32
        and seq_len == 100_000
        and d_model == 1024
        and num_heads == 16
        and ffn_dim == 1024
        and num_layers == 2
    )


def is_measured_triton_residual_norm_workload(
    *,
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    ffn_dim: int,
    num_layers: int,
) -> bool:
    """Return whether the full workload matches the measured Triton fusion."""

    return bool(
        batch_size == 10_000
        and seq_len == 128
        and d_model == 128
        and num_heads == 4
        and ffn_dim == 128
        and num_layers == 4
    )


__all__ = [
    "is_measured_mixed_fp16_core_efficient_workload",
    "is_measured_streamed_mixed_fp16_core_cudnn_workload",
    "is_measured_triton_residual_norm_workload",
    "is_mixed_fp16_core_efficient_attention_family",
    "is_streamed_mixed_fp16_core_cudnn_slice",
]
