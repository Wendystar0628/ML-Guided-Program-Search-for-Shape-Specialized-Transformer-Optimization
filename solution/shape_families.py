"""Pure shape-family predicates shared by planning and candidate gating."""

from __future__ import annotations


def is_mixed_fp16_core_efficient_runtime_family(
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> bool:
    """Return whether mixed FP16 core execution is runtime-legal."""

    extreme_batch = (
        batch_size >= 1024 and seq_len == 128 and num_heads == 4 and head_dim == 32
    )
    wide_projection = (
        batch_size == 64 and seq_len == 128 and num_heads == 4 and head_dim == 256
    )
    long_sequence = (
        batch_size == 64 and seq_len == 1024 and num_heads == 4 and head_dim == 32
    )
    short_graph = (
        batch_size in {64, 128}
        and seq_len == 128
        and num_heads in {1, 2, 4, 16}
        and num_heads * head_dim in {32, 128}
    )
    return extreme_batch or wide_projection or long_sequence or short_graph


def is_graph_mixed_fp16_core_candidate_workload(
    *,
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    ffn_dim: int,
    num_layers: int,
) -> bool:
    """Limit the graph mixed-core experiment to the official short family."""

    return bool(
        batch_size in {64, 128}
        and seq_len == 128
        and d_model in {32, 128}
        and num_heads in {1, 2, 4, 16}
        and d_model % num_heads == 0
        and ffn_dim == d_model
        and num_layers == 4
    )


def is_shape06_batch_tiled_workload(
    *,
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    ffn_dim: int,
    num_layers: int,
) -> bool:
    """Match the exact workload whose batch dimension is cache tiled."""

    return bool(
        batch_size == 10_000
        and seq_len == 128
        and d_model == 128
        and num_heads == 4
        and ffn_dim == 128
        and num_layers == 4
    )


def is_compiled_forward_candidate_workload(
    *,
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    ffn_dim: int,
    num_layers: int,
) -> bool:
    """Match resident workloads selected for fixed-plan full-stack compilation."""

    small_head = (
        batch_size == 64
        and seq_len == 128
        and d_model in {32, 128}
        and num_heads in {4, 16}
        and d_model // num_heads == 8
        and ffn_dim == d_model
        and num_layers == 4
    )

    shape13 = (
        batch_size == 64
        and seq_len == 1024
        and d_model == 128
        and num_heads == 4
        and ffn_dim == 128
        and num_layers == 4
    )
    shape08 = (
        batch_size == 64
        and seq_len == 128
        and d_model == 1024
        and num_heads == 4
        and ffn_dim == 1024
        and num_layers == 4
    )
    return bool(small_head or shape08 or shape13)


def is_shape13_triton_attention_tensor_family(
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> bool:
    """Match the exact Q/K/V contract of the measured Shape 13 kernel."""

    return bool(
        batch_size == 64 and seq_len == 1024 and num_heads == 4 and head_dim == 32
    )


def is_shape13_triton_attention_workload(
    *,
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    ffn_dim: int,
    num_layers: int,
) -> bool:
    """Match the complete workload measured for the Shape 13 specialization."""

    if num_heads <= 0 or d_model % num_heads:
        return False
    return bool(
        is_shape13_triton_attention_tensor_family(
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=d_model // num_heads,
        )
        and d_model == 128
        and ffn_dim == 128
        and num_layers == 4
    )


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
    "is_compiled_forward_candidate_workload",
    "is_graph_mixed_fp16_core_candidate_workload",
    "is_measured_mixed_fp16_core_efficient_workload",
    "is_measured_streamed_mixed_fp16_core_cudnn_workload",
    "is_measured_triton_residual_norm_workload",
    "is_mixed_fp16_core_efficient_runtime_family",
    "is_shape06_batch_tiled_workload",
    "is_shape13_triton_attention_tensor_family",
    "is_shape13_triton_attention_workload",
    "is_streamed_mixed_fp16_core_cudnn_slice",
]
