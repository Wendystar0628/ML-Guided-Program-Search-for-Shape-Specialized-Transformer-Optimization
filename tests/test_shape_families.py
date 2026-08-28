from __future__ import annotations

from solution.shape_families import (
    is_compiled_forward_candidate_workload,
    is_graph_mixed_fp16_core_candidate_workload,
    is_measured_mixed_fp16_core_efficient_workload,
    is_measured_streamed_mixed_fp16_core_cudnn_workload,
    is_measured_triton_residual_norm_workload,
    is_mixed_fp16_core_efficient_runtime_family,
    is_shape06_batch_tiled_workload,
    is_shape13_triton_attention_tensor_family,
    is_shape13_triton_attention_workload,
    is_streamed_mixed_fp16_core_cudnn_slice,
)


def test_solution_predicates_cover_runtime_slices_without_broadening_deployment() -> (
    None
):
    assert is_mixed_fp16_core_efficient_runtime_family(
        batch_size=10_000,
        seq_len=128,
        num_heads=4,
        head_dim=32,
    )
    assert is_mixed_fp16_core_efficient_runtime_family(
        batch_size=64,
        seq_len=128,
        num_heads=16,
        head_dim=8,
    )
    assert is_streamed_mixed_fp16_core_cudnn_slice(
        batch_size=8,
        seq_len=100_000,
        num_heads=16,
        head_dim=64,
        ffn_dim=1024,
        num_layers=2,
    )
    assert not is_streamed_mixed_fp16_core_cudnn_slice(
        batch_size=33,
        seq_len=100_000,
        num_heads=16,
        head_dim=64,
        ffn_dim=1024,
        num_layers=2,
    )

    assert is_graph_mixed_fp16_core_candidate_workload(
        batch_size=128,
        seq_len=128,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
    )
    assert not is_graph_mixed_fp16_core_candidate_workload(
        batch_size=128,
        seq_len=128,
        d_model=128,
        num_heads=4,
        ffn_dim=256,
        num_layers=4,
    )

    assert is_shape06_batch_tiled_workload(
        batch_size=10_000,
        seq_len=128,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
    )
    assert not is_shape06_batch_tiled_workload(
        batch_size=9_999,
        seq_len=128,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
    )

    assert is_compiled_forward_candidate_workload(
        batch_size=64,
        seq_len=1024,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
    )
    assert is_compiled_forward_candidate_workload(
        batch_size=64,
        seq_len=128,
        d_model=1024,
        num_heads=4,
        ffn_dim=1024,
        num_layers=4,
    )
    assert not is_compiled_forward_candidate_workload(
        batch_size=64,
        seq_len=128,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
    )

    assert is_shape13_triton_attention_tensor_family(
        batch_size=64,
        seq_len=1024,
        num_heads=4,
        head_dim=32,
    )
    assert is_shape13_triton_attention_workload(
        batch_size=64,
        seq_len=1024,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
    )
    assert not is_shape13_triton_attention_workload(
        batch_size=64,
        seq_len=1024,
        d_model=128,
        num_heads=4,
        ffn_dim=256,
        num_layers=4,
    )

    assert is_measured_mixed_fp16_core_efficient_workload(
        batch_size=64,
        seq_len=128,
        d_model=1024,
        num_heads=4,
        ffn_dim=1024,
        num_layers=4,
    )
    assert not is_measured_mixed_fp16_core_efficient_workload(
        batch_size=64,
        seq_len=128,
        d_model=1024,
        num_heads=4,
        ffn_dim=2048,
        num_layers=4,
    )
    assert is_measured_streamed_mixed_fp16_core_cudnn_workload(
        batch_size=32,
        seq_len=100_000,
        d_model=1024,
        num_heads=16,
        ffn_dim=1024,
        num_layers=2,
    )
    assert not is_measured_streamed_mixed_fp16_core_cudnn_workload(
        batch_size=16,
        seq_len=100_000,
        d_model=1024,
        num_heads=16,
        ffn_dim=1024,
        num_layers=2,
    )
    assert is_measured_triton_residual_norm_workload(
        batch_size=10_000,
        seq_len=128,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
    )
    assert not is_measured_triton_residual_norm_workload(
        batch_size=10_000,
        seq_len=128,
        d_model=1024,
        num_heads=4,
        ffn_dim=1024,
        num_layers=4,
    )
