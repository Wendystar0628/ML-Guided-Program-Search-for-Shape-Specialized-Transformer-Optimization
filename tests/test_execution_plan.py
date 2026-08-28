from __future__ import annotations

import torch

from policy_registry import get_policy_spec
from solution import execution_plan as execution_plan_module
from solution.execution_plan import ExecutionContext, resolve_execution_plan


def _context(
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    causal: bool = True,
    training: bool = False,
    grad_enabled: bool = False,
    has_valid_token_mask: bool = False,
    mask_compatible: bool = True,
    batch_size: int = 64,
    seq_len: int = 128,
    d_model: int = 128,
    num_heads: int = 4,
    ffn_dim: int | None = None,
    num_layers: int = 4,
) -> ExecutionContext:
    return ExecutionContext(
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=d_model,
        num_heads=num_heads,
        causal=causal,
        device=torch.device(device),
        dtype=dtype,
        training=training,
        grad_enabled=grad_enabled,
        input_contiguous=True,
        has_valid_token_mask=has_valid_token_mask,
        mask_compatible=mask_compatible,
        ffn_dim=d_model if ffn_dim is None else ffn_dim,
        num_layers=num_layers,
    )


def _resolve(policy: str, context: ExecutionContext):
    return resolve_execution_plan(
        get_policy_spec(policy),
        context,
        requested_policy=policy,
    )


def test_eager_sdpa_plan_selects_the_shape_independent_native_path() -> None:
    plan = _resolve("eager-sdpa", _context())

    assert plan.selected_policy == "eager-sdpa"
    assert plan.attention_backend == "causal_sdpa"
    assert plan.runtime_wrapper == "eager"
    assert plan.residual_norm_backend == "torch"
    assert plan.resolved_components == ("causal_sdpa",)
    assert plan.missing_components == ()
    assert plan.fallback_reasons == ()


def test_explicit_graph_policy_falls_back_atomically_without_cuda() -> None:
    plan = _resolve("graph", _context(device="cpu"))

    assert plan.selected_policy == "safe"
    assert plan.attention_backend == "safe_streaming"
    assert plan.runtime_wrapper == "eager"
    assert plan.resolved_components == ()
    assert plan.missing_components == ("cuda_graph",)
    assert plan.fallback_reasons == ("cuda_graph_not_eligible",)


def test_graph_fused_norm_plan_enforces_runtime_safety_not_performance_shapes() -> None:
    small = _resolve(
        "graph-fused-norm",
        _context(device="cuda", batch_size=1),
    )
    graph_fused_on_large = _resolve(
        "graph-fused-norm",
        _context(device="cuda", batch_size=10_000),
    )
    training = _resolve(
        "graph-fused-norm",
        _context(
            device="cuda",
            batch_size=1,
            training=True,
            grad_enabled=True,
        ),
    )

    assert small.selected_policy == "graph-fused-norm"
    assert small.runtime_wrapper == "cuda_graph"
    assert small.residual_norm_backend == "compiled_residual_layer_norm"
    assert graph_fused_on_large.selected_policy == "graph-fused-norm"
    assert graph_fused_on_large.runtime_wrapper == "cuda_graph"
    assert training.selected_policy == "safe"
    assert training.residual_norm_backend == "torch"
    assert training.resolved_components == ()
    assert training.missing_components == (
        "compiled_residual_layer_norm",
        "cuda_graph",
    )


def test_mixed_attention_supports_only_measured_shape_families(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        torch.backends.cuda,
        "mem_efficient_sdp_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (8, 9),
    )
    long_plan = _resolve(
        "mixed-fp16-efficient",
        _context(device="cuda", seq_len=1024),
    )
    long_head_dim_64_plan = _resolve(
        "mixed-fp16-efficient",
        _context(
            device="cuda",
            seq_len=1024,
            d_model=1024,
            num_heads=16,
        ),
    )
    graph_mixed_plan = _resolve(
        "graph-mixed-fp16-efficient",
        _context(device="cuda", batch_size=128, d_model=128, num_heads=16),
    )
    graph_mixed_compiled_plan = _resolve(
        "graph-mixed-fp16-efficient-compiled-norm",
        _context(device="cuda", batch_size=128, d_model=128, num_heads=16),
    )
    unsupported_long = _resolve(
        "mixed-fp16-efficient",
        _context(device="cuda", seq_len=512),
    )
    unsupported_head_dim = _resolve(
        "mixed-fp16-efficient",
        _context(device="cuda", seq_len=1024, d_model=128, num_heads=8),
    )
    unsupported_graph = _resolve(
        "graph-mixed-fp16-efficient",
        _context(device="cuda", batch_size=16, d_model=128),
    )

    assert long_plan.attention_backend == "mixed_fp16_efficient"
    assert long_head_dim_64_plan.attention_backend == "mixed_fp16_efficient"
    assert graph_mixed_plan.attention_backend == "mixed_fp16_efficient"
    assert graph_mixed_plan.runtime_wrapper == "cuda_graph"
    assert (
        graph_mixed_compiled_plan.selected_policy
        == "graph-mixed-fp16-efficient-compiled-norm"
    )
    assert graph_mixed_compiled_plan.runtime_wrapper == "cuda_graph"
    assert (
        graph_mixed_compiled_plan.residual_norm_backend
        == "compiled_residual_layer_norm"
    )
    assert unsupported_long.selected_policy == "safe"
    assert unsupported_head_dim.selected_policy == "safe"
    assert unsupported_graph.selected_policy == "safe"
    assert unsupported_long.missing_components == ("mixed_fp16_efficient_attention",)


def test_mixed_cudnn_plan_is_limited_to_unmasked_long_head_dim_64(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: True)
    monkeypatch.setattr(torch.backends.cudnn, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (8, 9),
    )
    supported = _resolve(
        "mixed-fp16-cudnn",
        _context(
            device="cuda",
            batch_size=1,
            seq_len=1024,
            d_model=1024,
            num_heads=16,
        ),
    )
    short = _resolve(
        "mixed-fp16-cudnn",
        _context(device="cuda", seq_len=512, d_model=1024, num_heads=16),
    )
    head_dim_32 = _resolve(
        "mixed-fp16-cudnn",
        _context(device="cuda", seq_len=1024, d_model=128, num_heads=4),
    )
    masked = _resolve(
        "mixed-fp16-cudnn",
        _context(
            device="cuda",
            seq_len=1024,
            d_model=1024,
            num_heads=16,
            has_valid_token_mask=True,
        ),
    )

    assert supported.selected_policy == "mixed-fp16-cudnn"
    assert supported.attention_backend == "mixed_fp16_cudnn"
    assert supported.resolved_components == ("mixed_fp16_cudnn_attention",)
    for unsupported in (short, head_dim_32, masked):
        assert unsupported.selected_policy == "safe"
        assert unsupported.attention_backend == "safe_streaming"
        assert unsupported.missing_components == ("mixed_fp16_cudnn_attention",)


def test_full_mixed_core_plans_report_actual_compute_contracts(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        torch.backends.cuda,
        "mem_efficient_sdp_enabled",
        lambda: True,
    )
    monkeypatch.setattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: True)
    monkeypatch.setattr(torch.backends.cudnn, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (8, 9),
    )

    efficient = _resolve(
        "mixed-fp16-core-efficient",
        _context(
            device="cuda",
            batch_size=64,
            seq_len=128,
            d_model=1024,
            num_heads=4,
        ),
    )
    streamed_cudnn = _resolve(
        "mixed-fp16-core-cudnn",
        _context(
            device="cuda",
            batch_size=1,
            seq_len=100_000,
            d_model=1024,
            num_heads=16,
            ffn_dim=1024,
            num_layers=2,
        ),
    )
    streamed_efficient = _resolve(
        "mixed-fp16-core-efficient",
        _context(
            device="cuda",
            batch_size=1,
            seq_len=100_000,
            d_model=1024,
            num_heads=16,
            ffn_dim=1024,
            num_layers=2,
        ),
    )

    assert efficient.selected_policy == "mixed-fp16-core-efficient"
    assert efficient.attention_backend == "mixed_fp16_efficient"
    assert efficient.attention_compute_dtype == "float16"
    assert efficient.linear_backend == "autocast_fp16"
    assert efficient.linear_compute_dtype == "float16"
    assert efficient.resolved_components == (
        "mixed_fp16_core",
        "mixed_fp16_efficient_attention",
    )
    assert streamed_cudnn.selected_policy == "mixed-fp16-core-cudnn"
    assert streamed_cudnn.attention_backend == "mixed_fp16_cudnn"
    assert streamed_cudnn.attention_compute_dtype == "float16"
    assert streamed_cudnn.linear_backend == "autocast_fp16"
    assert streamed_cudnn.linear_compute_dtype == "float16"
    assert streamed_efficient.selected_policy == "mixed-fp16-core-efficient"
    assert streamed_efficient.attention_backend == "mixed_fp16_efficient"
    assert streamed_efficient.linear_backend == "autocast_fp16"


def test_graph_mixed_core_composes_every_requested_component(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        torch.backends.cuda,
        "mem_efficient_sdp_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (8, 9),
    )

    plan = _resolve(
        "graph-mixed-fp16-core-efficient-compiled-norm",
        _context(
            device="cuda",
            batch_size=64,
            seq_len=128,
            d_model=128,
            num_heads=16,
            ffn_dim=128,
            num_layers=4,
        ),
    )

    assert plan.selected_policy == ("graph-mixed-fp16-core-efficient-compiled-norm")
    assert plan.attention_backend == "mixed_fp16_efficient"
    assert plan.attention_compute_dtype == "float16"
    assert plan.linear_backend == "autocast_fp16"
    assert plan.linear_compute_dtype == "float16"
    assert plan.runtime_wrapper == "cuda_graph"
    assert plan.residual_norm_backend == "compiled_residual_layer_norm"
    assert plan.resolved_components == (
        "compiled_residual_layer_norm",
        "cuda_graph",
        "mixed_fp16_core",
        "mixed_fp16_efficient_attention",
    )


def test_shape06_batch_tiled_plan_is_exact_and_self_describing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        torch.backends.cuda,
        "mem_efficient_sdp_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (8, 9),
    )

    exact = _resolve(
        "batch-tiled-mixed-fp16-core-efficient-compiled-norm",
        _context(
            device="cuda",
            batch_size=10_000,
            seq_len=128,
            d_model=128,
            num_heads=4,
            ffn_dim=128,
            num_layers=4,
        ),
    )
    nearby = _resolve(
        "batch-tiled-mixed-fp16-core-efficient-compiled-norm",
        _context(
            device="cuda",
            batch_size=9_999,
            seq_len=128,
            d_model=128,
            num_heads=4,
            ffn_dim=128,
            num_layers=4,
        ),
    )

    assert exact.selected_policy == (
        "batch-tiled-mixed-fp16-core-efficient-compiled-norm"
    )
    assert exact.runtime_wrapper == "batch_tiled_cuda_graph"
    assert exact.batch_tile_size == 128
    assert exact.resolved_components == (
        "batch_tiled_cuda_graph",
        "compiled_residual_layer_norm",
        "cuda_graph",
        "mixed_fp16_core",
        "mixed_fp16_efficient_attention",
    )
    assert nearby.selected_policy == "safe"
    assert nearby.runtime_wrapper == "eager"
    assert nearby.batch_tile_size is None


def test_compiled_forward_plan_is_exact_to_measured_shapes(monkeypatch) -> None:
    monkeypatch.setattr(
        torch.backends.cuda,
        "mem_efficient_sdp_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (8, 9),
    )

    exact = _resolve(
        "compiled-mixed-fp16-core-efficient",
        _context(
            device="cuda",
            batch_size=64,
            seq_len=1024,
            d_model=128,
            num_heads=4,
            ffn_dim=128,
            num_layers=4,
        ),
    )
    nearby = _resolve(
        "compiled-mixed-fp16-core-efficient",
        _context(
            device="cuda",
            batch_size=64,
            seq_len=128,
            d_model=128,
            num_heads=4,
            ffn_dim=128,
            num_layers=4,
        ),
    )
    wide = _resolve(
        "compiled-mixed-fp16-core-efficient",
        _context(
            device="cuda",
            batch_size=64,
            seq_len=128,
            d_model=1024,
            num_heads=4,
            ffn_dim=1024,
            num_layers=4,
        ),
    )

    assert exact.selected_policy == "compiled-mixed-fp16-core-efficient"
    assert exact.runtime_wrapper == "compiled_forward"
    assert exact.use_compiled_forward
    assert wide.selected_policy == "compiled-mixed-fp16-core-efficient"
    assert wide.use_compiled_forward
    assert exact.resolved_components == (
        "compiled_forward",
        "mixed_fp16_core",
        "mixed_fp16_efficient_attention",
    )
    assert nearby.selected_policy == "safe"
    assert nearby.runtime_wrapper == "eager"


def test_triton_residual_norm_composes_with_shape06_mixed_core(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        torch.backends.cuda,
        "mem_efficient_sdp_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (8, 9),
    )
    monkeypatch.setattr(
        execution_plan_module,
        "triton_residual_layer_norm_available",
        lambda: True,
    )

    plan = _resolve(
        "mixed-fp16-core-efficient-triton-norm",
        _context(
            device="cuda",
            batch_size=10_000,
            seq_len=128,
            d_model=128,
            num_heads=4,
            ffn_dim=128,
            num_layers=4,
        ),
    )

    assert plan.selected_policy == "mixed-fp16-core-efficient-triton-norm"
    assert plan.residual_norm_backend == "triton_residual_layer_norm"
    assert plan.resolved_components == (
        "mixed_fp16_core",
        "mixed_fp16_efficient_attention",
        "triton_residual_layer_norm",
    )


def test_full_mixed_core_falls_back_atomically_outside_its_shape_family(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        torch.backends.cuda,
        "mem_efficient_sdp_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (8, 9),
    )

    plan = _resolve(
        "mixed-fp16-core-efficient",
        _context(
            device="cuda",
            batch_size=64,
            seq_len=128,
            d_model=512,
            num_heads=4,
        ),
    )

    assert plan.selected_policy == "safe"
    assert plan.linear_backend == "torch"
    assert plan.linear_compute_dtype == "float32"
    assert "mixed_fp16_core" in plan.missing_components


def test_bfloat16_eager_sdpa_uses_the_comparator_safe_streaming_path() -> None:
    plan = _resolve("eager-sdpa", _context(dtype=torch.bfloat16))

    assert plan.selected_policy == "safe"
    assert plan.attention_backend == "safe_streaming"
    assert plan.missing_components == ("causal_sdpa",)


def test_description_reports_the_exact_mask_and_backend_mechanisms() -> None:
    native = _resolve(
        "eager-sdpa",
        _context(has_valid_token_mask=True),
    ).describe(
        dispatch_source=None,
        dispatch_table_sha256=None,
        dispatch_policy=None,
        route_origin="explicit",
        causal=True,
    )
    safe = _resolve(
        "safe",
        _context(has_valid_token_mask=False),
    ).describe(
        dispatch_source=None,
        dispatch_table_sha256=None,
        dispatch_policy=None,
        route_origin="explicit",
        causal=True,
    )
    noncausal = _resolve(
        "safe",
        _context(causal=False, has_valid_token_mask=True),
    ).describe(
        dispatch_source=None,
        dispatch_table_sha256=None,
        dispatch_policy=None,
        route_origin="explicit",
        causal=False,
    )

    assert native["causal_mask"] == "implicit_sdpa"
    assert native["valid_token_mask"] == "direct_key_mask"
    assert native["fallback_reasons"] == []
    assert safe["causal_mask"] == "query_block"
    assert safe["valid_token_mask"] == "none"
    assert noncausal["causal_mask"] == "none"
    assert noncausal["valid_token_mask"] == "direct_key_mask"
    assert "layer_backends" not in native
    assert "batch_strategy" not in native
    assert "batch_tile_size" not in native
