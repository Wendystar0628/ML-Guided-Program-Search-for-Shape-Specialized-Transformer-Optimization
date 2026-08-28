from __future__ import annotations

import torch

from policy_registry import get_policy_spec
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
    )


def _resolve(policy: str, context: ExecutionContext):
    return resolve_execution_plan(
        get_policy_spec(policy),
        context,
        requested_policy=policy,
    )


def test_auto_plan_selects_the_shape_independent_native_path() -> None:
    plan = _resolve("auto", _context())

    assert plan.selected_policy == "auto"
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
    graph_mixed_plan = _resolve(
        "graph-mixed-fp16-efficient",
        _context(device="cuda", batch_size=128, d_model=128, num_heads=16),
    )
    unsupported_long = _resolve(
        "mixed-fp16-efficient",
        _context(device="cuda", seq_len=512),
    )
    unsupported_graph = _resolve(
        "graph-mixed-fp16-efficient",
        _context(device="cuda", batch_size=16, d_model=128),
    )

    assert long_plan.attention_backend == "mixed_fp16_efficient"
    assert graph_mixed_plan.attention_backend == "mixed_fp16_efficient"
    assert graph_mixed_plan.runtime_wrapper == "cuda_graph"
    assert unsupported_long.selected_policy == "safe"
    assert unsupported_graph.selected_policy == "safe"
    assert unsupported_long.missing_components == ("mixed_fp16_efficient_attention",)


def test_bfloat16_auto_uses_the_comparator_safe_streaming_path() -> None:
    plan = _resolve("auto", _context(dtype=torch.bfloat16))

    assert plan.selected_policy == "safe"
    assert plan.attention_backend == "safe_streaming"
    assert plan.missing_components == ("causal_sdpa",)


def test_description_reports_the_exact_mask_and_backend_mechanisms() -> None:
    native = _resolve(
        "auto",
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
