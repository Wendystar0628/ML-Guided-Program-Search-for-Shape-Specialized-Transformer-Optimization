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
) -> ExecutionContext:
    return ExecutionContext(
        d_model=128,
        num_heads=4,
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
    assert plan.block_backend == "torch"
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


def test_inplace_block_uses_the_same_support_guard_as_the_helper(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        execution_plan_module,
        "supports_inplace_exact_gelu",
        lambda: True,
    )
    inference = _resolve("inplace-block", _context())
    training = _resolve(
        "inplace-block",
        _context(training=True, grad_enabled=True),
    )

    assert inference.selected_policy == "inplace-block"
    assert inference.block_backend == "inplace_exact_gelu"
    assert training.selected_policy == "safe"
    assert training.block_backend == "torch"
    assert training.resolved_components == ()
    assert training.missing_components == ("inplace_exact_gelu",)


def test_inplace_block_falls_back_when_the_runtime_has_no_exact_gelu(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        execution_plan_module,
        "supports_inplace_exact_gelu",
        lambda: False,
    )

    plan = _resolve("inplace-block", _context())

    assert plan.selected_policy == "safe"
    assert plan.resolved_components == ()
    assert plan.missing_components == ("inplace_exact_gelu",)


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
