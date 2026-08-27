"""Focused policy-identity tests for immutable execution planning."""

from __future__ import annotations

import torch

from solution import execution_plan
from solution.execution_plan import ExecutionContext, resolve_execution_plan
from solution.policies import get_policy_spec


def _context(*, dtype: torch.dtype = torch.float16) -> ExecutionContext:
    return ExecutionContext(
        batch_size=16,
        sequence_length=256,
        d_model=1024,
        num_heads=8,
        ffn_dim=4096,
        num_layers=6,
        causal=False,
        device=torch.device("cpu"),
        dtype=dtype,
        training=False,
        grad_enabled=False,
        input_contiguous=True,
        has_valid_token_mask=False,
        mask_compatible=True,
    )


def test_complete_policy_discards_an_unregistered_partial_plan(monkeypatch) -> None:
    monkeypatch.setattr(execution_plan, "supports_triton_qkv_layout", lambda **_: True)
    monkeypatch.setattr(execution_plan, "supports_wide_exact_gelu", lambda **_: False)

    plan = resolve_execution_plan(
        get_policy_spec("wide-triton-inplace"),
        _context(dtype=torch.bfloat16),
        requested_policy="wide-triton-inplace",
        dispatch_policy=None,
    )

    assert plan.selected_policy == "torch_fallback"
    assert plan.resolved_qkv_layout == "torch_zero_copy_view"
    assert plan.resolved_ffn == "torch_exact_gelu"
    assert set(plan.missing_components) == {
        "triton_qkv_layout",
        "wide_inplace_ffn",
    }
    assert "policy_requires_complete_application" in plan.fallback_reasons


def test_partial_policy_keeps_the_component_it_explicitly_allows(monkeypatch) -> None:
    monkeypatch.setattr(execution_plan, "supports_triton_qkv_layout", lambda **_: True)
    monkeypatch.setattr(
        execution_plan,
        "supports_triton_attention_softmax",
        lambda **_: False,
    )

    plan = resolve_execution_plan(
        get_policy_spec("triton"),
        _context(),
        requested_policy="triton",
        dispatch_policy=None,
    )

    assert plan.selected_policy == "triton_partial"
    assert plan.resolved_qkv_layout == "triton_single_pass"
    assert plan.missing_components == ("triton_attention_softmax",)
    assert "policy_requires_complete_application" not in plan.fallback_reasons
