"""Execution evidence must describe branches that actually ran."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from official import torch_transformer_benchmark as official
from runner.candidates import CANDIDATE_SPECS
from solution import transformer as transformer_module
from solution.kernels import ffn


def _config() -> official.TransformerConfig:
    return official.TransformerConfig(
        batch_size=1,
        seq_len=4,
        d_model=8,
        num_heads=2,
        ffn_dim=8,
        num_layers=2,
        causal=True,
    )


def test_linear_exact_gelu_reports_guard_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setattr(ffn, "_ATEN_GELU_INPLACE", None)
    value = torch.randn(3, 5)
    weight = torch.randn(7, 5)
    bias = torch.randn(7)

    with torch.inference_mode():
        actual, backend = ffn.linear_exact_gelu(value, weight, bias)

    expected = F.gelu(F.linear(value, weight, bias), approximate="none")
    assert backend == "torch"
    torch.testing.assert_close(actual, expected)


def test_inplace_plan_and_helper_share_one_runtime_support_guard(
    monkeypatch,
) -> None:
    monkeypatch.setattr(ffn, "_ATEN_GELU_INPLACE", None)
    model = transformer_module.UserOptimizedTransformer(_config()).eval()
    model.configure_runtime_policy(policy="inplace-block")
    model.set_execution_observation(True)

    with torch.inference_mode():
        model(torch.randn(1, 4, 8), torch.ones(1, 4, dtype=torch.bool))

    path = model.describe_execution_path()
    observed = path["observed_execution"]
    assert path["selected_policy"] == "safe"
    assert path["block_backend"] == "torch"
    assert path["missing_components"] == ["inplace_exact_gelu"]
    assert observed["complete"] is True
    assert observed["block_backends"] == ["torch", "torch"]
    assert not CANDIDATE_SPECS["inplace-block"].evidence.matches(
        solution_policy="inplace-block",
        execution_path=path,
    )


def test_inplace_policy_evidence_rejects_an_unexpected_torch_fallback(
    monkeypatch,
) -> None:
    def fallback_linear_exact_gelu(
        value: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, str]:
        hidden = F.linear(value, weight, bias)
        return F.gelu(hidden, approximate="none"), "torch"

    monkeypatch.setattr(
        transformer_module,
        "linear_exact_gelu",
        fallback_linear_exact_gelu,
    )
    model = transformer_module.UserOptimizedTransformer(_config()).eval()
    model.configure_runtime_policy(policy="inplace-block")
    model.set_execution_observation(True)

    with torch.inference_mode():
        model(torch.randn(1, 4, 8), torch.ones(1, 4, dtype=torch.bool))

    path = model.describe_execution_path()
    observed = path["observed_execution"]
    assert path["block_backend"] == "inplace_exact_gelu"
    assert observed["complete"] is True
    assert observed["block_backends"] == ["torch", "torch"]
    assert not CANDIDATE_SPECS["inplace-block"].evidence.matches(
        solution_policy="inplace-block",
        execution_path=path,
    )
