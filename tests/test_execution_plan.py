from __future__ import annotations

import torch

from solution.execution_plan import ExecutionContext, resolve_execution_plan
from solution.policies import get_policy_spec


def _context(
    *,
    batch_size: int = 1,
    device: str = "cpu",
    training: bool = False,
    grad_enabled: bool = False,
) -> ExecutionContext:
    return ExecutionContext(
        batch_size=batch_size,
        sequence_length=128,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
        causal=True,
        device=torch.device(device),
        dtype=torch.float32,
        training=training,
        grad_enabled=grad_enabled,
        input_contiguous=True,
        has_valid_token_mask=True,
        mask_compatible=True,
    )


def _resolve(policy: str, context: ExecutionContext):
    return resolve_execution_plan(
        get_policy_spec(policy),
        context,
        requested_policy=policy,
        dispatch_policy=None,
    )


def test_causal_sdpa_plan_is_shape_independent_and_fully_observable() -> None:
    plan = _resolve("causal-sdpa", _context())

    assert plan.selected_policy == "causal-sdpa"
    assert plan.attention_backend == "causal_sdpa"
    assert plan.runtime_wrapper == "eager"
    assert plan.batch_strategy == "full"
    assert plan.block_backend == "torch"
    assert len(plan.layers) == 4
    assert plan.missing_components == ()


def test_explicit_graph_policy_falls_back_atomically_without_cuda() -> None:
    plan = _resolve("graph", _context(device="cpu"))

    assert plan.selected_policy == "safe"
    assert plan.runtime_wrapper == "eager"
    assert plan.resolved_components == ()
    assert plan.missing_components == ("cuda_graph",)


def test_batch_tiling_is_selected_from_capacity_not_a_case_id() -> None:
    plan = _resolve("batch-tiled", _context(batch_size=10_000))

    assert plan.selected_policy == "batch-tiled"
    assert plan.batch_strategy == "tiled"
    assert plan.batch_tile_size is not None
    assert 0 < plan.batch_tile_size < 10_000


def test_inplace_block_requires_inference_mode() -> None:
    inference = _resolve("inplace-block", _context())
    training = _resolve(
        "inplace-block",
        _context(training=True, grad_enabled=True),
    )

    assert inference.selected_policy == "inplace-block"
    assert inference.block_backend == "inplace_exact_gelu"
    assert training.selected_policy == "safe"
    assert training.block_backend == "torch"


def test_description_serializes_the_plan_consumed_by_forward() -> None:
    plan = _resolve("causal-sdpa", _context())

    evidence = plan.describe(
        dispatch_source=None,
        dispatch_table_sha256=None,
        dispatch_policy=None,
        route_origin="explicit",
        causal=True,
    )

    assert evidence["selected_policy"] == plan.selected_policy
    assert evidence["attention_backend"] == plan.attention_backend
    assert evidence["runtime_wrapper"] == plan.runtime_wrapper
    assert evidence["batch_strategy"] == plan.batch_strategy
    assert evidence["block_backend"] == plan.block_backend
    assert evidence["causal_mask"] == "implicit"
