from __future__ import annotations

from types import SimpleNamespace

import torch

from autotune.evaluation import EvaluationScope
from autotune.search_engine import SearchBudget, SearchEngine, SearchRequest
from autotune.search_space import SearchContext
from autotune.shape14_search_space import Shape14SearchSpace
from autotune.study_storage import SearchStorage
from solution.config import (
    AttentionBackend,
    AttentionOutputBridge,
    FFNBackend,
    InitialNormBackend,
    PrecisionPlan,
    QKVMaterialization,
    ResidualNormBackend,
    RuntimeBackend,
)
from solution.plan import ExecutionContext
from solution.plan_builder import HardwareCapabilities, PlanBuilder


def _context() -> ExecutionContext:
    return ExecutionContext(
        batch_size=32,
        seq_len=100000,
        d_model=1024,
        num_heads=16,
        causal=True,
        device=torch.device("cuda"),
        dtype=torch.float32,
        training=False,
        grad_enabled=False,
        input_contiguous=True,
        has_valid_token_mask=False,
        mask_compatible=True,
        ffn_dim=1024,
        num_layers=2,
    )


class _AcceptingPlanBuilder:
    def evaluate(self, config, context, hardware=None):
        return SimpleNamespace(accepted=True)


def _hardware() -> HardwareCapabilities:
    return HardwareCapabilities(
        device_type="cuda",
        compute_capability=(8, 9),
        shared_memory_per_block=100_000,
        mem_efficient_sdp=True,
        cudnn_sdp=True,
        cudnn_available=True,
        torch_compile=True,
        triton_shape13_attention=True,
        triton_dh8_attention=True,
        triton_residual_norm=True,
        triton_mixed_residual_norm=True,
        triton_initial_norm=True,
        triton_exact_gelu=True,
        triton_streaming_dh64_attention=True,
    )


def test_shape14_space_contains_only_the_four_narrow_branches() -> None:
    space = Shape14SearchSpace(
        plan_builder=_AcceptingPlanBuilder(),
        context=SearchContext(execution_context=_context(), scope="streamed"),
    )

    assert len(space.branches) == 4
    assert space.mandatory_branch_ids == frozenset(
        branch.branch_id for branch in space.branches
    )
    assert tuple(branch.cardinality for branch in space.branches) == (2, 2, 16, 16)
    assert tuple(branch.structure.attention for branch in space.branches) == (
        AttentionBackend.REFERENCE_STREAMING,
        AttentionBackend.CAUSAL_SDPA,
        AttentionBackend.TRITON_STREAMING_DH64,
        AttentionBackend.TRITON_STREAMING_DH64,
    )

    portable, native, stage2, stage3 = space.branches
    assert portable.structure.precision_plan is PrecisionPlan.INPUT_DTYPE
    assert native.structure.precision_plan is (
        PrecisionPlan.FP16_ATTENTION_AND_FFN_INPUT
    )
    assert native.structure.attention_output_bridge is (
        AttentionOutputBridge.TORCH_BHSD_TO_BSD
    )
    assert {branch.domains[3].choices for branch in (stage2, stage3)} == {(2,), (3,)}

    for branch in space.branches:
        assert branch.structure.qkv_materialization is QKVMaterialization.VIEW
        assert branch.structure.ffn is FFNBackend.TORCH
        assert branch.structure.residual_norm is ResidualNormBackend.TORCH
        assert branch.structure.initial_norm is InitialNormBackend.TORCH
        assert branch.structure.runtime is RuntimeBackend.STREAMED
        assert branch.domains[-1].name == "microbatch_size"
        assert branch.domains[-1].choices == (1, 2)

    for branch in (stage2, stage3):
        assert branch.structure.precision_plan is (
            PrecisionPlan.FP16_ATTENTION_AND_FFN_INPUT
        )
        assert branch.structure.attention_output_bridge is (
            AttentionOutputBridge.ATTENTION_DIRECT_BSD
        )
        domains = {domain.name: domain.choices for domain in branch.domains}
        assert domains["projection_pattern"] == ("all_shadow",)
        assert domains["attention_tile"] == (
            "32x64",
            "32x128",
            "64x64",
            "64x128",
        )
        assert domains["attention_num_warps"] == (4, 8)


def test_every_shape14_candidate_passes_static_plan_compilation() -> None:
    builder = PlanBuilder()
    context = _context()
    hardware = _hardware()
    space = Shape14SearchSpace(
        plan_builder=builder,
        context=SearchContext(
            execution_context=context,
            scope="streamed",
            hardware=hardware,
        ),
    )

    candidates = tuple(
        branch.config_at(index)
        for branch in space.branches
        for index in range(branch.cardinality)
    )

    assert len(candidates) == 36
    assert all(
        builder.evaluate(item, context, hardware).accepted for item in candidates
    )


def test_search_engine_routes_streamed_requests_to_shape14_space(tmp_path) -> None:
    request = SearchRequest(
        case_id="official_14",
        execution_context=_context(),
        hardware=object(),
        scope=EvaluationScope.STREAMED,
        environment="test-environment",
        search_identity="search-v1",
        enhanced_identity="enhanced-v1",
        promotion_identity="promotion-v1",
        budget=SearchBudget(max_seconds=1.0),
        seed=7,
    )
    engine = SearchEngine(
        storage=SearchStorage(tmp_path),
        evaluator=object(),
        plan_builder=_AcceptingPlanBuilder(),
    )

    plan = engine.plan(request)

    assert isinstance(plan.search_space, Shape14SearchSpace)
    assert len(plan.search_space.branches) == 4
    assert len(plan.identities) == 4
