"""Narrow, explicit program space for the streamed Shape 14 workload."""

from __future__ import annotations

from solution.config import (
    AttentionBackend,
    AttentionOutputBridge,
    ConfigSpec,
    FFNBackend,
    InitialNormBackend,
    PrecisionPlan,
    QKVMaterialization,
    ResidualNormBackend,
    RuntimeBackend,
)

from .search_space import (
    BranchSpace,
    ParameterDomain,
    PlanBuilderLike,
    SearchContext,
    StructureSpec,
)

_SHAPE14_TILES = ("32x64", "32x128", "64x64", "64x128")
_SHAPE14_WARPS = (4, 8)
_SHAPE14_MICROBATCHES = (1, 2)


def _portable_branch() -> BranchSpace:
    return BranchSpace(
        structure=StructureSpec(
            attention=AttentionBackend.REFERENCE_STREAMING,
            precision_plan=PrecisionPlan.INPUT_DTYPE,
            qkv_materialization=QKVMaterialization.VIEW,
            attention_output_bridge=AttentionOutputBridge.TORCH_BHSD_TO_BSD,
            ffn=FFNBackend.TORCH,
            residual_norm=ResidualNormBackend.TORCH,
            initial_norm=InitialNormBackend.TORCH,
            runtime=RuntimeBackend.STREAMED,
        ),
        domains=(
            ParameterDomain("projection_pattern", ("all_input",), "all_input"),
            ParameterDomain("microbatch_size", _SHAPE14_MICROBATCHES, 1),
        ),
        scope="streamed",
    )


def _native_sdpa_branch() -> BranchSpace:
    return BranchSpace(
        structure=StructureSpec(
            attention=AttentionBackend.CAUSAL_SDPA,
            precision_plan=PrecisionPlan.FP16_ATTENTION_AND_FFN_INPUT,
            qkv_materialization=QKVMaterialization.VIEW,
            attention_output_bridge=AttentionOutputBridge.TORCH_BHSD_TO_BSD,
            ffn=FFNBackend.TORCH,
            residual_norm=ResidualNormBackend.TORCH,
            initial_norm=InitialNormBackend.TORCH,
            runtime=RuntimeBackend.STREAMED,
        ),
        domains=(
            ParameterDomain("projection_pattern", ("all_shadow",), "all_shadow"),
            ParameterDomain("microbatch_size", _SHAPE14_MICROBATCHES, 1),
        ),
        scope="streamed",
    )


def _triton_dh64_branch(*, stages: int) -> BranchSpace:
    return BranchSpace(
        structure=StructureSpec(
            attention=AttentionBackend.TRITON_STREAMING_DH64,
            precision_plan=PrecisionPlan.FP16_ATTENTION_AND_FFN_INPUT,
            qkv_materialization=QKVMaterialization.VIEW,
            attention_output_bridge=AttentionOutputBridge.ATTENTION_DIRECT_BSD,
            ffn=FFNBackend.TORCH,
            residual_norm=ResidualNormBackend.TORCH,
            initial_norm=InitialNormBackend.TORCH,
            runtime=RuntimeBackend.STREAMED,
        ),
        domains=(
            ParameterDomain("projection_pattern", ("all_shadow",), "all_shadow"),
            ParameterDomain("attention_tile", _SHAPE14_TILES, "32x64"),
            ParameterDomain("attention_num_warps", _SHAPE14_WARPS, 4),
            ParameterDomain("attention_num_stages", (stages,), stages),
            ParameterDomain("microbatch_size", _SHAPE14_MICROBATCHES, 1),
        ),
        scope="streamed",
    )


def _shape14_branches() -> tuple[BranchSpace, ...]:
    return (
        _portable_branch(),
        _native_sdpa_branch(),
        _triton_dh64_branch(stages=2),
        _triton_dh64_branch(stages=3),
    )


class Shape14SearchSpace:
    """Four fixed Shape 14 branches with a deliberately narrow launch domain."""

    def __init__(
        self,
        *,
        plan_builder: PlanBuilderLike,
        context: SearchContext,
    ) -> None:
        self._validate_context(context)
        self.plan_builder = plan_builder
        self.context = context
        branches = tuple(
            branch
            for branch in _shape14_branches()
            if self.accepted(branch.default_config())
        )
        if not branches or not any(
            branch.structure.attention is AttentionBackend.REFERENCE_STREAMING
            for branch in branches
        ):
            raise ValueError("plan builder rejected the Shape 14 portable control")
        self.branches = branches
        self.mandatory_branch_ids = frozenset(branch.branch_id for branch in branches)
        self._by_id = {branch.branch_id: branch for branch in branches}

    @staticmethod
    def _validate_context(context: SearchContext) -> None:
        execution = context.execution_context
        shape = (
            context.batch_size,
            execution.seq_len,
            execution.d_model,
            execution.num_heads,
            execution.ffn_dim,
            execution.num_layers,
            execution.causal,
        )
        if context.scope != "streamed" or shape != (
            32,
            100000,
            1024,
            16,
            1024,
            2,
            True,
        ):
            raise ValueError(
                "Shape14SearchSpace requires the official streamed Shape 14"
            )
        if execution.has_valid_token_mask:
            raise ValueError("Shape 14 search requires an all-valid workload")

    def accepted(self, config: ConfigSpec) -> bool:
        result = self.plan_builder.evaluate(
            config,
            self.context.execution_context,
            self.context.hardware,
        )
        return bool(result.accepted)

    def branch(self, branch_id: str) -> BranchSpace:
        try:
            return self._by_id[branch_id]
        except KeyError as exc:
            raise KeyError(f"unknown branch_id: {branch_id}") from exc

    def branch_for(self, config: ConfigSpec) -> BranchSpace | None:
        for branch in self.branches:
            if branch.parameters_for(config) is not None:
                return branch
        return None


__all__ = ["Shape14SearchSpace"]
