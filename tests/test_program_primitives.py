from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import autotune.search_space as space_module
from autotune.search_engine import SearchBudget
from autotune.search_space import (
    DEFAULT_MAX_STRUCTURE_BRANCHES,
    BranchSpace,
    ParameterDomain,
    SearchContext,
    StructureSpec,
)
from official.torch_transformer_benchmark import TransformerConfig, compare_outputs
from solution.config import (
    COMPILED_FORWARD_MODES,
    DEFAULT_COMPILED_FORWARD_MODE,
    AttentionBackend,
    AttentionOutputBridge,
    ConfigSpec,
    FFNBackend,
    InitialNormBackend,
    PrecisionPlan,
    ProgramConfig,
    ProjectionBackend,
    QKVMaterialization,
    ResidualNormBackend,
    RuntimeBackend,
    ScheduleConfig,
    TritonAttentionParams,
    TritonNormParams,
)
from solution.plan import ExecutionContext
from solution.plan_builder import HardwareCapabilities, PlanBuilder
from solution.transformer import UserOptimizedTransformer

_PROJECTION_FIELDS = (
    "qkv_projection",
    "attention_output_projection",
    "ffn_input_projection",
    "ffn_output_projection",
)
_FP16_FIELDS = {
    PrecisionPlan.INPUT_DTYPE: frozenset(),
    PrecisionPlan.FP16_QKV_ATTENTION: frozenset({"qkv_projection"}),
    PrecisionPlan.FP16_ATTENTION_BRANCH: frozenset(
        {"qkv_projection", "attention_output_projection"}
    ),
    PrecisionPlan.FP16_FFN_INPUT_FP32_GELU: frozenset(
        {"ffn_input_projection"}
    ),
    PrecisionPlan.FP16_ATTENTION_AND_FFN_INPUT: frozenset(
        {
            "qkv_projection",
            "attention_output_projection",
            "ffn_input_projection",
        }
    ),
    PrecisionPlan.FP16_FFN_BRANCH: frozenset(
        {"ffn_input_projection", "ffn_output_projection"}
    ),
    PrecisionPlan.FP16_FFN_OUTPUT: frozenset({"ffn_output_projection"}),
    PrecisionPlan.FP16_CORE: frozenset(_PROJECTION_FIELDS),
}


def test_default_structure_budget_remains_bounded() -> None:
    assert SearchBudget(max_seconds=1.0).max_structure_branches == 36
    assert DEFAULT_MAX_STRUCTURE_BRANCHES == 36


def _selection_branch(
    *,
    attention: AttentionBackend,
    qkv_materialization: QKVMaterialization,
    ffn: FFNBackend,
    runtime: RuntimeBackend,
) -> BranchSpace:
    return BranchSpace(
        structure=StructureSpec(
            attention=attention,
            precision_plan=PrecisionPlan.INPUT_DTYPE,
            qkv_materialization=qkv_materialization,
            attention_output_bridge=AttentionOutputBridge.TORCH_BHSD_TO_BSD,
            ffn=ffn,
            residual_norm=ResidualNormBackend.TORCH,
            initial_norm=InitialNormBackend.TORCH,
            runtime=runtime,
        ),
        domains=(),
        scope="resident",
    )


def test_structure_selection_covers_values_then_rotates_seeded_exploration() -> None:
    candidates = tuple(
        _selection_branch(
            attention=attention,
            qkv_materialization=qkv_materialization,
            ffn=ffn,
            runtime=runtime,
        )
        for attention, qkv_materialization, ffn, runtime in itertools.product(
            (
                AttentionBackend.CAUSAL_SDPA,
                AttentionBackend.FP16_EFFICIENT_SDPA,
                AttentionBackend.FP16_CUDNN_SDPA,
            ),
            (QKVMaterialization.VIEW, QKVMaterialization.CONTIGUOUS),
            (FFNBackend.TORCH, FFNBackend.COMPILED),
            (RuntimeBackend.EAGER, RuntimeBackend.CUDA_GRAPH),
        )
    )
    required = (candidates[0],)

    first, mandatory = space_module._select_structure_branches(
        candidates,
        required=required,
        limit=12,
        seed=17,
    )
    repeated, repeated_mandatory = space_module._select_structure_branches(
        candidates,
        required=required,
        limit=12,
        seed=17,
    )
    rotated, rotated_mandatory = space_module._select_structure_branches(
        candidates,
        required=required,
        limit=12,
        seed=18,
    )

    universe = set().union(
        *(space_module._primitive_tokens(branch) for branch in candidates)
    )
    mandatory_branches = tuple(
        branch for branch in first if branch.branch_id in mandatory
    )
    covered = set().union(
        *(space_module._primitive_tokens(branch) for branch in mandatory_branches)
    )
    pairwise_end = len(mandatory) + (12 - len(mandatory)) // 2

    assert required[0].branch_id in mandatory
    assert covered == universe
    assert tuple(branch.branch_id for branch in repeated) == tuple(
        branch.branch_id for branch in first
    )
    assert repeated_mandatory == mandatory == rotated_mandatory
    assert tuple(branch.branch_id for branch in rotated[:pairwise_end]) == tuple(
        branch.branch_id for branch in first[:pairwise_end]
    )
    assert {branch.branch_id for branch in rotated[pairwise_end:]} != {
        branch.branch_id for branch in first[pairwise_end:]
    }


def test_program_space_uses_a_cheap_legal_witness_when_default_is_rejected(
    monkeypatch,
) -> None:
    structure = StructureSpec(
        attention=AttentionBackend.CAUSAL_SDPA,
        precision_plan=PrecisionPlan.INPUT_DTYPE,
        qkv_materialization=QKVMaterialization.VIEW,
        attention_output_bridge=AttentionOutputBridge.TORCH_BHSD_TO_BSD,
        ffn=FFNBackend.TORCH,
        residual_norm=ResidualNormBackend.TORCH,
        initial_norm=InitialNormBackend.TORCH,
        runtime=RuntimeBackend.CUDA_GRAPH,
    )
    evaluated: list[ConfigSpec] = []

    class _PlanBuilder:
        def evaluate(self, config, context, hardware=None):
            evaluated.append(config)
            return SimpleNamespace(
                accepted=config.schedule.reuse_unchanged_input,
            )

    monkeypatch.setattr(
        space_module,
        "_structure_specs",
        lambda context: (structure,),
    )
    space = space_module.ProgramSearchSpace(
        plan_builder=_PlanBuilder(),
        context=SearchContext(
            execution_context=_official07_context(),
            scope="resident",
        ),
        max_branches=1,
        seed=9,
    )

    assert len(evaluated) == 2
    assert len(space.branches) == 1
    assert space.branches[0].default_config().schedule.reuse_unchanged_input


def test_compiled_forward_search_exposes_all_supported_modes() -> None:
    assert DEFAULT_COMPILED_FORWARD_MODE == "max-autotune"
    assert COMPILED_FORWARD_MODES == {
        "max-autotune",
        "max-autotune-no-cudagraphs",
        "reduce-overhead",
    }
    for mode in COMPILED_FORWARD_MODES:
        schedule = ScheduleConfig(
            runtime=RuntimeBackend.COMPILED_FORWARD,
            compile_mode=mode,
        )
        assert schedule.compile_mode == mode


def _program(
    precision_plan: PrecisionPlan = PrecisionPlan.INPUT_DTYPE,
    *,
    fp16_backend: ProjectionBackend = ProjectionBackend.FP16_SHADOW,
    attention: AttentionBackend = AttentionBackend.CAUSAL_SDPA,
    qkv_materialization: QKVMaterialization = QKVMaterialization.VIEW,
    attention_output_bridge: AttentionOutputBridge = (
        AttentionOutputBridge.TORCH_BHSD_TO_BSD
    ),
) -> ProgramConfig:
    active = _FP16_FIELDS[precision_plan]
    projections = {
        field_name: (
            fp16_backend if field_name in active else ProjectionBackend.INPUT_DTYPE
        )
        for field_name in _PROJECTION_FIELDS
    }
    return ProgramConfig(
        attention=attention,
        **projections,
        precision_plan=precision_plan,
        qkv_materialization=qkv_materialization,
        attention_output_bridge=attention_output_bridge,
        ffn=FFNBackend.TORCH,
        residual_norm=ResidualNormBackend.TORCH,
        initial_norm=InitialNormBackend.TORCH,
    )


def _official07_context(*, has_valid_token_mask: bool = False) -> ExecutionContext:
    return ExecutionContext(
        batch_size=64,
        seq_len=128,
        d_model=32,
        num_heads=4,
        causal=True,
        device=torch.device("cuda"),
        dtype=torch.float32,
        training=False,
        grad_enabled=False,
        input_contiguous=True,
        has_valid_token_mask=has_valid_token_mask,
        mask_compatible=True,
        ffn_dim=32,
        num_layers=4,
    )


@pytest.mark.parametrize("precision_plan", tuple(PrecisionPlan))
def test_precision_plans_bind_exactly_their_projection_roles(
    precision_plan: PrecisionPlan,
) -> None:
    program = _program(precision_plan)

    for field_name in _PROJECTION_FIELDS:
        expected = (
            ProjectionBackend.FP16_SHADOW
            if field_name in _FP16_FIELDS[precision_plan]
            else ProjectionBackend.INPUT_DTYPE
        )
        assert getattr(program, field_name) is expected

    config = ConfigSpec(
        program=program,
        schedule=ScheduleConfig(runtime=RuntimeBackend.EAGER),
    )
    restored = ConfigSpec.from_dict(config.to_dict())
    assert restored == config
    assert restored.config_id == config.config_id


@pytest.mark.parametrize(
    ("precision_plan", "invalid_field"),
    (
        (PrecisionPlan.INPUT_DTYPE, "qkv_projection"),
        (PrecisionPlan.FP16_QKV_ATTENTION, "qkv_projection"),
        (
            PrecisionPlan.FP16_ATTENTION_BRANCH,
            "attention_output_projection",
        ),
        (
            PrecisionPlan.FP16_FFN_INPUT_FP32_GELU,
            "ffn_input_projection",
        ),
        (
            PrecisionPlan.FP16_ATTENTION_AND_FFN_INPUT,
            "ffn_input_projection",
        ),
        (PrecisionPlan.FP16_FFN_BRANCH, "ffn_input_projection"),
        (PrecisionPlan.FP16_FFN_OUTPUT, "ffn_output_projection"),
        (PrecisionPlan.FP16_CORE, "ffn_output_projection"),
    ),
)
def test_precision_plans_reject_one_role_outside_their_cast_graph(
    precision_plan: PrecisionPlan,
    invalid_field: str,
) -> None:
    valid = _program(precision_plan).to_dict()
    valid[invalid_field] = (
        ProjectionBackend.INPUT_DTYPE.value
        if invalid_field in _FP16_FIELDS[precision_plan]
        else ProjectionBackend.FP16_SHADOW.value
    )

    with pytest.raises(ValueError, match=invalid_field):
        ProgramConfig.from_dict(valid)


def test_branch_build_and_parameter_recovery_preserve_new_program_primitives() -> None:
    structure = StructureSpec(
        attention=AttentionBackend.CAUSAL_SDPA,
        precision_plan=PrecisionPlan.FP16_CORE,
        qkv_materialization=QKVMaterialization.CONTIGUOUS,
        attention_output_bridge=AttentionOutputBridge.TORCH_BHSD_TO_BSD,
        ffn=FFNBackend.TORCH,
        residual_norm=ResidualNormBackend.TORCH,
        initial_norm=InitialNormBackend.TORCH,
        runtime=RuntimeBackend.EAGER,
    )
    patterns = (
        "all_autocast",
        "all_shadow",
        "shadow_qkv_projection",
        "shadow_attention_output_projection",
        "shadow_ffn_input_projection",
        "shadow_ffn_output_projection",
    )
    branch = BranchSpace(
        structure=structure,
        domains=(
            ParameterDomain(
                "projection_pattern",
                patterns,
                default="all_autocast",
            ),
        ),
        scope="resident",
    )
    parameters = {"projection_pattern": "shadow_ffn_input_projection"}

    config = branch.build(parameters)

    assert StructureSpec.from_config(config) == structure
    assert branch.parameters_for(config) == parameters

    decoded = tuple(branch.config_at(index) for index in range(branch.cardinality))
    assert len({candidate.config_id for candidate in decoded}) == branch.cardinality
    assert tuple(branch.index_for(candidate) for candidate in decoded) == tuple(
        range(branch.cardinality)
    )
    assert config.program.qkv_projection is ProjectionBackend.AUTOCAST_FP16
    assert config.program.attention_output_projection is ProjectionBackend.AUTOCAST_FP16
    assert config.program.ffn_input_projection is ProjectionBackend.FP16_SHADOW
    assert config.program.ffn_output_projection is ProjectionBackend.AUTOCAST_FP16


def test_generated_projection_patterns_cover_every_legal_weight_implementation() -> (
    None
):
    patterns = space_module._projection_patterns(PrecisionPlan.FP16_CORE)

    assert len(patterns) == 16
    assert len({values for _, values in patterns}) == 16
    assert patterns[0][0] == "all_autocast"
    assert patterns[-1][0] == "all_shadow"
    assert any(
        name
        == "shadow_qkv_projection__attention_output_projection__ffn_input_projection"
        for name, _ in patterns
    )


@pytest.mark.parametrize(
    ("precision_plan", "expected_count"),
    (
        (PrecisionPlan.FP16_FFN_INPUT_FP32_GELU, 2),
        (PrecisionPlan.FP16_ATTENTION_AND_FFN_INPUT, 8),
    ),
)
def test_new_precision_plans_generate_every_projection_implementation(
    precision_plan: PrecisionPlan,
    expected_count: int,
) -> None:
    patterns = space_module._projection_patterns(precision_plan)

    assert len(patterns) == expected_count
    assert len({values for _, values in patterns}) == expected_count


def test_generated_schedule_domains_cover_supported_shapes_without_padding_small_widths() -> (
    None
):
    d32 = space_module._gemm_tile_choices(
        output_width=32,
        input_width=32,
        supported_block_k=(16, 32, 64),
    )
    d128 = space_module._gemm_tile_choices(
        output_width=128,
        input_width=128,
        supported_block_k=(16, 32, 64),
    )
    ffn = space_module._gemm_tile_choices(
        output_width=128,
        input_width=128,
        supported_block_k=(16, 32, 64, 128),
    )

    assert len(d32) == 16
    assert all(int(tile.split("x")[1]) <= 32 for tile in d32)
    assert all(int(tile.split("x")[2]) <= 32 for tile in d32)
    assert len(d128) == 48
    assert "16x16x16" in d128
    assert "128x128x64" in d128
    assert len(ffn) == 64
    assert "128x128x128" in ffn
    assert space_module._batch_tile_choices(1) == ()
    assert space_module._batch_tile_choices(64) == (1, 2, 4, 8, 16, 32)
    assert space_module._batch_tile_choices(10000)[-1] == 8192
    assert space_module._streaming_attention_tile_choices() == (
        "16x64",
        "16x128",
        "32x64",
        "32x128",
        "64x64",
        "64x128",
    )


def test_structure_generation_prunes_only_statically_impossible_combinations() -> None:
    masked = SearchContext(
        execution_context=_official07_context(has_valid_token_mask=True),
        scope="resident",
    )
    structures = space_module._structure_specs(masked)

    assert structures
    assert all(
        item.runtime is not RuntimeBackend.BATCH_TILED_CUDA_GRAPH for item in structures
    )
    assert all(
        item.attention
        in {
            AttentionBackend.REFERENCE_STREAMING,
            AttentionBackend.CAUSAL_SDPA,
        }
        for item in structures
    )
    assert all(
        item.residual_norm is not ResidualNormBackend.COMPILED for item in structures
    )
    assert all(
        item.runtime is not RuntimeBackend.COMPILED_FORWARD
        or (
            item.residual_norm is ResidualNormBackend.TORCH
            and item.initial_norm is InitialNormBackend.TORCH
        )
        for item in structures
    )


def test_streamed_shape14_search_compares_library_and_custom_attention() -> None:
    context = SearchContext(
        execution_context=ExecutionContext(
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
        ),
        scope="streamed",
    )

    attentions = {item.attention for item in space_module._structure_specs(context)}

    assert AttentionBackend.REFERENCE_STREAMING in attentions
    assert AttentionBackend.CAUSAL_SDPA in attentions
    assert AttentionBackend.TRITON_STREAMING_DH64 in attentions


def test_batch_tiled_d32_norm_domain_is_valid_for_the_inner_batch_template() -> None:
    context = SearchContext(
        execution_context=_official07_context(),
        scope="resident",
    )
    structure = StructureSpec(
        attention=AttentionBackend.CAUSAL_SDPA,
        precision_plan=PrecisionPlan.INPUT_DTYPE,
        qkv_materialization=QKVMaterialization.VIEW,
        attention_output_bridge=AttentionOutputBridge.TORCH_BHSD_TO_BSD,
        ffn=FFNBackend.TORCH,
        residual_norm=ResidualNormBackend.TRITON,
        initial_norm=InitialNormBackend.TORCH,
        runtime=RuntimeBackend.BATCH_TILED_CUDA_GRAPH,
    )
    domains = {
        domain.name: domain
        for domain in space_module._domains_for_structure(structure, context)
    }

    assert domains["residual_block_rows"].choices == (4, 8)
    assert domains["residual_num_warps"].choices == (2, 4, 8)
    assert 16 not in domains["residual_block_rows"].choices


def test_exact_d32_plan_rejects_a_warp_count_the_specialized_kernel_cannot_run() -> (
    None
):
    program = replace(
        _program(PrecisionPlan.FP16_CORE),
        residual_norm=ResidualNormBackend.TRITON,
    )
    config = ConfigSpec(
        program=program,
        schedule=ScheduleConfig(
            runtime=RuntimeBackend.EAGER,
            residual_norm_launch=TritonNormParams(block_rows=4, num_warps=1),
        ),
    )
    context = _official07_context()
    hardware = replace(
        HardwareCapabilities.detect(torch.device("cuda")),
        triton_residual_norm=True,
        triton_d32_residual_norm=True,
    )

    result = PlanBuilder().evaluate(config, context, hardware)

    assert not result.accepted
    assert any(
        violation.code == "unsupported_launch_value" for violation in result.violations
    )


def test_structure_generation_prunes_unimplemented_attention_output_bridges() -> None:
    context = SearchContext(
        execution_context=ExecutionContext(
            batch_size=64,
            seq_len=1024,
            d_model=128,
            num_heads=4,
            causal=True,
            device=torch.device("cuda"),
            dtype=torch.float32,
            training=False,
            grad_enabled=False,
            input_contiguous=True,
            has_valid_token_mask=False,
            mask_compatible=True,
            ffn_dim=128,
            num_layers=4,
        ),
        scope="resident",
    )
    structures: Iterable[StructureSpec] = space_module._structure_specs(context)
    saw_shape13_direct = False
    saw_shape13_torch = False
    saw_shape13_fused_projection = False

    for structure in structures:
        if structure.attention in {
            AttentionBackend.TRITON_DH8,
            AttentionBackend.TRITON_STREAMING_DH64,
        }:
            assert (
                structure.attention_output_bridge
                is AttentionOutputBridge.ATTENTION_DIRECT_BSD
            )
        elif structure.attention is AttentionBackend.TRITON_SHAPE13:
            saw_shape13_direct |= (
                structure.attention_output_bridge
                is AttentionOutputBridge.ATTENTION_DIRECT_BSD
            )
            saw_shape13_torch |= (
                structure.attention_output_bridge
                is AttentionOutputBridge.TORCH_BHSD_TO_BSD
            )
            saw_shape13_fused_projection |= (
                structure.attention_output_bridge
                is AttentionOutputBridge.TRITON_BHSD_PROJECTION
            )
        else:
            assert structure.attention_output_bridge in {
                AttentionOutputBridge.TORCH_BHSD_TO_BSD,
                AttentionOutputBridge.TRITON_BHSD_PROJECTION,
            }

    assert saw_shape13_direct
    assert saw_shape13_torch
    assert saw_shape13_fused_projection


def _tiny_model() -> UserOptimizedTransformer:
    model = UserOptimizedTransformer(
        TransformerConfig(
            batch_size=1,
            seq_len=4,
            d_model=8,
            num_heads=2,
            ffn_dim=16,
            num_layers=1,
            causal=True,
        )
    )
    return model.eval()


def _assert_new_signature_fields(signature: dict[str, object]) -> None:
    assert signature["qkv_projection_backend"] == "input_dtype"
    assert signature["attention_output_projection_backend"] == "input_dtype"
    assert signature["ffn_input_projection_backend"] == "input_dtype"
    assert signature["ffn_output_projection_backend"] == "input_dtype"
    assert signature["precision_plan"] == "input_dtype"
    assert signature["qkv_materialization"] == "contiguous"
    assert signature["attention_output_bridge"] == "torch_bhsd_to_bsd"
    assert signature["attention_output_layout"] == "bhsd"
    assert signature["ffn_backend"] == "torch"
    assert signature["qkv_projection_compute_dtype"] == "float32"
    assert signature["attention_output_projection_compute_dtype"] == "float32"
    assert signature["ffn_input_projection_compute_dtype"] == "float32"
    assert signature["ffn_activation_output_dtype"] == "float32"
    assert signature["ffn_output_projection_compute_dtype"] == "float32"
    assert signature["ffn_launch"] is None
    assert signature["qkv_projection_calls"] == 1
    assert signature["qkv_materialization_calls"] == 1
    assert signature["attention_output_bridge_calls"] == 1
    assert signature["attention_output_projection_calls"] == 1
    assert signature["ffn_calls"] == 1
    assert signature["ffn_input_projection_calls"] == 1
    assert signature["ffn_output_projection_calls"] == 1
    assert signature["complete"] is True


def test_execution_plan_and_observation_report_the_real_new_primitive_path() -> None:
    config = ConfigSpec(
        program=_program(qkv_materialization=QKVMaterialization.CONTIGUOUS),
        schedule=ScheduleConfig(runtime=RuntimeBackend.EAGER),
    )
    model = _tiny_model()
    model.configure_execution(config=config)
    model.set_execution_observation(True)

    preview = model.describe_execution_path()
    assert preview["observed_execution_signature"] is None
    _assert_new_signature_fields(preview["expected_execution_signature"])

    value = torch.randn(1, 4, 8)
    mask = torch.ones(1, 4, dtype=torch.bool)
    with torch.inference_mode():
        output = model(value, mask)

    description = model.describe_execution_path()
    expected = description["expected_execution_signature"]
    actual = description["observed_execution_signature"]
    assert output.shape == value.shape
    assert actual == expected
    _assert_new_signature_fields(actual)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fp16_core_projection_plan_has_a_real_cuda_execution_signature() -> None:
    config = ConfigSpec(
        program=_program(PrecisionPlan.FP16_CORE),
        schedule=ScheduleConfig(runtime=RuntimeBackend.EAGER),
    )
    model = _tiny_model().cuda()
    model.configure_execution(config=config)
    model.set_execution_observation(True)
    value = torch.randn(1, 4, 8, device="cuda")
    mask = torch.ones(1, 4, dtype=torch.bool, device="cuda")

    with torch.inference_mode():
        output = model(value, mask)

    description = model.describe_execution_path()
    actual = description["observed_execution_signature"]
    assert output.shape == value.shape
    assert actual == description["expected_execution_signature"]
    assert actual["precision_plan"] == PrecisionPlan.FP16_CORE.value
    assert actual["qkv_projection_compute_dtype"] == "float16"
    assert actual["attention_output_projection_compute_dtype"] == "float16"
    assert actual["ffn_input_projection_compute_dtype"] == "float16"
    assert actual["ffn_output_projection_compute_dtype"] == "float16"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "precision_plan",
    (
        PrecisionPlan.FP16_FFN_INPUT_FP32_GELU,
        PrecisionPlan.FP16_ATTENTION_AND_FFN_INPUT,
    ),
)
def test_new_precision_plans_restore_fp32_before_exact_gelu(
    precision_plan: PrecisionPlan,
) -> None:
    reference = _tiny_model().cuda()
    candidate = _tiny_model().cuda()
    candidate.load_state_dict(reference.state_dict())
    reference.configure_execution(
        config=ConfigSpec(
            program=_program(PrecisionPlan.INPUT_DTYPE),
            schedule=ScheduleConfig(runtime=RuntimeBackend.EAGER),
        )
    )
    candidate.configure_execution(
        config=ConfigSpec(
            program=_program(precision_plan),
            schedule=ScheduleConfig(runtime=RuntimeBackend.EAGER),
        )
    )
    candidate.set_execution_observation(True)
    value = torch.randn(1, 4, 8, device="cuda")
    mask = torch.ones(1, 4, dtype=torch.bool, device="cuda")

    with torch.inference_mode():
        expected_output = reference(value, mask)
        actual_output = candidate(value, mask)

    accuracy = compare_outputs(
        expected_output,
        actual_output,
        rtol=0.02,
        atol=0.002,
    )
    description = candidate.describe_execution_path()
    actual = description["observed_execution_signature"]
    assert accuracy.passed
    assert actual == description["expected_execution_signature"]
    assert actual["precision_plan"] == precision_plan.value
    assert actual["ffn_input_projection_compute_dtype"] == "float16"
    assert actual["ffn_activation_output_dtype"] == "float32"
    assert actual["ffn_output_projection_compute_dtype"] == "float32"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_shape14_online_attention_compiles_for_an_eager_microbatch() -> None:
    config = ConfigSpec(
        program=_program(
            PrecisionPlan.FP16_ATTENTION_BRANCH,
            attention=AttentionBackend.TRITON_STREAMING_DH64,
            qkv_materialization=QKVMaterialization.CONTIGUOUS,
            attention_output_bridge=AttentionOutputBridge.ATTENTION_DIRECT_BSD,
        ),
        schedule=ScheduleConfig(
            runtime=RuntimeBackend.EAGER,
            attention_launch=TritonAttentionParams(64, 64, 4, 2),
        ),
    )
    context = ExecutionContext(
        batch_size=1,
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

    plan = PlanBuilder().build(
        config,
        context,
        HardwareCapabilities.detect(torch.device("cuda")),
    )

    assert plan.attention_backend is AttentionBackend.TRITON_STREAMING_DH64
    assert plan.runtime_backend is RuntimeBackend.EAGER
