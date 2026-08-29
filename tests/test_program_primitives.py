from __future__ import annotations

from collections.abc import Iterable

import pytest
import torch

import autotune.space as space_module
from autotune.space import BranchSpace, ParameterDomain, StructureSpec
from official.torch_transformer_benchmark import TransformerConfig
from solution.config import (
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
)
from solution.model import UserOptimizedTransformer

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
    PrecisionPlan.FP16_FFN_BRANCH: frozenset(
        {"ffn_input_projection", "ffn_output_projection"}
    ),
    PrecisionPlan.FP16_CORE: frozenset(_PROJECTION_FIELDS),
}


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
        (PrecisionPlan.FP16_FFN_BRANCH, "ffn_input_projection"),
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
    assert config.program.qkv_projection is ProjectionBackend.AUTOCAST_FP16
    assert config.program.attention_output_projection is ProjectionBackend.AUTOCAST_FP16
    assert config.program.ffn_input_projection is ProjectionBackend.FP16_SHADOW
    assert config.program.ffn_output_projection is ProjectionBackend.AUTOCAST_FP16


def test_structure_generation_prunes_unimplemented_attention_output_bridges() -> None:
    structures: Iterable[StructureSpec] = space_module._structure_specs("resident")
    saw_shape13_direct = False
    saw_shape13_torch = False

    for structure in structures:
        if structure.attention is AttentionBackend.TRITON_DH8:
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
        else:
            assert (
                structure.attention_output_bridge
                is AttentionOutputBridge.TORCH_BHSD_TO_BSD
            )

    assert saw_shape13_direct
    assert saw_shape13_torch


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
    assert signature["ffn_output_projection_compute_dtype"] == "float32"
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
