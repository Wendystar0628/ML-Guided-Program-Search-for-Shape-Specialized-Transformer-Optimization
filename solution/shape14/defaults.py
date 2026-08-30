"""Conservative executable fallback for the official Shape 14 workload."""

from __future__ import annotations

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
    TritonAttentionParams,
)


def conservative_streamed_config() -> ConfigSpec:
    """Return the measured low-risk Shape 14 program used before tuning."""

    return ConfigSpec(
        program=ProgramConfig(
            attention=AttentionBackend.TRITON_STREAMING_DH64,
            qkv_projection=ProjectionBackend.FP16_SHADOW,
            attention_output_projection=ProjectionBackend.FP16_SHADOW,
            ffn_input_projection=ProjectionBackend.FP16_SHADOW,
            ffn_output_projection=ProjectionBackend.INPUT_DTYPE,
            precision_plan=PrecisionPlan.FP16_ATTENTION_AND_FFN_INPUT,
            qkv_materialization=QKVMaterialization.VIEW,
            attention_output_bridge=AttentionOutputBridge.ATTENTION_DIRECT_BSD,
            ffn=FFNBackend.TORCH,
            residual_norm=ResidualNormBackend.TORCH,
            initial_norm=InitialNormBackend.TORCH,
        ),
        schedule=ScheduleConfig(
            runtime=RuntimeBackend.STREAMED,
            attention_launch=TritonAttentionParams(
                block_m=32,
                block_n=64,
                num_warps=4,
                num_stages=2,
            ),
            microbatch_size=1,
        ),
    )


__all__ = ["conservative_streamed_config"]
