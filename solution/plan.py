"""Immutable execution plans built from typed Transformer configurations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from .config import (
    AttentionBackend,
    AttentionOutputBridge,
    AttentionOutputLayout,
    ConfigSpec,
    FFNBackend,
    InitialNormBackend,
    PrecisionPlan,
    ProjectionBackend,
    QKVMaterialization,
    ResidualNormBackend,
    RuntimeBackend,
    TritonAttentionParams,
    TritonFFNParams,
    TritonGemmParams,
    TritonNormParams,
    TritonQKVParams,
)


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Input and model facts that affect execution-plan eligibility."""

    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    causal: bool
    device: torch.device
    dtype: torch.dtype
    training: bool
    grad_enabled: bool
    input_contiguous: bool
    has_valid_token_mask: bool
    mask_compatible: bool
    ffn_dim: int | None = None
    num_layers: int | None = None

    @property
    def head_dim(self) -> int | None:
        if self.num_heads <= 0 or self.d_model % self.num_heads:
            return None
        return self.d_model // self.num_heads

    @property
    def dtype_name(self) -> str:
        return str(self.dtype).removeprefix("torch.")

    @property
    def inference(self) -> bool:
        return not self.training and not self.grad_enabled

    def with_batch_size(self, batch_size: int) -> ExecutionContext:
        """Return the inner context seen by a batch-tiled schedule."""

        return replace(self, batch_size=batch_size)


@dataclass(frozen=True, slots=True)
class ExpectedExecutionTrace:
    """Generic path evidence derived from the same plan consumed by forward."""

    runtime_backend: RuntimeBackend
    attention_backend: AttentionBackend
    qkv_projection_backend: ProjectionBackend
    attention_output_projection_backend: ProjectionBackend
    ffn_input_projection_backend: ProjectionBackend
    ffn_output_projection_backend: ProjectionBackend
    precision_plan: PrecisionPlan
    qkv_materialization: QKVMaterialization
    attention_output_bridge: AttentionOutputBridge
    attention_output_layout: AttentionOutputLayout
    ffn_backend: FFNBackend
    residual_norm_backend: ResidualNormBackend
    initial_norm_backend: InitialNormBackend
    attention_compute_dtype: str
    qkv_projection_compute_dtype: str
    attention_output_projection_compute_dtype: str
    ffn_input_projection_compute_dtype: str
    ffn_activation_output_dtype: str
    ffn_output_projection_compute_dtype: str
    attention_calls: int
    qkv_projection_calls: int
    qkv_materialization_calls: int
    attention_output_bridge_calls: int
    attention_output_projection_calls: int
    ffn_calls: int
    ffn_input_projection_calls: int
    ffn_output_projection_calls: int
    residual_norm_calls: int
    initial_norm_calls: int
    runtime_calls: int
    attention_launch: TritonAttentionParams | None = None
    qkv_launch: TritonQKVParams | None = None
    attention_output_projection_launch: TritonGemmParams | None = None
    residual_norm_launch: TritonNormParams | None = None
    initial_norm_launch: TritonNormParams | None = None
    ffn_launch: TritonFFNParams | None = None
    ffn_input_launch: TritonGemmParams | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_backend": self.runtime_backend.value,
            "attention_backend": self.attention_backend.value,
            "qkv_projection_backend": self.qkv_projection_backend.value,
            "attention_output_projection_backend": (
                self.attention_output_projection_backend.value
            ),
            "ffn_input_projection_backend": (self.ffn_input_projection_backend.value),
            "ffn_output_projection_backend": (self.ffn_output_projection_backend.value),
            "precision_plan": self.precision_plan.value,
            "qkv_materialization": self.qkv_materialization.value,
            "attention_output_bridge": self.attention_output_bridge.value,
            "attention_output_layout": self.attention_output_layout.value,
            "ffn_backend": self.ffn_backend.value,
            "residual_norm_backend": self.residual_norm_backend.value,
            "initial_norm_backend": self.initial_norm_backend.value,
            "attention_compute_dtype": self.attention_compute_dtype,
            "qkv_projection_compute_dtype": (self.qkv_projection_compute_dtype),
            "attention_output_projection_compute_dtype": (
                self.attention_output_projection_compute_dtype
            ),
            "ffn_input_projection_compute_dtype": (
                self.ffn_input_projection_compute_dtype
            ),
            "ffn_activation_output_dtype": self.ffn_activation_output_dtype,
            "ffn_output_projection_compute_dtype": (
                self.ffn_output_projection_compute_dtype
            ),
            "attention_calls": self.attention_calls,
            "qkv_projection_calls": self.qkv_projection_calls,
            "qkv_materialization_calls": self.qkv_materialization_calls,
            "attention_output_bridge_calls": self.attention_output_bridge_calls,
            "attention_output_projection_calls": (
                self.attention_output_projection_calls
            ),
            "ffn_calls": self.ffn_calls,
            "ffn_input_projection_calls": self.ffn_input_projection_calls,
            "ffn_output_projection_calls": self.ffn_output_projection_calls,
            "residual_norm_calls": self.residual_norm_calls,
            "initial_norm_calls": self.initial_norm_calls,
            "runtime_calls": self.runtime_calls,
            "attention_launch": (
                None
                if self.attention_launch is None
                else self.attention_launch.to_dict()
            ),
            "qkv_launch": (
                None if self.qkv_launch is None else self.qkv_launch.to_dict()
            ),
            "attention_output_projection_launch": (
                None
                if self.attention_output_projection_launch is None
                else self.attention_output_projection_launch.to_dict()
            ),
            "residual_norm_launch": (
                None
                if self.residual_norm_launch is None
                else self.residual_norm_launch.to_dict()
            ),
            "initial_norm_launch": (
                None
                if self.initial_norm_launch is None
                else self.initial_norm_launch.to_dict()
            ),
            "ffn_launch": (
                None if self.ffn_launch is None else self.ffn_launch.to_dict()
            ),
            "ffn_input_launch": (
                None
                if self.ffn_input_launch is None
                else self.ffn_input_launch.to_dict()
            ),
            "complete": True,
        }


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Strict plan that forward executes without hidden fallback."""

    config: ConfigSpec
    outer_context: ExecutionContext
    inner_context: ExecutionContext
    attention_backend: AttentionBackend
    attention_compute_dtype: str
    qkv_projection_backend: ProjectionBackend
    qkv_projection_compute_dtype: str
    qkv_materialization: QKVMaterialization
    attention_output_bridge: AttentionOutputBridge
    attention_output_layout: AttentionOutputLayout
    attention_output_projection_backend: ProjectionBackend
    attention_output_projection_compute_dtype: str
    ffn_backend: FFNBackend
    ffn_input_projection_backend: ProjectionBackend
    ffn_input_projection_compute_dtype: str
    ffn_activation_output_dtype: str
    ffn_output_projection_backend: ProjectionBackend
    ffn_output_projection_compute_dtype: str
    precision_plan: PrecisionPlan
    residual_norm_backend: ResidualNormBackend
    initial_norm_backend: InitialNormBackend
    runtime_backend: RuntimeBackend
    compile_mode: str | None
    batch_tile_size: int | None
    microbatch_size: int | None
    reuse_unchanged_input: bool
    attention_launch: TritonAttentionParams | None
    qkv_launch: TritonQKVParams | None
    attention_output_projection_launch: TritonGemmParams | None
    residual_norm_launch: TritonNormParams | None
    initial_norm_launch: TritonNormParams | None
    ffn_launch: TritonFFNParams | None
    ffn_input_launch: TritonGemmParams | None
    use_d32_residual_norm: bool
    use_masked_residual_norm: bool
    use_masked_initial_norm: bool
    expected_trace: ExpectedExecutionTrace

    @property
    def config_id(self) -> str:
        return self.config.config_id

    @property
    def use_cuda_graph(self) -> bool:
        return self.runtime_backend is RuntimeBackend.CUDA_GRAPH

    @property
    def use_batch_tiled_cuda_graph(self) -> bool:
        return self.runtime_backend is RuntimeBackend.BATCH_TILED_CUDA_GRAPH

    @property
    def use_compiled_forward(self) -> bool:
        return self.runtime_backend is RuntimeBackend.COMPILED_FORWARD

    @property
    def use_streamed_execution(self) -> bool:
        return self.runtime_backend is RuntimeBackend.STREAMED

    @property
    def use_triton_initial_fp16_norm(self) -> bool:
        return self.initial_norm_backend is InitialNormBackend.TRITON_FP16

    @property
    def use_fused_initial_norm_qkv(self) -> bool:
        return self.initial_norm_backend is InitialNormBackend.TRITON_FUSED_QKV

    @property
    def use_linear_boundary_fusion(self) -> bool:
        return self.residual_norm_backend is ResidualNormBackend.TRITON_LINEAR_MIXED

    def describe(self) -> dict[str, Any]:
        """Serialize the immutable plan and its automatically derived evidence."""

        if not self.outer_context.causal:
            causal_mask = "none"
        elif self.attention_backend in {
            AttentionBackend.TRITON_SHAPE13,
            AttentionBackend.TRITON_DH8,
            AttentionBackend.TRITON_STREAMING_DH64,
        }:
            causal_mask = "online_causal"
        elif self.attention_backend in {
            AttentionBackend.CAUSAL_SDPA,
            AttentionBackend.FP16_CUDNN_SDPA,
            AttentionBackend.FP16_EFFICIENT_SDPA,
        }:
            causal_mask = "implicit_sdpa"
        else:
            causal_mask = "query_block"

        return {
            "config": self.config.to_dict(),
            "qkv_projection": "packed",
            "qkv_projection_backend": self.qkv_projection_backend.value,
            "qkv_projection_compute_dtype": (self.qkv_projection_compute_dtype),
            "qkv_materialization": self.qkv_materialization.value,
            "attention_backend": self.attention_backend.value,
            "attention_compute_dtype": self.attention_compute_dtype,
            "attention_output_bridge": self.attention_output_bridge.value,
            "attention_output_layout": self.attention_output_layout.value,
            "attention_output_projection_backend": (
                self.attention_output_projection_backend.value
            ),
            "attention_output_projection_compute_dtype": (
                self.attention_output_projection_compute_dtype
            ),
            "ffn_backend": self.ffn_backend.value,
            "ffn_input_projection_backend": (self.ffn_input_projection_backend.value),
            "ffn_input_projection_compute_dtype": (
                self.ffn_input_projection_compute_dtype
            ),
            "ffn_activation_output_dtype": self.ffn_activation_output_dtype,
            "ffn_output_projection_backend": (self.ffn_output_projection_backend.value),
            "ffn_output_projection_compute_dtype": (
                self.ffn_output_projection_compute_dtype
            ),
            "precision_plan": self.precision_plan.value,
            "residual_norm_backend": self.residual_norm_backend.value,
            "initial_norm_backend": self.initial_norm_backend.value,
            "runtime_backend": self.runtime_backend.value,
            "compile_mode": self.compile_mode,
            "batch_tile_size": self.batch_tile_size,
            "microbatch_size": self.microbatch_size,
            "reuse_unchanged_input": self.reuse_unchanged_input,
            "use_d32_residual_norm": self.use_d32_residual_norm,
            "use_masked_residual_norm": self.use_masked_residual_norm,
            "use_masked_initial_norm": self.use_masked_initial_norm,
            "qkv_launch": (
                None if self.qkv_launch is None else self.qkv_launch.to_dict()
            ),
            "attention_output_projection_launch": (
                None
                if self.attention_output_projection_launch is None
                else self.attention_output_projection_launch.to_dict()
            ),
            "ffn_launch": (
                None if self.ffn_launch is None else self.ffn_launch.to_dict()
            ),
            "ffn_input_launch": (
                None
                if self.ffn_input_launch is None
                else self.ffn_input_launch.to_dict()
            ),
            "outer_batch_size": self.outer_context.batch_size,
            "inner_batch_size": self.inner_context.batch_size,
            "causal_mask": causal_mask,
            "valid_token_mask": (
                "direct_key_mask" if self.outer_context.has_valid_token_mask else "none"
            ),
        }


__all__ = [
    "ExecutionContext",
    "ExecutionPlan",
    "ExpectedExecutionTrace",
]
