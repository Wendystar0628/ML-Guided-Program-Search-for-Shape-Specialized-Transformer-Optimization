"""Immutable execution plans compiled from typed Transformer configurations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from .config import (
    AttentionBackend,
    AttentionOutputLayout,
    ConfigSpec,
    InitialNormBackend,
    LinearBackend,
    ResidualNormBackend,
    RuntimeBackend,
    TritonAttentionParams,
    TritonNormParams,
)


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Input and model facts that affect static execution eligibility."""

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
    linear_backend: LinearBackend
    residual_norm_backend: ResidualNormBackend
    initial_norm_backend: InitialNormBackend
    attention_compute_dtype: str
    linear_compute_dtype: str
    attention_output_layout: AttentionOutputLayout
    attention_calls: int
    linear_calls: int
    residual_norm_calls: int
    initial_norm_calls: int
    runtime_calls: int
    attention_launch: TritonAttentionParams | None = None
    residual_norm_launch: TritonNormParams | None = None
    initial_norm_launch: TritonNormParams | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_backend": self.runtime_backend.value,
            "attention_backend": self.attention_backend.value,
            "linear_backend": self.linear_backend.value,
            "residual_norm_backend": self.residual_norm_backend.value,
            "initial_norm_backend": self.initial_norm_backend.value,
            "attention_compute_dtype": self.attention_compute_dtype,
            "linear_compute_dtype": self.linear_compute_dtype,
            "attention_calls": self.attention_calls,
            "linear_calls": self.linear_calls,
            "residual_norm_calls": self.residual_norm_calls,
            "initial_norm_calls": self.initial_norm_calls,
            "runtime_calls": self.runtime_calls,
            "complete": True,
        }


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Strictly compiled plan; forward must execute it without hidden fallback."""

    config: ConfigSpec
    outer_context: ExecutionContext
    inner_context: ExecutionContext
    attention_backend: AttentionBackend
    attention_compute_dtype: str
    attention_output_layout: AttentionOutputLayout
    linear_backend: LinearBackend
    linear_compute_dtype: str
    residual_norm_backend: ResidualNormBackend
    initial_norm_backend: InitialNormBackend
    runtime_backend: RuntimeBackend
    compile_mode: str | None
    batch_tile_size: int | None
    microbatch_size: int | None
    reuse_unchanged_input: bool
    attention_launch: TritonAttentionParams | None
    residual_norm_launch: TritonNormParams | None
    initial_norm_launch: TritonNormParams | None
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

    def describe(self) -> dict[str, Any]:
        """Serialize the immutable plan and its automatically derived evidence."""

        if not self.outer_context.causal:
            causal_mask = "none"
        elif self.attention_backend in {
            AttentionBackend.TRITON_SHAPE13,
            AttentionBackend.TRITON_DH8,
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
            "attention_backend": self.attention_backend.value,
            "attention_compute_dtype": self.attention_compute_dtype,
            "attention_output_layout": self.attention_output_layout.value,
            "linear_backend": self.linear_backend.value,
            "linear_compute_dtype": self.linear_compute_dtype,
            "residual_norm_backend": self.residual_norm_backend.value,
            "initial_norm_backend": self.initial_norm_backend.value,
            "runtime_backend": self.runtime_backend.value,
            "compile_mode": self.compile_mode,
            "batch_tile_size": self.batch_tile_size,
            "microbatch_size": self.microbatch_size,
            "reuse_unchanged_input": self.reuse_unchanged_input,
            "outer_batch_size": self.outer_context.batch_size,
            "inner_batch_size": self.inner_context.batch_size,
            "causal_mask": causal_mask,
            "valid_token_mask": (
                "direct_key_mask"
                if self.outer_context.has_valid_token_mask
                else "none"
            ),
        }


__all__ = [
    "ExecutionContext",
    "ExecutionPlan",
    "ExpectedExecutionTrace",
]
