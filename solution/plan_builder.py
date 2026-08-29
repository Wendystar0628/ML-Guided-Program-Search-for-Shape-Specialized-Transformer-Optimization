"""Build execution plans from typed search configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .config import (
    AttentionBackend,
    ConfigSpec,
    InitialNormBackend,
    LinearBackend,
    ResidualNormBackend,
    RuntimeBackend,
)
from .operators import (
    triton_dh8_causal_attention_available,
    triton_initial_fp16_layer_norm_available,
    triton_mixed_residual_layer_norm_available,
    triton_residual_layer_norm_available,
    triton_shape13_causal_attention_available,
)
from .plan import (
    ExecutionContext,
    ExecutionPlan,
    ExpectedExecutionTrace,
)


@dataclass(frozen=True, slots=True)
class ConstraintViolation:
    """One statically known reason a sampled configuration cannot execute."""

    code: str
    field: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "field": self.field,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class CompileRejection:
    """Structured rejection retained as an infeasible search observation."""

    config_id: str
    violations: tuple[ConstraintViolation, ...]

    def __post_init__(self) -> None:
        if not self.violations:
            raise ValueError("CompileRejection requires at least one violation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "violations": [violation.to_dict() for violation in self.violations],
        }


class ConfigRejectedError(ValueError):
    """Raised when strict runtime compilation receives an invalid config."""

    def __init__(self, rejection: CompileRejection) -> None:
        self.rejection = rejection
        summary = "; ".join(
            f"{item.field}:{item.code}" for item in rejection.violations
        )
        super().__init__(f"configuration {rejection.config_id} rejected: {summary}")


@dataclass(frozen=True, slots=True)
class CompilationResult:
    """Non-throwing compiler result used by the search engine."""

    config_id: str
    plan: ExecutionPlan | None
    rejection: CompileRejection | None

    def __post_init__(self) -> None:
        if (self.plan is None) == (self.rejection is None):
            raise ValueError("exactly one of plan and rejection must be present")

    @property
    def accepted(self) -> bool:
        return self.plan is not None

    @property
    def violations(self) -> tuple[ConstraintViolation, ...]:
        return () if self.rejection is None else self.rejection.violations

    def require_plan(self) -> ExecutionPlan:
        if self.plan is not None:
            return self.plan
        assert self.rejection is not None
        raise ConfigRejectedError(self.rejection)


@dataclass(frozen=True, slots=True)
class HardwareCapabilities:
    """Small injectable hardware/backend snapshot used by static compilation."""

    device_type: str
    compute_capability: tuple[int, int] | None
    shared_memory_per_block: int | None
    mem_efficient_sdp: bool
    cudnn_sdp: bool
    cudnn_available: bool
    torch_compile: bool
    triton_shape13_attention: bool
    triton_dh8_attention: bool
    triton_residual_norm: bool
    triton_mixed_residual_norm: bool
    triton_initial_norm: bool

    @classmethod
    def detect(cls, device: torch.device) -> HardwareCapabilities:
        device_type = device.type
        compute_capability: tuple[int, int] | None = None
        shared_memory: int | None = None
        if device_type == "cuda" and torch.cuda.is_available():
            index = device.index
            if index is None:
                index = torch.cuda.current_device()
            compute_capability = torch.cuda.get_device_capability(index)
            properties = torch.cuda.get_device_properties(index)
            shared_memory = int(
                getattr(
                    properties,
                    "shared_memory_per_block_optin",
                    properties.shared_memory_per_block,
                )
            )
        return cls(
            device_type=device_type,
            compute_capability=compute_capability,
            shared_memory_per_block=shared_memory,
            mem_efficient_sdp=bool(
                device_type == "cuda"
                and torch.backends.cuda.mem_efficient_sdp_enabled()
            ),
            cudnn_sdp=bool(
                device_type == "cuda" and torch.backends.cuda.cudnn_sdp_enabled()
            ),
            cudnn_available=bool(
                device_type == "cuda" and torch.backends.cudnn.is_available()
            ),
            torch_compile=callable(getattr(torch, "compile", None)),
            triton_shape13_attention=triton_shape13_causal_attention_available(),
            triton_dh8_attention=triton_dh8_causal_attention_available(),
            triton_residual_norm=triton_residual_layer_norm_available(),
            triton_mixed_residual_norm=(triton_mixed_residual_layer_norm_available()),
            triton_initial_norm=triton_initial_fp16_layer_norm_available(),
        )


class PlanBuilder:
    """Build exactly one execution plan without policy lookup or fallback."""

    def evaluate(
        self,
        config: ConfigSpec,
        context: ExecutionContext,
        hardware: HardwareCapabilities | None = None,
    ) -> CompilationResult:
        if not isinstance(config, ConfigSpec):
            raise TypeError("config must be ConfigSpec")
        if not isinstance(context, ExecutionContext):
            raise TypeError("context must be ExecutionContext")
        capabilities = hardware or HardwareCapabilities.detect(context.device)
        violations: list[ConstraintViolation] = []

        def reject(code: str, field: str, message: str) -> None:
            violations.append(ConstraintViolation(code, field, message))

        self._validate_context(context, capabilities, reject)
        inner_context = self._validate_runtime(config, context, capabilities, reject)
        self._validate_linear(config, inner_context, reject)
        self._validate_attention(config, inner_context, capabilities, reject)
        self._validate_residual_norm(config, inner_context, capabilities, reject)
        self._validate_initial_norm(config, inner_context, capabilities, reject)

        if violations:
            rejection = CompileRejection(config.config_id, tuple(violations))
            return CompilationResult(config.config_id, None, rejection)

        plan = self._build_plan(config, context, inner_context)
        return CompilationResult(config.config_id, plan, None)

    def build(
        self,
        config: ConfigSpec,
        context: ExecutionContext,
        hardware: HardwareCapabilities | None = None,
    ) -> ExecutionPlan:
        """Build a plan or raise; never substitute another configuration."""

        return self.evaluate(config, context, hardware).require_plan()

    @staticmethod
    def _validate_context(
        context: ExecutionContext,
        hardware: HardwareCapabilities,
        reject: Any,
    ) -> None:
        for field_name in ("batch_size", "seq_len", "d_model", "num_heads"):
            value = getattr(context, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                reject("non_positive", field_name, f"{field_name} must be positive")
        if context.num_heads > 0 and context.d_model % context.num_heads:
            reject(
                "not_divisible",
                "num_heads",
                "d_model must be divisible by num_heads",
            )
        if context.has_valid_token_mask and not context.mask_compatible:
            reject(
                "incompatible_mask",
                "valid_token_mask",
                "token mask does not match the input shape, device, or dtype",
            )
        if hardware.device_type != context.device.type:
            reject(
                "device_mismatch",
                "hardware.device_type",
                "hardware capabilities do not describe the input device",
            )

    @staticmethod
    def _validate_runtime(
        config: ConfigSpec,
        context: ExecutionContext,
        hardware: HardwareCapabilities,
        reject: Any,
    ) -> ExecutionContext:
        runtime = config.schedule.runtime
        if runtime is RuntimeBackend.EAGER:
            return context

        if context.device.type != "cuda":
            reject("requires_cuda", "schedule.runtime", "runtime requires CUDA")
        if not context.inference:
            reject(
                "requires_inference",
                "schedule.runtime",
                "runtime is available only for inference",
            )
        if not context.input_contiguous:
            reject(
                "requires_contiguous_input",
                "schedule.runtime",
                "runtime requires a contiguous input",
            )

        if runtime is RuntimeBackend.CUDA_GRAPH:
            return context
        if runtime is RuntimeBackend.BATCH_TILED_CUDA_GRAPH:
            tile_size = config.schedule.batch_tile_size
            assert tile_size is not None
            if context.has_valid_token_mask:
                reject(
                    "mask_not_supported",
                    "schedule.runtime",
                    "batch-tiled graph does not accept a token mask",
                )
            if tile_size >= context.batch_size:
                reject(
                    "tile_not_smaller_than_batch",
                    "schedule.batch_tile_size",
                    "batch tile must be smaller than the logical batch",
                )
            return context.with_batch_size(tile_size)
        if runtime is RuntimeBackend.STREAMED:
            microbatch_size = config.schedule.microbatch_size
            assert microbatch_size is not None
            if context.has_valid_token_mask:
                reject(
                    "mask_not_supported",
                    "schedule.runtime",
                    "streamed execution currently requires an all-valid workload",
                )
            if microbatch_size > context.batch_size:
                reject(
                    "microbatch_exceeds_batch",
                    "schedule.microbatch_size",
                    "microbatch cannot exceed the logical batch",
                )
            return context.with_batch_size(microbatch_size)
        if runtime is RuntimeBackend.COMPILED_FORWARD:
            if not hardware.torch_compile:
                reject(
                    "backend_unavailable",
                    "schedule.runtime",
                    "torch.compile is unavailable",
                )
            return context
        raise AssertionError(f"unhandled runtime backend: {runtime}")

    @staticmethod
    def _validate_linear(
        config: ConfigSpec,
        context: ExecutionContext,
        reject: Any,
    ) -> None:
        if config.program.linear is LinearBackend.INPUT_DTYPE:
            return
        if context.device.type != "cuda":
            reject("requires_cuda", "program.linear", "FP16 linear requires CUDA")
        if not context.inference:
            reject(
                "requires_inference",
                "program.linear",
                "FP16 linear is available only for inference",
            )

    @staticmethod
    def _validate_attention(
        config: ConfigSpec,
        context: ExecutionContext,
        hardware: HardwareCapabilities,
        reject: Any,
    ) -> None:
        backend = config.program.attention
        head_dim = context.head_dim
        qkv_dtype = (
            "float16"
            if config.program.linear
            in {LinearBackend.AUTOCAST_FP16, LinearBackend.FP16_SHADOW}
            else context.dtype_name
        )
        if backend is AttentionBackend.REFERENCE_STREAMING:
            if context.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
                reject(
                    "unsupported_dtype",
                    "program.attention",
                    "reference attention supports FP16, BF16, and FP32",
                )
            return
        if backend is AttentionBackend.CAUSAL_SDPA:
            if qkv_dtype not in {"float16", "float32"}:
                reject(
                    "comparator_unsafe_dtype",
                    "program.attention",
                    "native SDPA is excluded for BF16 comparator safety",
                )
            return

        if not context.inference:
            reject(
                "requires_inference",
                "program.attention",
                "selected attention backend is available only for inference",
            )
        if not context.causal:
            reject(
                "requires_causal",
                "program.attention",
                "selected attention backend implements causal attention",
            )
        if context.device.type != "cuda":
            reject("requires_cuda", "program.attention", "backend requires CUDA")
        if context.has_valid_token_mask:
            reject(
                "mask_not_supported",
                "program.attention",
                "backend does not accept a token mask",
            )
        if hardware.compute_capability is None or hardware.compute_capability < (8, 0):
            reject(
                "compute_capability",
                "hardware.compute_capability",
                "backend requires CUDA compute capability 8.0 or newer",
            )

        if backend is AttentionBackend.FP16_EFFICIENT_SDPA:
            if qkv_dtype not in {"float16", "float32"}:
                reject(
                    "unsupported_dtype",
                    "program.attention",
                    "Efficient SDPA accepts FP16 or FP32 QKV",
                )
            if not hardware.mem_efficient_sdp:
                reject(
                    "backend_unavailable",
                    "program.attention",
                    "memory-efficient SDPA is disabled",
                )
            return
        if backend is AttentionBackend.FP16_CUDNN_SDPA:
            if qkv_dtype not in {"float16", "float32"}:
                reject(
                    "unsupported_dtype",
                    "program.attention",
                    "cuDNN SDPA accepts FP16 or FP32 QKV",
                )
            if not hardware.cudnn_sdp or not hardware.cudnn_available:
                reject(
                    "backend_unavailable",
                    "program.attention",
                    "cuDNN SDPA is unavailable",
                )
            return

        launch = config.schedule.attention_launch
        assert launch is not None
        if launch.block_m not in {16, 32, 64, 128}:
            reject(
                "unsupported_launch_value",
                "schedule.attention_launch.block_m",
                "current Triton attention supports block_m in {16, 32, 64, 128}",
            )
        if launch.block_n not in {16, 32, 64, 128}:
            reject(
                "unsupported_launch_value",
                "schedule.attention_launch.block_n",
                "current Triton attention supports block_n in {16, 32, 64, 128}",
            )
        if launch.num_warps not in {2, 4, 8}:
            reject(
                "unsupported_launch_value",
                "schedule.attention_launch.num_warps",
                "current Triton attention supports 2, 4, or 8 warps",
            )
        if launch.num_stages not in {1, 2, 3, 4}:
            reject(
                "unsupported_launch_value",
                "schedule.attention_launch.num_stages",
                "current Triton attention supports 1 to 4 stages",
            )
        if qkv_dtype != "float16":
            reject(
                "requires_fp16_qkv",
                "program.linear",
                "Triton attention requires a linear branch that produces FP16 QKV",
            )
        if context.seq_len % launch.block_m:
            reject(
                "block_m_not_divisor",
                "schedule.attention_launch.block_m",
                "current Triton attention requires sequence length divisible by block_m",
            )
        if context.seq_len % launch.block_n:
            reject(
                "block_n_not_divisor",
                "schedule.attention_launch.block_n",
                "current Triton attention requires sequence length divisible by block_n",
            )
        if launch.block_m % launch.block_n:
            reject(
                "block_n_not_divisor",
                "schedule.attention_launch.block_n",
                "current causal loop requires block_n to divide block_m",
            )
        if backend is AttentionBackend.TRITON_SHAPE13:
            if not hardware.triton_shape13_attention:
                reject(
                    "backend_unavailable",
                    "program.attention",
                    "Shape13 Triton attention is unavailable",
                )
            if (
                context.batch_size,
                context.num_heads,
                context.seq_len,
                head_dim,
            ) != (64, 4, 1024, 32):
                reject(
                    "unsupported_shape",
                    "program.attention",
                    "Shape13 Triton template requires B64/H4/S1024/Dh32",
                )
            return
        if backend is AttentionBackend.TRITON_DH8:
            if not hardware.triton_dh8_attention:
                reject(
                    "backend_unavailable",
                    "program.attention",
                    "Dh8 Triton attention is unavailable",
                )
            if (
                context.batch_size,
                context.num_heads,
                context.seq_len,
                head_dim,
            ) != (64, 16, 128, 8):
                reject(
                    "unsupported_shape",
                    "program.attention",
                    "Dh8 Triton template requires B64/H16/S128/Dh8",
                )
            return
        raise AssertionError(f"unhandled attention backend: {backend}")

    @staticmethod
    def _validate_residual_norm(
        config: ConfigSpec,
        context: ExecutionContext,
        hardware: HardwareCapabilities,
        reject: Any,
    ) -> None:
        backend = config.program.residual_norm
        if backend is ResidualNormBackend.TORCH:
            return
        launch = config.schedule.residual_norm_launch
        if launch is not None and (
            launch.block_rows not in {1, 2, 4, 8}
            or launch.num_warps not in {1, 2, 4, 8}
        ):
            reject(
                "unsupported_launch_value",
                "schedule.residual_norm_launch",
                "current Triton norm templates support rows/warps in {1, 2, 4, 8}",
            )
        if context.device.type != "cuda":
            reject("requires_cuda", "program.residual_norm", "backend requires CUDA")
        if context.dtype != torch.float32:
            reject(
                "requires_float32_input",
                "program.residual_norm",
                "backend currently requires FP32 residual state",
            )
        if not context.inference:
            reject(
                "requires_inference",
                "program.residual_norm",
                "backend is available only for inference",
            )
        if context.has_valid_token_mask:
            reject(
                "mask_not_supported",
                "program.residual_norm",
                "fused residual norm does not apply token masking",
            )
        if backend is ResidualNormBackend.COMPILED:
            if not hardware.torch_compile:
                reject(
                    "backend_unavailable",
                    "program.residual_norm",
                    "torch.compile is unavailable",
                )
            if config.schedule.runtime is RuntimeBackend.COMPILED_FORWARD:
                reject(
                    "nested_compilation",
                    "program.residual_norm",
                    "compiled residual norm cannot be nested in compiled forward",
                )
            return
        if backend is ResidualNormBackend.TRITON:
            if not hardware.triton_residual_norm:
                reject(
                    "backend_unavailable",
                    "program.residual_norm",
                    "Triton residual norm is unavailable",
                )
            if context.d_model != 128:
                reject(
                    "unsupported_shape",
                    "program.residual_norm",
                    "current Triton residual norm template requires D128",
                )
            if config.schedule.runtime is RuntimeBackend.COMPILED_FORWARD:
                reject(
                    "compile_incompatible",
                    "program.residual_norm",
                    "Triton residual norm branch is not nested in compiled forward",
                )
            return
        if backend is ResidualNormBackend.TRITON_MIXED:
            if not hardware.triton_mixed_residual_norm:
                reject(
                    "backend_unavailable",
                    "program.residual_norm",
                    "Triton mixed residual norm is unavailable",
                )
            if context.d_model != 128:
                reject(
                    "unsupported_shape",
                    "program.residual_norm",
                    "mixed Triton norm template requires D128",
                )
            if config.program.linear is LinearBackend.INPUT_DTYPE:
                reject(
                    "requires_fp16_update",
                    "program.linear",
                    "mixed Triton norm requires FP16 branch updates",
                )
            if config.schedule.runtime is RuntimeBackend.COMPILED_FORWARD:
                reject(
                    "runtime_incompatible",
                    "schedule.runtime",
                    "mixed Triton norm is not nested in compiled forward",
                )
            return
        raise AssertionError(f"unhandled residual norm backend: {backend}")

    @staticmethod
    def _validate_initial_norm(
        config: ConfigSpec,
        context: ExecutionContext,
        hardware: HardwareCapabilities,
        reject: Any,
    ) -> None:
        if config.program.initial_norm is InitialNormBackend.TORCH:
            return
        launch = config.schedule.initial_norm_launch
        assert launch is not None
        if launch.block_rows not in {1, 2, 4, 8} or launch.num_warps not in {
            1,
            2,
            4,
            8,
        }:
            reject(
                "unsupported_launch_value",
                "schedule.initial_norm_launch",
                "current Triton norm templates support rows/warps in {1, 2, 4, 8}",
            )
        if not hardware.triton_initial_norm:
            reject(
                "backend_unavailable",
                "program.initial_norm",
                "Triton initial norm is unavailable",
            )
        if context.device.type != "cuda" or context.dtype != torch.float32:
            reject(
                "requires_cuda_float32",
                "program.initial_norm",
                "Triton initial norm requires CUDA FP32 input",
            )
        if context.d_model != 128:
            reject(
                "unsupported_shape",
                "program.initial_norm",
                "Triton initial norm template requires D128",
            )
        if config.schedule.runtime is RuntimeBackend.COMPILED_FORWARD:
            reject(
                "runtime_incompatible",
                "schedule.runtime",
                "Triton initial norm is not nested in compiled forward",
            )
        if config.program.linear is LinearBackend.INPUT_DTYPE:
            reject(
                "linear_incompatible",
                "program.linear",
                "Triton initial norm requires FP16 linears",
            )
        if config.program.residual_norm is not ResidualNormBackend.TRITON_MIXED:
            reject(
                "residual_norm_incompatible",
                "program.residual_norm",
                "Triton initial norm requires the mixed residual stream",
            )

    @staticmethod
    def _build_plan(
        config: ConfigSpec,
        outer_context: ExecutionContext,
        inner_context: ExecutionContext,
    ) -> ExecutionPlan:
        attention_backend = config.program.attention
        linear_backend = config.program.linear
        attention_compute_dtype = (
            "float16"
            if attention_backend
            in {
                AttentionBackend.FP16_EFFICIENT_SDPA,
                AttentionBackend.FP16_CUDNN_SDPA,
                AttentionBackend.TRITON_SHAPE13,
                AttentionBackend.TRITON_DH8,
            }
            else (
                "float16"
                if linear_backend
                in {LinearBackend.AUTOCAST_FP16, LinearBackend.FP16_SHADOW}
                else inner_context.dtype_name
            )
        )
        linear_compute_dtype = (
            "float16"
            if linear_backend
            in {LinearBackend.AUTOCAST_FP16, LinearBackend.FP16_SHADOW}
            else inner_context.dtype_name
        )
        layers = max(0, int(inner_context.num_layers or 0))
        expected = ExpectedExecutionTrace(
            runtime_backend=config.schedule.runtime,
            attention_backend=attention_backend,
            linear_backend=linear_backend,
            residual_norm_backend=config.program.residual_norm,
            initial_norm_backend=config.program.initial_norm,
            attention_compute_dtype=attention_compute_dtype,
            linear_compute_dtype=linear_compute_dtype,
            attention_output_layout=config.attention_output_layout,
            attention_calls=layers,
            linear_calls=4 * layers,
            residual_norm_calls=2 * layers,
            initial_norm_calls=1 if layers else 0,
            runtime_calls=1,
            attention_launch=config.schedule.attention_launch,
            residual_norm_launch=config.schedule.residual_norm_launch,
            initial_norm_launch=config.schedule.initial_norm_launch,
        )
        return ExecutionPlan(
            config=config,
            outer_context=outer_context,
            inner_context=inner_context,
            attention_backend=attention_backend,
            attention_compute_dtype=attention_compute_dtype,
            attention_output_layout=config.attention_output_layout,
            linear_backend=linear_backend,
            linear_compute_dtype=linear_compute_dtype,
            residual_norm_backend=config.program.residual_norm,
            initial_norm_backend=config.program.initial_norm,
            runtime_backend=config.schedule.runtime,
            compile_mode=config.schedule.compile_mode,
            batch_tile_size=config.schedule.batch_tile_size,
            microbatch_size=config.schedule.microbatch_size,
            reuse_unchanged_input=config.schedule.reuse_unchanged_input,
            attention_launch=config.schedule.attention_launch,
            residual_norm_launch=config.schedule.residual_norm_launch,
            initial_norm_launch=config.schedule.initial_norm_launch,
            expected_trace=expected,
        )


__all__ = [
    "CompilationResult",
    "CompileRejection",
    "ConfigRejectedError",
    "ConstraintViolation",
    "HardwareCapabilities",
    "PlanBuilder",
]
