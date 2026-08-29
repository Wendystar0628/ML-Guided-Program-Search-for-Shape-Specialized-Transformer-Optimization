"""Build execution plans from typed search configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .config import (
    AttentionBackend,
    AttentionOutputBridge,
    ConfigSpec,
    FFNBackend,
    InitialNormBackend,
    PrecisionPlan,
    ProjectionBackend,
    QKVMaterialization,
    ResidualNormBackend,
    RuntimeBackend,
)
from .operators import (
    triton_dh8_causal_attention_available,
    triton_exact_gelu_available,
    triton_initial_fp16_layer_norm_available,
    triton_mixed_residual_layer_norm_available,
    triton_residual_layer_norm_available,
    triton_shape13_causal_attention_available,
)
from .operators.attention.triton_streaming_dh64 import (
    triton_streaming_dh64_causal_attention_available,
)
from .operators.ffn.triton_linear_exact_gelu import (
    triton_linear_exact_gelu_available,
)
from .operators.layer.triton_d32_fusion import (
    triton_d32_residual_layer_norm_available,
)
from .operators.norm.triton_masked import triton_masked_layer_norm_available
from .operators.projection.triton_attention_output import (
    triton_attention_output_projection_available,
)
from .operators.projection.triton_qkv_layout import (
    triton_qkv_native_bhsd_available,
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
    triton_exact_gelu: bool
    triton_streaming_dh64_attention: bool = False
    triton_qkv_native_bhsd: bool = False
    triton_attention_output_projection: bool = False
    triton_linear_exact_gelu: bool = False
    triton_d32_residual_norm: bool = False
    triton_masked_norm: bool = False

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
            triton_exact_gelu=triton_exact_gelu_available(),
            triton_streaming_dh64_attention=(
                triton_streaming_dh64_causal_attention_available()
            ),
            triton_qkv_native_bhsd=triton_qkv_native_bhsd_available(),
            triton_attention_output_projection=(
                triton_attention_output_projection_available()
            ),
            triton_linear_exact_gelu=triton_linear_exact_gelu_available(),
            triton_d32_residual_norm=(triton_d32_residual_layer_norm_available()),
            triton_masked_norm=triton_masked_layer_norm_available(),
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
        self._validate_projections(config, inner_context, reject)
        self._validate_qkv_materialization(
            config,
            inner_context,
            capabilities,
            reject,
        )
        self._validate_attention(
            config,
            inner_context,
            capabilities,
            reject,
            outer_context=context,
        )
        self._validate_attention_bridge(
            config,
            inner_context,
            capabilities,
            reject,
        )
        self._validate_ffn(config, inner_context, capabilities, reject)
        self._validate_residual_norm(config, inner_context, capabilities, reject)
        self._validate_initial_norm(config, inner_context, capabilities, reject)

        if violations:
            rejection = CompileRejection(config.config_id, tuple(violations))
            return CompilationResult(config.config_id, None, rejection)

        plan = self._build_plan(config, context, inner_context, capabilities)
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
            elif context.batch_size % microbatch_size:
                reject(
                    "microbatch_not_divisor",
                    "schedule.microbatch_size",
                    "microbatch must divide the logical batch",
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
    def _validate_projections(
        config: ConfigSpec,
        context: ExecutionContext,
        reject: Any,
    ) -> None:
        for field_name in (
            "qkv_projection",
            "attention_output_projection",
            "ffn_input_projection",
            "ffn_output_projection",
        ):
            backend = getattr(config.program, field_name)
            if backend is ProjectionBackend.INPUT_DTYPE:
                continue
            if context.device.type != "cuda":
                reject(
                    "requires_cuda",
                    f"program.{field_name}",
                    "FP16 projection requires CUDA",
                )
            if not context.inference:
                reject(
                    "requires_inference",
                    f"program.{field_name}",
                    "FP16 projection is available only for inference",
                )

    @staticmethod
    def _validate_qkv_materialization(
        config: ConfigSpec,
        context: ExecutionContext,
        hardware: HardwareCapabilities,
        reject: Any,
    ) -> None:
        if (
            config.program.qkv_materialization
            is not QKVMaterialization.TRITON_NATIVE_BHSD
        ):
            return
        if config.program.qkv_projection is not ProjectionBackend.FP16_SHADOW:
            reject(
                "backend_incompatible",
                "program.qkv_projection",
                "native BHSD QKV requires FP16 shadow weights",
            )
        if not hardware.triton_qkv_native_bhsd:
            reject(
                "backend_unavailable",
                "program.qkv_materialization",
                "Triton native-layout QKV projection is unavailable",
            )
        if context.device.type != "cuda" or not context.inference:
            reject(
                "requires_cuda_inference",
                "program.qkv_materialization",
                "native BHSD QKV is available only for CUDA inference",
            )
        if context.d_model not in {32, 128, 1024}:
            reject(
                "unsupported_shape",
                "program.qkv_materialization",
                "native BHSD QKV supports D32, D128, and D1024",
            )
        if context.seq_len not in {32, 128, 1024}:
            reject(
                "unsupported_shape",
                "program.qkv_materialization",
                "native BHSD QKV supports S32, S128, and S1024",
            )
        if context.num_heads not in {1, 2, 4, 16}:
            reject(
                "unsupported_shape",
                "program.qkv_materialization",
                "native BHSD QKV supports 1, 2, 4, or 16 heads",
            )
        if config.schedule.runtime is RuntimeBackend.COMPILED_FORWARD:
            reject(
                "runtime_incompatible",
                "schedule.runtime",
                "native BHSD QKV is not nested in compiled forward",
            )
        launch = config.schedule.qkv_launch
        assert launch is not None
        if (
            launch.block_m not in {16, 32, 64, 128}
            or launch.block_n not in {16, 32, 64, 128}
            or launch.block_k not in {16, 32, 64}
            or launch.num_warps not in {2, 4, 8}
        ):
            reject(
                "unsupported_launch_value",
                "schedule.qkv_launch",
                "native BHSD QKV launch is outside the implemented template",
            )

    @staticmethod
    def _validate_attention(
        config: ConfigSpec,
        context: ExecutionContext,
        hardware: HardwareCapabilities,
        reject: Any,
        *,
        outer_context: ExecutionContext,
    ) -> None:
        backend = config.program.attention
        head_dim = context.head_dim
        qkv_dtype = PlanBuilder._projection_dtype(
            config.program.qkv_projection,
            context,
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
        if backend is AttentionBackend.TRITON_STREAMING_DH64:
            if not hardware.triton_streaming_dh64_attention:
                reject(
                    "backend_unavailable",
                    "program.attention",
                    "Shape 14 Triton streaming attention is unavailable",
                )
            if (
                context.batch_size not in {1, 2, 4}
                or context.num_layers != 2
                or context.ffn_dim != 1024
                or context.num_heads != 16
                or context.seq_len != 100000
                or context.d_model != 1024
                or context.causal is not True
                or context.has_valid_token_mask
                or head_dim != 64
            ):
                reject(
                    "unsupported_shape",
                    "program.attention",
                    "Shape 14 Triton attention requires a B1/2/4 microbatch "
                    "with L2/H16/S100000/Dh64",
                )
            if qkv_dtype != "float16":
                reject(
                    "requires_fp16_qkv",
                    "program.qkv_projection",
                    "Shape 14 Triton attention requires FP16 QKV",
                )
            if (
                launch.block_m not in {16, 32, 64}
                or launch.block_n not in {16, 32, 64, 128}
                or launch.num_warps not in {2, 4, 8}
                or launch.num_stages not in {1, 2, 3, 4}
                or (launch.block_n == 128 and launch.num_stages == 4)
            ):
                reject(
                    "unsupported_launch_value",
                    "schedule.attention_launch",
                    "Shape 14 streaming launch is outside the implemented template",
                )
            return
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
                "program.qkv_projection",
                "Triton attention requires an FP16 QKV projection",
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
                context.batch_size != 64
                or context.num_heads not in {4, 16}
                or context.seq_len != 128
                or head_dim != 8
            ):
                reject(
                    "unsupported_shape",
                    "program.attention",
                    "Dh8 Triton template requires B64/H{4,16}/S128/Dh8",
                )
            return
        raise AssertionError(f"unhandled attention backend: {backend}")

    @staticmethod
    def _validate_attention_bridge(
        config: ConfigSpec,
        context: ExecutionContext,
        hardware: HardwareCapabilities,
        reject: Any,
    ) -> None:
        backend = config.program.attention
        bridge = config.program.attention_output_bridge
        direct_backends = {
            AttentionBackend.TRITON_SHAPE13,
            AttentionBackend.TRITON_DH8,
            AttentionBackend.TRITON_STREAMING_DH64,
        }
        if bridge is AttentionOutputBridge.ATTENTION_DIRECT_BSD:
            if backend not in direct_backends:
                reject(
                    "backend_incompatible",
                    "program.attention_output_bridge",
                    "direct BSD output is implemented only by specialized Triton attention",
                )
            return
        if backend in {
            AttentionBackend.TRITON_DH8,
            AttentionBackend.TRITON_STREAMING_DH64,
        }:
            reject(
                "backend_incompatible",
                "program.attention_output_bridge",
                "selected attention produces BSD directly",
            )
        if bridge is not AttentionOutputBridge.TRITON_BHSD_PROJECTION:
            return
        if (
            config.program.attention_output_projection
            is not ProjectionBackend.FP16_SHADOW
        ):
            reject(
                "backend_incompatible",
                "program.attention_output_projection",
                "Triton BHSD projection requires FP16 shadow weights",
            )
        if not hardware.triton_attention_output_projection:
            reject(
                "backend_unavailable",
                "program.attention_output_bridge",
                "Triton BHSD output projection is unavailable",
            )
        if context.device.type != "cuda" or not context.inference:
            reject(
                "requires_cuda_inference",
                "program.attention_output_bridge",
                "Triton BHSD output projection requires CUDA inference",
            )
        if context.seq_len not in {32, 128, 1024} or context.d_model not in {
            32,
            128,
            1024,
        }:
            reject(
                "unsupported_shape",
                "program.attention_output_bridge",
                "Triton BHSD output projection supports S/D in {32,128,1024}",
            )
        if config.schedule.runtime is RuntimeBackend.COMPILED_FORWARD:
            reject(
                "runtime_incompatible",
                "schedule.runtime",
                "Triton BHSD output projection is not nested in compiled forward",
            )
        launch = config.schedule.attention_output_projection_launch
        assert launch is not None
        if (
            launch.block_m not in {16, 32, 64, 128}
            or launch.block_n not in {16, 32, 64, 128}
            or launch.block_k not in {16, 32, 64}
            or launch.num_warps not in {2, 4, 8}
            or launch.num_stages not in {1, 2, 3, 4}
        ):
            reject(
                "unsupported_launch_value",
                "schedule.attention_output_projection_launch",
                "Triton BHSD projection launch is outside the implemented template",
            )

    @staticmethod
    def _validate_ffn(
        config: ConfigSpec,
        context: ExecutionContext,
        hardware: HardwareCapabilities,
        reject: Any,
    ) -> None:
        backend = config.program.ffn
        if backend is FFNBackend.TORCH:
            return
        if context.device.type != "cuda":
            reject("requires_cuda", "program.ffn", "selected FFN requires CUDA")
        if not context.inference:
            reject(
                "requires_inference",
                "program.ffn",
                "selected FFN is available only for inference",
            )
        if config.schedule.runtime is RuntimeBackend.COMPILED_FORWARD:
            reject(
                "nested_compilation",
                "program.ffn",
                "specialized FFN cannot be nested in compiled forward",
            )
        if backend is FFNBackend.COMPILED:
            if not hardware.torch_compile:
                reject(
                    "backend_unavailable",
                    "program.ffn",
                    "torch.compile is unavailable",
                )
            return
        if backend is FFNBackend.TRITON_LINEAR_EXACT_GELU:
            if not hardware.triton_linear_exact_gelu:
                reject(
                    "backend_unavailable",
                    "program.ffn",
                    "Triton Linear + Exact-GELU is unavailable",
                )
            if config.program.ffn_input_projection not in {
                ProjectionBackend.INPUT_DTYPE,
                ProjectionBackend.FP16_SHADOW,
            } or (
                config.program.ffn_output_projection
                is not ProjectionBackend.FP16_SHADOW
            ):
                reject(
                    "backend_incompatible",
                    "program.ffn_input_projection",
                    "fused Linear + Exact-GELU requires native or FP16-shadow input "
                    "weights and FP16-shadow output weights",
                )
            if context.d_model not in {32, 128, 1024} or context.ffn_dim not in {
                32,
                128,
                1024,
            }:
                reject(
                    "unsupported_shape",
                    "program.ffn",
                    "fused Linear + Exact-GELU supports D/FFN in {32,128,1024}",
                )
            launch = config.schedule.ffn_input_launch
            assert launch is not None
            if (
                launch.block_m not in {16, 32, 64, 128}
                or launch.block_n not in {16, 32, 64, 128}
                or launch.block_k not in {16, 32, 64, 128}
                or launch.num_warps not in {1, 2, 4, 8}
                or launch.num_stages not in {1, 2, 3, 4}
            ):
                reject(
                    "unsupported_launch_value",
                    "schedule.ffn_input_launch",
                    "fused Linear + Exact-GELU launch is outside the implemented template",
                )
            return
        if backend is not FFNBackend.TRITON_EXACT_GELU:
            raise AssertionError(f"unhandled FFN backend: {backend}")
        if not hardware.triton_exact_gelu:
            reject(
                "backend_unavailable",
                "program.ffn",
                "Triton Exact-GELU is unavailable",
            )
        if config.program.precision_plan is not PrecisionPlan.FP16_FFN_OUTPUT:
            reject(
                "precision_incompatible",
                "program.precision_plan",
                "Triton Exact-GELU requires the FP16 FFN output plan",
            )
        if context.dtype != torch.float32:
            reject(
                "requires_float32_input",
                "program.ffn",
                "fused Exact-GELU cast requires FP32 FFN input",
            )
        launch = config.schedule.ffn_launch
        assert launch is not None
        if launch.block_size not in {128, 256, 512, 1024}:
            reject(
                "unsupported_launch_value",
                "schedule.ffn_launch.block_size",
                "Triton Exact-GELU supports block sizes 128, 256, 512, and 1024",
            )
        if launch.num_warps not in {1, 2, 4, 8}:
            reject(
                "unsupported_launch_value",
                "schedule.ffn_launch.num_warps",
                "Triton Exact-GELU supports 1, 2, 4, or 8 warps",
            )

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
        if context.has_valid_token_mask and backend is ResidualNormBackend.COMPILED:
            reject(
                "mask_not_supported",
                "program.residual_norm",
                "compiled residual norm does not apply token masking",
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
        exact_d32 = (
            not context.has_valid_token_mask
            and context.batch_size == 64
            and context.seq_len == 128
            and context.d_model == 32
            and hardware.triton_d32_residual_norm
        )
        allowed_rows = {4, 8, 16} if exact_d32 else {1, 2, 4, 8}
        if launch is not None and (
            launch.block_rows not in allowed_rows
            or launch.num_warps not in {1, 2, 4, 8}
        ):
            reject(
                "unsupported_launch_value",
                "schedule.residual_norm_launch",
                "residual norm launch is outside the selected Triton template",
            )
        if context.has_valid_token_mask and not hardware.triton_masked_norm:
            reject(
                "backend_unavailable",
                "program.residual_norm",
                "mask-aware Triton residual norm is unavailable",
            )
        if backend is ResidualNormBackend.TRITON:
            if not hardware.triton_residual_norm:
                reject(
                    "backend_unavailable",
                    "program.residual_norm",
                    "Triton residual norm is unavailable",
                )
            if context.d_model not in {32, 128, 1024}:
                reject(
                    "unsupported_shape",
                    "program.residual_norm",
                    "Triton residual norm supports D32, D128, and D1024",
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
            if context.d_model not in {32, 128, 1024}:
                reject(
                    "unsupported_shape",
                    "program.residual_norm",
                    "mixed Triton norm supports D32, D128, and D1024",
                )
            fp16_outputs = {
                ProjectionBackend.AUTOCAST_FP16,
                ProjectionBackend.FP16_SHADOW,
            }
            if (
                config.program.attention_output_projection not in fp16_outputs
                or config.program.ffn_output_projection not in fp16_outputs
            ):
                reject(
                    "requires_fp16_update",
                    "program.precision_plan",
                    "mixed Triton norm requires FP16 attention and FFN updates",
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
        if context.has_valid_token_mask and not hardware.triton_masked_norm:
            reject(
                "backend_unavailable",
                "program.initial_norm",
                "mask-aware Triton initial norm is unavailable",
            )
        if context.device.type != "cuda" or context.dtype != torch.float32:
            reject(
                "requires_cuda_float32",
                "program.initial_norm",
                "Triton initial norm requires CUDA FP32 input",
            )
        if context.d_model not in {32, 128, 1024}:
            reject(
                "unsupported_shape",
                "program.initial_norm",
                "Triton initial norm supports D32, D128, and D1024",
            )
        if config.schedule.runtime is RuntimeBackend.COMPILED_FORWARD:
            reject(
                "runtime_incompatible",
                "schedule.runtime",
                "Triton initial norm is not nested in compiled forward",
            )
        if config.program.precision_plan is not PrecisionPlan.FP16_CORE:
            reject(
                "precision_incompatible",
                "program.precision_plan",
                "Triton initial norm requires the full FP16 core plan",
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
        hardware: HardwareCapabilities,
    ) -> ExecutionPlan:
        attention_backend = config.program.attention
        qkv_projection = config.program.qkv_projection
        attention_output_projection = config.program.attention_output_projection
        ffn_input_projection = config.program.ffn_input_projection
        ffn_output_projection = config.program.ffn_output_projection
        qkv_projection_dtype = PlanBuilder._projection_dtype(
            qkv_projection,
            inner_context,
        )
        attention_output_projection_dtype = PlanBuilder._projection_dtype(
            attention_output_projection,
            inner_context,
        )
        ffn_input_projection_dtype = PlanBuilder._projection_dtype(
            ffn_input_projection,
            inner_context,
        )
        ffn_output_projection_dtype = PlanBuilder._projection_dtype(
            ffn_output_projection,
            inner_context,
        )
        ffn_activation_output_dtype = (
            "float16"
            if config.program.ffn
            in {
                FFNBackend.TRITON_EXACT_GELU,
                FFNBackend.TRITON_LINEAR_EXACT_GELU,
            }
            else ffn_input_projection_dtype
        )
        attention_compute_dtype = (
            "float16"
            if attention_backend
            in {
                AttentionBackend.FP16_EFFICIENT_SDPA,
                AttentionBackend.FP16_CUDNN_SDPA,
                AttentionBackend.TRITON_SHAPE13,
                AttentionBackend.TRITON_DH8,
                AttentionBackend.TRITON_STREAMING_DH64,
            }
            else (qkv_projection_dtype)
        )
        use_d32_residual_norm = bool(
            config.program.residual_norm
            in {ResidualNormBackend.TRITON, ResidualNormBackend.TRITON_MIXED}
            and not inner_context.has_valid_token_mask
            and inner_context.batch_size == 64
            and inner_context.seq_len == 128
            and inner_context.d_model == 32
            and hardware.triton_d32_residual_norm
        )
        use_masked_residual_norm = bool(
            config.program.residual_norm
            in {ResidualNormBackend.TRITON, ResidualNormBackend.TRITON_MIXED}
            and inner_context.has_valid_token_mask
            and hardware.triton_masked_norm
        )
        use_masked_initial_norm = bool(
            config.program.initial_norm is InitialNormBackend.TRITON_FP16
            and inner_context.has_valid_token_mask
            and hardware.triton_masked_norm
        )
        layers = max(0, int(inner_context.num_layers or 0))
        expected = ExpectedExecutionTrace(
            runtime_backend=config.schedule.runtime,
            attention_backend=attention_backend,
            qkv_projection_backend=qkv_projection,
            attention_output_projection_backend=attention_output_projection,
            ffn_input_projection_backend=ffn_input_projection,
            ffn_output_projection_backend=ffn_output_projection,
            precision_plan=config.program.precision_plan,
            qkv_materialization=config.program.qkv_materialization,
            attention_output_bridge=config.program.attention_output_bridge,
            attention_output_layout=config.attention_output_layout,
            ffn_backend=config.program.ffn,
            residual_norm_backend=config.program.residual_norm,
            initial_norm_backend=config.program.initial_norm,
            attention_compute_dtype=attention_compute_dtype,
            qkv_projection_compute_dtype=qkv_projection_dtype,
            attention_output_projection_compute_dtype=(
                attention_output_projection_dtype
            ),
            ffn_input_projection_compute_dtype=ffn_input_projection_dtype,
            ffn_activation_output_dtype=ffn_activation_output_dtype,
            ffn_output_projection_compute_dtype=ffn_output_projection_dtype,
            attention_calls=layers,
            qkv_projection_calls=layers,
            qkv_materialization_calls=layers,
            attention_output_bridge_calls=layers,
            attention_output_projection_calls=layers,
            ffn_calls=layers,
            ffn_input_projection_calls=layers,
            ffn_output_projection_calls=layers,
            residual_norm_calls=2 * layers,
            initial_norm_calls=1 if layers else 0,
            runtime_calls=1,
            attention_launch=config.schedule.attention_launch,
            qkv_launch=config.schedule.qkv_launch,
            attention_output_projection_launch=(
                config.schedule.attention_output_projection_launch
            ),
            residual_norm_launch=config.schedule.residual_norm_launch,
            initial_norm_launch=config.schedule.initial_norm_launch,
            ffn_launch=config.schedule.ffn_launch,
            ffn_input_launch=config.schedule.ffn_input_launch,
        )
        return ExecutionPlan(
            config=config,
            outer_context=outer_context,
            inner_context=inner_context,
            attention_backend=attention_backend,
            attention_compute_dtype=attention_compute_dtype,
            qkv_projection_backend=qkv_projection,
            qkv_projection_compute_dtype=qkv_projection_dtype,
            qkv_materialization=config.program.qkv_materialization,
            attention_output_bridge=config.program.attention_output_bridge,
            attention_output_layout=config.attention_output_layout,
            attention_output_projection_backend=attention_output_projection,
            attention_output_projection_compute_dtype=(
                attention_output_projection_dtype
            ),
            ffn_backend=config.program.ffn,
            ffn_input_projection_backend=ffn_input_projection,
            ffn_input_projection_compute_dtype=ffn_input_projection_dtype,
            ffn_activation_output_dtype=ffn_activation_output_dtype,
            ffn_output_projection_backend=ffn_output_projection,
            ffn_output_projection_compute_dtype=ffn_output_projection_dtype,
            precision_plan=config.program.precision_plan,
            residual_norm_backend=config.program.residual_norm,
            initial_norm_backend=config.program.initial_norm,
            runtime_backend=config.schedule.runtime,
            compile_mode=config.schedule.compile_mode,
            batch_tile_size=config.schedule.batch_tile_size,
            microbatch_size=config.schedule.microbatch_size,
            reuse_unchanged_input=config.schedule.reuse_unchanged_input,
            attention_launch=config.schedule.attention_launch,
            qkv_launch=config.schedule.qkv_launch,
            attention_output_projection_launch=(
                config.schedule.attention_output_projection_launch
            ),
            residual_norm_launch=config.schedule.residual_norm_launch,
            initial_norm_launch=config.schedule.initial_norm_launch,
            ffn_launch=config.schedule.ffn_launch,
            ffn_input_launch=config.schedule.ffn_input_launch,
            use_d32_residual_norm=use_d32_residual_norm,
            use_masked_residual_norm=use_masked_residual_norm,
            use_masked_initial_norm=use_masked_initial_norm,
            expected_trace=expected,
        )

    @staticmethod
    def _projection_dtype(
        backend: ProjectionBackend,
        context: ExecutionContext,
    ) -> str:
        if backend in {
            ProjectionBackend.AUTOCAST_FP16,
            ProjectionBackend.FP16_SHADOW,
        }:
            return "float16"
        return context.dtype_name


__all__ = [
    "CompilationResult",
    "CompileRejection",
    "ConfigRejectedError",
    "ConstraintViolation",
    "HardwareCapabilities",
    "PlanBuilder",
]
