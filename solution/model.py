"""Official-compatible Transformer with a compact, plan-driven GPU path."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import (
    AttentionBackend,
    AttentionOutputLayout,
    ConfigSpec,
    InitialNormBackend,
    LinearBackend,
    ResidualNormBackend,
    RuntimeBackend,
    portable_config,
)
from .operators import (
    MIXED_FP16_CUDNN_BACKEND,
    MIXED_FP16_EFFICIENT_BACKEND,
    TRITON_DH8_CAUSAL_ATTENTION_BSD_BACKEND,
    TRITON_MIXED_RESIDUAL_LAYER_NORM_BACKEND,
    TRITON_RESIDUAL_LAYER_NORM_BACKEND,
    TRITON_SHAPE13_CAUSAL_ATTENTION_BACKEND,
    causal_sdpa,
    mixed_fp16_cudnn_attention,
    mixed_fp16_efficient_attention,
    prevalidated_mixed_fp16_efficient_attention,
    prevalidated_triton_dh8_causal_attention_bsd,
    prevalidated_triton_shape13_causal_attention,
    reference_causal_attention,
    residual_add,
    residual_layer_norm,
    split_qkv,
    triton_initial_fp16_layer_norm,
    triton_mixed_residual_layer_norm,
    triton_residual_layer_norm,
)
from .plan import ExecutionContext, ExecutionPlan
from .plan_builder import PlanBuilder
from .runtimes import BatchTiledGraphReplay, CompiledForward, CudaGraphReplay


@dataclass(slots=True)
class _ExecutionObservation:
    config_id: str
    runtime_backend: str = RuntimeBackend.EAGER.value
    attention_backends: list[str] = field(default_factory=list)
    residual_norm_backends: list[str] = field(default_factory=list)
    initial_norm_backends: list[str] = field(default_factory=list)
    attention_compute_dtypes: list[str] = field(default_factory=list)
    linear_backends: list[str] = field(default_factory=list)
    linear_compute_dtypes: list[str] = field(default_factory=list)
    layer_input_dtypes: list[str] = field(default_factory=list)
    layer_output_dtypes: list[str] = field(default_factory=list)

    @staticmethod
    def _uniform(values: list[str]) -> str | None:
        if not values:
            return None
        first = values[0]
        return first if all(value == first for value in values) else None

    def describe(self, expected_layers: int) -> dict[str, Any]:
        complete = (
            len(self.attention_backends) == expected_layers
            and len(self.residual_norm_backends) == 2 * expected_layers
            and len(self.initial_norm_backends) == (1 if expected_layers else 0)
            and len(self.attention_compute_dtypes) == expected_layers
            and len(self.linear_backends) == 4 * expected_layers
            and len(self.linear_compute_dtypes) == 4 * expected_layers
            and len(self.layer_input_dtypes) == expected_layers
            and len(self.layer_output_dtypes) == expected_layers
        )
        return {
            "config_id": self.config_id,
            "runtime_backend": self.runtime_backend,
            "attention_backend": self._uniform(self.attention_backends),
            "linear_backend": self._uniform(self.linear_backends),
            "residual_norm_backend": self._uniform(self.residual_norm_backends),
            "initial_norm_backend": self._uniform(self.initial_norm_backends),
            "attention_compute_dtype": self._uniform(self.attention_compute_dtypes),
            "linear_compute_dtype": self._uniform(self.linear_compute_dtypes),
            "attention_calls": len(self.attention_backends),
            "linear_calls": len(self.linear_backends),
            "residual_norm_calls": len(self.residual_norm_backends),
            "initial_norm_calls": len(self.initial_norm_backends),
            "runtime_calls": 1,
            "complete": complete,
        }


def _strict_backend_marker(
    actual: str,
    *,
    expected_marker: str,
    planned_backend: str,
) -> str:
    """Normalize a successful primitive marker or expose an implementation drift."""

    if actual != expected_marker:
        raise RuntimeError(
            "execution backend marker does not match the compiled plan: "
            f"expected marker {expected_marker!r}, received {actual!r}"
        )
    return planned_backend


class _FP16ShadowLinear(nn.Linear):
    """Linear layer with optional non-persistent FP16 inference weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("_fp16_shadow_weight", None, persistent=False)
        self.register_buffer("_fp16_shadow_bias", None, persistent=False)

    def materialize_fp16_shadow(self) -> None:
        """Create immutable inference shadows from the current FP32 parameters."""

        if self._fp16_shadow_weight is not None:
            return
        self._fp16_shadow_weight = self.weight.detach().to(torch.float16).contiguous()
        self._fp16_shadow_bias = (
            None
            if self.bias is None
            else self.bias.detach().to(torch.float16).contiguous()
        )

    def clear_fp16_shadow(self) -> None:
        """Discard derived tensors after parameters move or are replaced."""

        self._fp16_shadow_weight = None
        self._fp16_shadow_bias = None

    def forward_fp16_shadow(self, value: torch.Tensor) -> torch.Tensor:
        """Run explicit FP16 linear algebra without autocast weight conversion."""

        weight = self._fp16_shadow_weight
        if weight is None:
            raise RuntimeError("FP16 shadow weights must be materialized before use")
        return F.linear(value.to(torch.float16), weight, self._fp16_shadow_bias)


class _SelfAttention(nn.Module):
    """Packed QKV projection followed by the planned causal attention path."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_proj = _FP16ShadowLinear(d_model, 3 * d_model, bias=True)
        self.out_proj = _FP16ShadowLinear(d_model, d_model, bias=True)

    def forward(
        self,
        value: torch.Tensor,
        valid_token_mask: torch.Tensor | None,
        causal: bool,
        plan: ExecutionPlan,
        observation: _ExecutionObservation | None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = value.shape
        autocast_core = plan.linear_backend is LinearBackend.AUTOCAST_FP16
        shadow_core = plan.linear_backend is LinearBackend.FP16_SHADOW
        with torch.autocast(
            device_type=value.device.type,
            dtype=torch.float16,
            enabled=autocast_core,
        ):
            packed_qkv = (
                self.qkv_proj.forward_fp16_shadow(value)
                if shadow_core
                else self.qkv_proj(value)
            )
            query, key, projected_value = split_qkv(
                packed_qkv,
                self.num_heads,
            )
            if plan.attention_backend is AttentionBackend.TRITON_DH8:
                launch = plan.attention_launch
                if launch is None:
                    raise RuntimeError("Triton Dh8 plan is missing launch parameters")
                context, actual_marker = prevalidated_triton_dh8_causal_attention_bsd(
                    query,
                    key,
                    projected_value,
                    scale=self.scale,
                    block_m=launch.block_m,
                    block_n=launch.block_n,
                    num_warps=launch.num_warps,
                    num_stages=launch.num_stages,
                )
                actual_attention_backend = _strict_backend_marker(
                    actual_marker,
                    expected_marker=TRITON_DH8_CAUSAL_ATTENTION_BSD_BACKEND,
                    planned_backend=plan.attention_backend.value,
                )
            elif plan.attention_backend is AttentionBackend.TRITON_SHAPE13:
                launch = plan.attention_launch
                if launch is None:
                    raise RuntimeError(
                        "Triton Shape 13 plan is missing launch parameters"
                    )
                context, actual_marker = prevalidated_triton_shape13_causal_attention(
                    query,
                    key,
                    projected_value,
                    scale=self.scale,
                    block_m=launch.block_m,
                    block_n=launch.block_n,
                    num_warps=launch.num_warps,
                    num_stages=launch.num_stages,
                )
                actual_attention_backend = _strict_backend_marker(
                    actual_marker,
                    expected_marker=TRITON_SHAPE13_CAUSAL_ATTENTION_BACKEND,
                    planned_backend=plan.attention_backend.value,
                )
            elif plan.attention_backend is AttentionBackend.FP16_CUDNN_SDPA:
                context, actual_marker = mixed_fp16_cudnn_attention(
                    query,
                    key,
                    projected_value,
                    valid_token_mask,
                    scale=self.scale,
                    causal=causal,
                    training=self.training,
                )
                actual_attention_backend = _strict_backend_marker(
                    actual_marker,
                    expected_marker=MIXED_FP16_CUDNN_BACKEND,
                    planned_backend=plan.attention_backend.value,
                )
            elif plan.attention_backend is AttentionBackend.FP16_EFFICIENT_SDPA:
                if plan.use_compiled_forward:
                    context, actual_marker = (
                        prevalidated_mixed_fp16_efficient_attention(
                            query,
                            key,
                            projected_value,
                            scale=self.scale,
                        )
                    )
                else:
                    context, actual_marker = mixed_fp16_efficient_attention(
                        query,
                        key,
                        projected_value,
                        valid_token_mask,
                        scale=self.scale,
                        causal=causal,
                        training=self.training,
                    )
                actual_attention_backend = _strict_backend_marker(
                    actual_marker,
                    expected_marker=MIXED_FP16_EFFICIENT_BACKEND,
                    planned_backend=plan.attention_backend.value,
                )
            elif plan.attention_backend is AttentionBackend.CAUSAL_SDPA:
                context = causal_sdpa(
                    query,
                    key,
                    projected_value,
                    valid_token_mask,
                    scale=self.scale,
                    causal=causal,
                )
                actual_attention_backend = plan.attention_backend.value
            elif plan.attention_backend is AttentionBackend.REFERENCE_STREAMING:
                context = reference_causal_attention(
                    query,
                    key,
                    projected_value,
                    valid_token_mask,
                    scale=self.scale,
                    causal=causal,
                )
                actual_attention_backend = plan.attention_backend.value
            else:
                raise AssertionError(
                    f"unhandled attention backend: {plan.attention_backend}"
                )
            if plan.attention_output_layout is AttentionOutputLayout.BHSD:
                context = (
                    context.transpose(1, 2)
                    .contiguous()
                    .view(batch_size, sequence_length, self.d_model)
                )
            elif plan.attention_output_layout is not AttentionOutputLayout.BSD:
                raise RuntimeError(
                    f"unsupported attention output layout: "
                    f"{plan.attention_output_layout}"
                )
            output = (
                self.out_proj.forward_fp16_shadow(context)
                if shadow_core
                else self.out_proj(context)
            )
        if observation is not None:
            observation.attention_backends.append(actual_attention_backend)
            observation.attention_compute_dtypes.append(
                "float16"
                if plan.attention_backend
                in {
                    AttentionBackend.FP16_EFFICIENT_SDPA,
                    AttentionBackend.FP16_CUDNN_SDPA,
                    AttentionBackend.TRITON_SHAPE13,
                    AttentionBackend.TRITON_DH8,
                }
                else str(query.dtype).removeprefix("torch.")
            )
            observation.linear_backends.extend([plan.linear_backend.value] * 2)
            observation.linear_compute_dtypes.extend(
                [
                    str(packed_qkv.dtype).removeprefix("torch."),
                    str(output.dtype).removeprefix("torch."),
                ]
            )
        mixed_residual_stream = (
            plan.residual_norm_backend is ResidualNormBackend.TRITON_MIXED
        )
        if output.dtype != value.dtype and not mixed_residual_stream:
            output = output.to(value.dtype)
        if valid_token_mask is not None:
            output.masked_fill_(~valid_token_mask[..., None], 0)
        return output


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = _SelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = _FP16ShadowLinear(d_model, ffn_dim)
        self.ffn_out = _FP16ShadowLinear(ffn_dim, d_model)

    def attention_update(
        self,
        normalized: torch.Tensor,
        valid_token_mask: torch.Tensor | None,
        causal: bool,
        plan: ExecutionPlan,
        observation: _ExecutionObservation | None,
    ) -> torch.Tensor:
        """Compute the attention branch from an already normalized value."""

        return self.attention(
            normalized,
            valid_token_mask,
            causal,
            plan,
            observation,
        )

    def ffn_update(
        self,
        normalized: torch.Tensor,
        plan: ExecutionPlan,
        observation: _ExecutionObservation | None,
    ) -> torch.Tensor:
        """Compute the FFN branch from an already normalized value."""

        autocast_core = plan.linear_backend is LinearBackend.AUTOCAST_FP16
        shadow_core = plan.linear_backend is LinearBackend.FP16_SHADOW
        with torch.autocast(
            device_type=normalized.device.type,
            dtype=torch.float16,
            enabled=autocast_core,
        ):
            projected = (
                self.ffn_in.forward_fp16_shadow(normalized)
                if shadow_core
                else self.ffn_in(normalized)
            )
            hidden = F.gelu(projected, approximate="none")
            ffn_update = (
                self.ffn_out.forward_fp16_shadow(hidden)
                if shadow_core
                else self.ffn_out(hidden)
            )
        if observation is not None:
            observation.linear_backends.extend([plan.linear_backend.value] * 2)
            observation.linear_compute_dtypes.extend(
                [
                    str(projected.dtype).removeprefix("torch."),
                    str(ffn_update.dtype).removeprefix("torch."),
                ]
            )
        mixed_residual_stream = (
            plan.residual_norm_backend is ResidualNormBackend.TRITON_MIXED
        )
        if ffn_update.dtype != normalized.dtype and not mixed_residual_stream:
            ffn_update = ffn_update.to(normalized.dtype)
        return ffn_update


class UserOptimizedTransformer(nn.Module):
    """Solution entry with the exact official constructor and forward API."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                _TransformerBlock(
                    config.d_model,
                    config.num_heads,
                    config.ffn_dim,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

        self._cuda_graph_replay: CudaGraphReplay | None = None
        self._batch_tiled_graph_replay: BatchTiledGraphReplay | None = None
        self._compiled_forward = CompiledForward()
        self._runtime_plan_signature: tuple[object, ...] | None = None
        self._runtime_plan: ExecutionPlan | None = None
        self._execution_observation_enabled = False
        self._last_execution_observation: dict[str, Any] | None = None
        self._classified_mask: torch.Tensor | None = None
        self._classified_mask_version: int | None = None
        self._classified_mask_is_all_valid = False
        self._fp16_shadow_weights_ready = False
        self._plan_builder = PlanBuilder()
        self._explicit_config: ConfigSpec | None = None
        self._active_config = portable_config()
        self._deployment_signature: tuple[object, ...] | None = None

    def _invalidate_runtime_state(self) -> None:
        self._cuda_graph_replay = None
        self._batch_tiled_graph_replay = None
        self._compiled_forward.clear()
        self._runtime_plan_signature = None
        self._runtime_plan = None
        self._last_execution_observation = None
        self._classified_mask = None
        self._classified_mask_version = None
        self._classified_mask_is_all_valid = False

    def _apply(self, function: Any, recurse: bool = True) -> UserOptimizedTransformer:
        result = super()._apply(function, recurse=recurse)
        self._clear_fp16_shadow_weights()
        self._invalidate_runtime_state()
        self._deployment_signature = None
        if self._explicit_config is None and next(self.parameters()).is_cuda:
            self._resolve_default_config()
        return result

    def _materialize_fp16_shadow_weights(self) -> None:
        """Build all derived FP16 linear tensors once, outside compiled forward."""

        if self._fp16_shadow_weights_ready:
            return
        for module in self.modules():
            if isinstance(module, _FP16ShadowLinear):
                module.materialize_fp16_shadow()
        self._fp16_shadow_weights_ready = True

    def _clear_fp16_shadow_weights(self) -> None:
        """Invalidate every derived shadow while preserving FP32 parameters."""

        for module in self.modules():
            if isinstance(module, _FP16ShadowLinear):
                module.clear_fp16_shadow()
        self._fp16_shadow_weights_ready = False

    def _resolve_default_config(self, value: torch.Tensor | None = None) -> None:
        """Resolve one exact deployed config, otherwise select the portable config."""

        if self._explicit_config is not None:
            return
        parameter = next(self.parameters())
        device = parameter.device if value is None else value.device
        dtype = parameter.dtype if value is None else value.dtype
        shape = (
            (self.config.batch_size, self.config.seq_len, self.config.d_model)
            if value is None
            else tuple(value.shape)
        )
        matmul_precision = torch.get_float32_matmul_precision()
        allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        cudnn_allow_tf32 = bool(torch.backends.cudnn.allow_tf32)
        signature = (
            device.type,
            device.index,
            dtype,
            shape,
            matmul_precision,
            allow_tf32,
            cudnn_allow_tf32,
        )
        if signature == self._deployment_signature:
            return
        resolved_config: ConfigSpec | None = None
        if device.type == "cuda":
            from deployment.registry import (
                EnvironmentFingerprint,
                ShapeFingerprint,
                resolve_deployed_config,
            )

            hardware = EnvironmentFingerprint.detect(device)
            deployed_shape = ShapeFingerprint(
                batch_size=int(shape[0]),
                qkv_dim=int(self.config.d_model),
                heads=int(self.config.num_heads),
                seq_len=int(shape[1]),
                layers=int(self.config.num_layers),
                causal=bool(self.config.causal),
                ffn_dim=int(self.config.ffn_dim),
                dtype=str(dtype).removeprefix("torch."),
                padding_ratio=0.0,
                input_scale=1.0,
            )
            try:
                resolved_config = resolve_deployed_config(
                    hardware=hardware,
                    shape=deployed_shape,
                )
            except FileNotFoundError:
                resolved_config = None

        selected = resolved_config or portable_config()
        if selected != self._active_config:
            self._invalidate_runtime_state()
        self._active_config = selected
        self._deployment_signature = signature

    def configure_execution(self, *, config: ConfigSpec) -> None:
        """Install one explicit config after strict compilation for this model."""

        if not isinstance(config, ConfigSpec):
            raise TypeError("config must be ConfigSpec")
        self._plan_builder.build(config, self._execution_context())
        self._invalidate_runtime_state()
        self._explicit_config = config
        self._active_config = config
        self._deployment_signature = None

    def _execution_context(
        self,
        value: torch.Tensor | None = None,
        valid_token_mask: torch.Tensor | None = None,
    ) -> ExecutionContext:
        parameter = next(self.parameters())
        if value is None:
            device = parameter.device
            dtype = parameter.dtype
            shape = (self.config.batch_size, self.config.seq_len, self.config.d_model)
            input_contiguous = True
            has_valid_token_mask = False
            mask_compatible = True
            grad_enabled = False
        else:
            device = value.device
            dtype = value.dtype
            shape = tuple(value.shape)
            input_contiguous = value.is_contiguous()
            has_valid_token_mask = valid_token_mask is not None
            mask_compatible = valid_token_mask is None or (
                valid_token_mask.device == value.device
                and valid_token_mask.dtype == torch.bool
                and valid_token_mask.ndim == 2
                and tuple(valid_token_mask.shape) == tuple(value.shape[:2])
            )
            grad_enabled = torch.is_grad_enabled()
        d_model = shape[-1]
        return ExecutionContext(
            batch_size=shape[0],
            seq_len=shape[1],
            d_model=d_model,
            num_heads=self.config.num_heads,
            causal=bool(self.config.causal),
            device=device,
            dtype=dtype,
            training=self.training,
            grad_enabled=grad_enabled,
            input_contiguous=input_contiguous,
            has_valid_token_mask=has_valid_token_mask,
            mask_compatible=mask_compatible,
            ffn_dim=self.config.ffn_dim,
            num_layers=self.config.num_layers,
        )

    def _execution_plan(
        self,
        value: torch.Tensor | None = None,
        valid_token_mask: torch.Tensor | None = None,
    ) -> ExecutionPlan:
        return self._plan_builder.build(
            self._active_config,
            self._execution_context(value, valid_token_mask),
        )

    def _cached_execution_plan(
        self,
        value: torch.Tensor,
        valid_token_mask: torch.Tensor | None,
    ) -> ExecutionPlan:
        mask_signature = None
        if valid_token_mask is not None:
            mask_signature = (
                valid_token_mask.device,
                valid_token_mask.dtype,
                tuple(valid_token_mask.shape),
            )
        signature = (
            self._active_config.config_id,
            value.device,
            value.dtype,
            tuple(value.shape),
            value.is_contiguous(),
            mask_signature,
            self.training,
            torch.is_grad_enabled(),
            bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
            bool(torch.backends.cuda.cudnn_sdp_enabled()),
        )
        if signature != self._runtime_plan_signature:
            self._runtime_plan = self._execution_plan(value, valid_token_mask)
            self._runtime_plan_signature = signature
        assert self._runtime_plan is not None
        return self._runtime_plan

    def describe_execution_path(self) -> dict[str, Any]:
        """Describe the last plan, or a pure preview before first execution."""

        plan = self._runtime_plan or self._execution_plan()
        description = plan.describe()
        expected_signature = plan.expected_trace.to_dict()
        expected_signature["config_id"] = plan.config_id
        description.update(
            requested_config_id=self._active_config.config_id,
            planned_config_id=plan.config_id,
            runtime_backend=plan.runtime_backend.value,
            expected_execution_signature=expected_signature,
            observed_execution_signature=self._last_execution_observation,
        )
        return description

    def set_execution_observation(self, enabled: bool) -> None:
        self._execution_observation_enabled = bool(enabled)
        if enabled:
            self._cuda_graph_replay = None
            self._batch_tiled_graph_replay = None
            self._last_execution_observation = None

    def _forward_eager(
        self,
        value: torch.Tensor,
        valid_token_mask: torch.Tensor | None,
        plan: ExecutionPlan,
        *,
        observe: bool = True,
    ) -> torch.Tensor:
        observation = (
            _ExecutionObservation(config_id=plan.config_id)
            if observe
            and self._execution_observation_enabled
            and not torch.compiler.is_compiling()
            else None
        )
        if not self.layers:
            value = self.final_norm(value)
        else:
            if plan.use_triton_initial_fp16_norm:
                launch = plan.initial_norm_launch
                if launch is None:
                    raise RuntimeError(
                        "Triton initial norm plan is missing launch parameters"
                    )
                normalized = triton_initial_fp16_layer_norm(
                    value,
                    self.layers[0].norm1,
                    block_rows=launch.block_rows,
                    num_warps=launch.num_warps,
                )
                if observation is not None:
                    observation.initial_norm_backends.append(
                        InitialNormBackend.TRITON_FP16.value
                    )
            else:
                normalized = self.layers[0].norm1(value)
                if observation is not None:
                    observation.initial_norm_backends.append(
                        InitialNormBackend.TORCH.value
                    )
            if (
                plan.residual_norm_backend is ResidualNormBackend.TRITON_MIXED
                and not plan.use_triton_initial_fp16_norm
            ):
                normalized = normalized.to(torch.float16)
            for layer_index, layer in enumerate(self.layers):
                if observation is not None:
                    observation.layer_input_dtypes.append(
                        str(value.dtype).removeprefix("torch.")
                    )
                attention_update = layer.attention_update(
                    normalized,
                    valid_token_mask,
                    bool(self.config.causal),
                    plan,
                    observation,
                )
                value, normalized = self._apply_residual_norm(
                    value,
                    attention_update,
                    layer.norm2,
                    valid_token_mask,
                    plan,
                    observation,
                    final_boundary=False,
                )
                ffn_update = layer.ffn_update(normalized, plan, observation)
                next_norm = (
                    self.layers[layer_index + 1].norm1
                    if layer_index + 1 < len(self.layers)
                    else self.final_norm
                )
                value, normalized = self._apply_residual_norm(
                    value,
                    ffn_update,
                    next_norm,
                    valid_token_mask,
                    plan,
                    observation,
                    final_boundary=layer_index + 1 == len(self.layers),
                )
                if observation is not None:
                    observation.layer_output_dtypes.append(
                        str(normalized.dtype).removeprefix("torch.")
                    )
            value = normalized
        if valid_token_mask is not None:
            value.masked_fill_(~valid_token_mask[..., None], 0)
        if observation is not None:
            self._last_execution_observation = observation.describe(len(self.layers))
        return value

    @staticmethod
    def _apply_residual_norm(
        value: torch.Tensor,
        update: torch.Tensor,
        layer_norm: nn.LayerNorm,
        valid_token_mask: torch.Tensor | None,
        plan: ExecutionPlan,
        observation: _ExecutionObservation | None,
        *,
        final_boundary: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one residual boundary and its following LayerNorm."""

        if plan.residual_norm_backend is ResidualNormBackend.COMPILED:
            value, normalized, actual_marker = residual_layer_norm(
                value,
                update,
                layer_norm,
            )
            actual_backend = _strict_backend_marker(
                actual_marker,
                expected_marker="compiled_residual_layer_norm",
                planned_backend=plan.residual_norm_backend.value,
            )
        elif plan.residual_norm_backend is ResidualNormBackend.TRITON:
            launch = plan.residual_norm_launch
            if launch is None:
                raise RuntimeError(
                    "Triton residual norm plan is missing launch parameters"
                )
            value, normalized, actual_marker = triton_residual_layer_norm(
                value,
                update,
                layer_norm,
                block_rows=launch.block_rows,
                num_warps=launch.num_warps,
            )
            actual_backend = _strict_backend_marker(
                actual_marker,
                expected_marker=TRITON_RESIDUAL_LAYER_NORM_BACKEND,
                planned_backend=plan.residual_norm_backend.value,
            )
        elif plan.residual_norm_backend is ResidualNormBackend.TRITON_MIXED:
            launch = plan.residual_norm_launch
            if launch is None:
                raise RuntimeError(
                    "Triton mixed residual norm plan is missing launch parameters"
                )
            value, normalized, actual_marker = triton_mixed_residual_layer_norm(
                value,
                update,
                layer_norm,
                final_boundary=final_boundary,
                block_rows=launch.block_rows,
                num_warps=launch.num_warps,
            )
            actual_backend = _strict_backend_marker(
                actual_marker,
                expected_marker=TRITON_MIXED_RESIDUAL_LAYER_NORM_BACKEND,
                planned_backend=plan.residual_norm_backend.value,
            )
        elif plan.residual_norm_backend is ResidualNormBackend.TORCH:
            value = residual_add(value, update, valid_token_mask)
            normalized = layer_norm(value)
            actual_backend = plan.residual_norm_backend.value
        else:
            raise AssertionError(
                f"unhandled residual norm backend: {plan.residual_norm_backend}"
            )
        if observation is not None:
            observation.residual_norm_backends.append(actual_backend)
        return value, normalized

    def _effective_valid_token_mask(
        self,
        value: torch.Tensor,
        valid_token_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Collapse an unchanged all-valid official mask once per tensor."""

        if valid_token_mask is None or torch.compiler.is_compiling():
            return valid_token_mask
        if not (
            valid_token_mask.device == value.device
            and valid_token_mask.dtype == torch.bool
            and valid_token_mask.ndim == 2
            and tuple(valid_token_mask.shape) == tuple(value.shape[:2])
        ):
            return valid_token_mask
        try:
            version = valid_token_mask._version
        except RuntimeError:
            # Versionless inference tensors may be mutated without an
            # observable counter, so their classification is never cached.
            return None if bool(valid_token_mask.all().item()) else valid_token_mask
        if (
            self._classified_mask is not valid_token_mask
            or self._classified_mask_version != version
        ):
            self._classified_mask = valid_token_mask
            self._classified_mask_version = version
            self._classified_mask_is_all_valid = bool(valid_token_mask.all().item())
        return None if self._classified_mask_is_all_valid else valid_token_mask

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if torch.compiler.is_compiling():
            if self._active_config.schedule.runtime not in {
                RuntimeBackend.EAGER,
                RuntimeBackend.STREAMED,
            }:
                raise RuntimeError(
                    "a Solution-owned runtime cannot run under torch.compile"
                )
            valid_token_mask = self._effective_valid_token_mask(x, valid_token_mask)
            plan = self._execution_plan(x, valid_token_mask)
        else:
            self._resolve_default_config(x)
            valid_token_mask = self._effective_valid_token_mask(x, valid_token_mask)
            plan = self._cached_execution_plan(x, valid_token_mask)

        if plan.linear_backend is LinearBackend.FP16_SHADOW:
            self._materialize_fp16_shadow_weights()

        if plan.use_batch_tiled_cuda_graph:
            if plan.batch_tile_size is None:
                raise RuntimeError("batch-tiled plan is missing its tile size")
            if self._batch_tiled_graph_replay is None:
                self._batch_tiled_graph_replay = BatchTiledGraphReplay(
                    plan.batch_tile_size
                )
            output = self._batch_tiled_graph_replay.run(
                lambda value, mask: self._forward_eager(value, mask, plan),
                x,
                valid_token_mask,
            )
            if (
                self._execution_observation_enabled
                and self._last_execution_observation is not None
            ):
                self._last_execution_observation["runtime_backend"] = (
                    RuntimeBackend.BATCH_TILED_CUDA_GRAPH.value
                )
            return output
        if plan.use_compiled_forward:
            if plan.compile_mode is None:
                raise RuntimeError("compiled-forward plan is missing its compile mode")
            if self._execution_observation_enabled:
                self._forward_eager(x, valid_token_mask, plan)
            output = self._compiled_forward.run(
                lambda value, mask: self._forward_eager(
                    value,
                    mask,
                    plan,
                    observe=False,
                ),
                x,
                valid_token_mask,
                plan_key=plan,
                compile_mode=plan.compile_mode,
            )
            if (
                self._execution_observation_enabled
                and self._last_execution_observation is not None
            ):
                self._last_execution_observation["runtime_backend"] = (
                    RuntimeBackend.COMPILED_FORWARD.value
                )
            return output
        if plan.use_cuda_graph:
            if self._cuda_graph_replay is None:
                self._cuda_graph_replay = CudaGraphReplay(
                    reuse_unchanged_input=plan.reuse_unchanged_input
                )
            output = self._cuda_graph_replay.run(
                lambda value, mask: self._forward_eager(value, mask, plan),
                x,
                valid_token_mask,
            )
            if (
                self._execution_observation_enabled
                and self._last_execution_observation is not None
            ):
                self._last_execution_observation["runtime_backend"] = (
                    RuntimeBackend.CUDA_GRAPH.value
                )
            return output
        return self._forward_eager(x, valid_token_mask, plan)


def _packed_source_keys(target_key: str) -> tuple[str, str, str] | None:
    marker = ".attention.qkv_proj."
    if marker not in target_key:
        return None
    prefix, suffix = target_key.split(marker, maxsplit=1)
    attention_prefix = f"{prefix}.attention."
    return tuple(
        f"{attention_prefix}{projection}.{suffix}"
        for projection in ("q_proj", "k_proj", "v_proj")
    )


def copy_model_weights(
    baseline: nn.Module,
    optimized: nn.Module,
    strict: bool = True,
) -> None:
    """Copy official weights and derive each packed QKV tensor once."""

    source_state = baseline.state_dict()
    target_state = optimized.state_dict()
    mapped_state: dict[str, torch.Tensor] = {}
    consumed_source_keys: set[str] = set()

    for target_key, target_tensor in target_state.items():
        packed_keys = _packed_source_keys(target_key)
        if packed_keys is None:
            if target_key in source_state:
                mapped_state[target_key] = source_state[target_key]
                consumed_source_keys.add(target_key)
            continue
        if not all(source_key in source_state for source_key in packed_keys):
            continue
        packed_tensor = torch.cat(
            [source_state[source_key] for source_key in packed_keys],
            dim=0,
        )
        if packed_tensor.shape != target_tensor.shape:
            raise RuntimeError(
                f"packed tensor shape mismatch for {target_key}: "
                f"source={tuple(packed_tensor.shape)}, "
                f"target={tuple(target_tensor.shape)}"
            )
        mapped_state[target_key] = packed_tensor
        consumed_source_keys.update(packed_keys)

    missing_target_keys = sorted(set(target_state) - set(mapped_state))
    unused_source_keys = sorted(set(source_state) - consumed_source_keys)
    if strict and (missing_target_keys or unused_source_keys):
        raise RuntimeError(
            "strict weight mapping failed; "
            f"missing target keys={missing_target_keys}, "
            f"unused source keys={unused_source_keys}"
        )

    incompatible = optimized.load_state_dict(mapped_state, strict=strict)
    clear_shadow_weights = getattr(optimized, "_clear_fp16_shadow_weights", None)
    if callable(clear_shadow_weights):
        clear_shadow_weights()
    if not strict:
        warning_parts = []
        if missing_target_keys or incompatible.missing_keys:
            warning_parts.append(
                "missing target keys="
                f"{sorted(set(missing_target_keys + incompatible.missing_keys))}"
            )
        if unused_source_keys:
            warning_parts.append(f"unused source keys={unused_source_keys}")
        if incompatible.unexpected_keys:
            warning_parts.append(
                f"unexpected mapped keys={sorted(incompatible.unexpected_keys)}"
            )
        if warning_parts:
            warnings.warn(
                "non-strict weight mapping: " + ", ".join(warning_parts),
                RuntimeWarning,
                stacklevel=2,
            )


__all__ = ["UserOptimizedTransformer", "copy_model_weights"]
