"""Official-compatible Transformer with a compact, plan-driven GPU path."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from policy_registry import ResidualNormBackend, RuntimeWrapper, get_policy_spec

from .batch_tiled_graph import BatchTiledGraphReplay
from .compiled_forward import CompiledForward
from .cuda_graph import CudaGraphReplay
from .dispatch import OfflineDispatcher
from .execution_plan import (
    ExecutionContext,
    ExecutionPlan,
    resolve_execution_plan,
)
from .kernels import (
    causal_sdpa,
    mixed_fp16_cudnn_attention,
    mixed_fp16_efficient_attention,
    prevalidated_mixed_fp16_efficient_attention,
    reference_causal_attention,
    residual_add,
    residual_layer_norm,
    split_qkv,
    triton_residual_layer_norm,
)


@dataclass(slots=True)
class _ExecutionObservation:
    attention_backends: list[str] = field(default_factory=list)
    residual_norm_backends: list[str] = field(default_factory=list)
    attention_compute_dtypes: list[str] = field(default_factory=list)
    linear_backends: list[str] = field(default_factory=list)
    linear_compute_dtypes: list[str] = field(default_factory=list)

    def describe(self, expected_layers: int) -> dict[str, Any]:
        complete = (
            len(self.attention_backends) == expected_layers
            and len(self.residual_norm_backends) == 2 * expected_layers
        )
        mixed_core_observed = bool(
            self.attention_compute_dtypes
            or self.linear_backends
            or self.linear_compute_dtypes
        )
        if mixed_core_observed:
            complete = complete and (
                len(self.attention_compute_dtypes) == expected_layers
                and len(self.linear_backends) == 4 * expected_layers
                and len(self.linear_compute_dtypes) == 4 * expected_layers
            )
        result = {
            "attention_backends": list(self.attention_backends),
            "residual_norm_backends": list(self.residual_norm_backends),
            "expected_layers": expected_layers,
            "complete": complete,
        }
        if mixed_core_observed:
            result.update(
                attention_compute_dtypes=list(self.attention_compute_dtypes),
                linear_backends=list(self.linear_backends),
                linear_compute_dtypes=list(self.linear_compute_dtypes),
            )
        return result


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
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(
        self,
        value: torch.Tensor,
        valid_token_mask: torch.Tensor | None,
        causal: bool,
        plan: ExecutionPlan,
        observation: _ExecutionObservation | None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = value.shape
        mixed_core = plan.linear_backend == "autocast_fp16"
        with torch.autocast(
            device_type=value.device.type,
            dtype=torch.float16,
            enabled=mixed_core,
        ):
            packed_qkv = self.qkv_proj(value)
            query, key, projected_value = split_qkv(
                packed_qkv,
                self.num_heads,
            )
            if plan.attention_backend == "mixed_fp16_cudnn":
                context, actual_attention_backend = mixed_fp16_cudnn_attention(
                    query,
                    key,
                    projected_value,
                    valid_token_mask,
                    scale=self.scale,
                    causal=causal,
                    training=self.training,
                )
            elif plan.attention_backend == "mixed_fp16_efficient":
                if plan.use_compiled_forward:
                    context, actual_attention_backend = (
                        prevalidated_mixed_fp16_efficient_attention(
                            query,
                            key,
                            projected_value,
                            scale=self.scale,
                        )
                    )
                else:
                    context, actual_attention_backend = mixed_fp16_efficient_attention(
                        query,
                        key,
                        projected_value,
                        valid_token_mask,
                        scale=self.scale,
                        causal=causal,
                        training=self.training,
                    )
            elif plan.attention_backend == "causal_sdpa":
                context = causal_sdpa(
                    query,
                    key,
                    projected_value,
                    valid_token_mask,
                    scale=self.scale,
                    causal=causal,
                )
                actual_attention_backend = "causal_sdpa"
            else:
                context = reference_causal_attention(
                    query,
                    key,
                    projected_value,
                    valid_token_mask,
                    scale=self.scale,
                    causal=causal,
                )
                actual_attention_backend = "safe_streaming"
            context = (
                context.transpose(1, 2)
                .contiguous()
                .view(batch_size, sequence_length, self.d_model)
            )
            output = self.out_proj(context)
        if observation is not None:
            observation.attention_backends.append(actual_attention_backend)
            if mixed_core:
                observation.attention_compute_dtypes.append(
                    str(query.dtype).removeprefix("torch.")
                )
                observation.linear_backends.extend([plan.linear_backend] * 2)
                observation.linear_compute_dtypes.extend(
                    [
                        str(packed_qkv.dtype).removeprefix("torch."),
                        str(output.dtype).removeprefix("torch."),
                    ]
                )
        if output.dtype != value.dtype:
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
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

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

        mixed_core = plan.linear_backend == "autocast_fp16"
        with torch.autocast(
            device_type=normalized.device.type,
            dtype=torch.float16,
            enabled=mixed_core,
        ):
            projected = self.ffn_in(normalized)
            hidden = F.gelu(projected, approximate="none")
            ffn_update = self.ffn_out(hidden)
        if observation is not None and mixed_core:
            observation.linear_backends.extend([plan.linear_backend] * 2)
            observation.linear_compute_dtypes.extend(
                [
                    str(projected.dtype).removeprefix("torch."),
                    str(ffn_update.dtype).removeprefix("torch."),
                ]
            )
        if ffn_update.dtype != normalized.dtype:
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

        self._dispatcher: OfflineDispatcher | None = None
        self._dispatch_signature: tuple[object, ...] | None = None
        self.dispatch_policy: str | None = None
        self.dispatch_route_origin: str | None = None
        self.dispatch_route_source: str | None = None
        self.dispatch_route_sha256: str | None = None
        self._enable_dispatch()

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

    def _enable_dispatch(self) -> None:
        self._dispatcher = OfflineDispatcher()
        self._dispatch_signature = None
        self._select_named_policy("eager-sdpa", requested_policy="dispatch")
        self.dispatch_policy = "eager-sdpa"
        self.dispatch_route_origin = "fallback"
        self.dispatch_route_source = self._dispatcher.source
        self.dispatch_route_sha256 = self._dispatcher.table_sha256
        if next(self.parameters()).is_cuda:
            self._resolve_dispatch()

    def _apply(self, function: Any, recurse: bool = True) -> UserOptimizedTransformer:
        result = super()._apply(function, recurse=recurse)
        self._invalidate_runtime_state()
        self._dispatch_signature = None
        if self._dispatcher is not None and next(self.parameters()).is_cuda:
            self._resolve_dispatch()
        return result

    def _resolve_dispatch(self, value: torch.Tensor | None = None) -> None:
        if self._dispatcher is None:
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
        signature = (
            device.type,
            device.index,
            dtype,
            shape,
            matmul_precision,
            allow_tf32,
        )
        if signature == self._dispatch_signature:
            return
        resolution = self._dispatcher.resolve_result(
            self.config,
            device=device,
            dtype=dtype,
            shape=shape,
            matmul_precision=matmul_precision,
            allow_tf32=allow_tf32,
        )
        self._select_named_policy(resolution.policy, requested_policy="dispatch")
        self.dispatch_policy = resolution.policy
        self.dispatch_route_origin = resolution.origin
        self.dispatch_route_source = resolution.source
        self.dispatch_route_sha256 = resolution.table_sha256
        self._dispatch_signature = signature

    def _select_named_policy(
        self,
        policy: str,
        *,
        requested_policy: str | None = None,
    ) -> None:
        spec = get_policy_spec(policy)
        self._invalidate_runtime_state()
        self._active_policy_id = spec.policy_id
        self.requested_policy = requested_policy or spec.policy_id

    def configure_runtime_policy(self, *, policy: str) -> None:
        """Select one policy, or restore deterministic offline dispatch."""

        if policy.strip().lower() == "dispatch":
            self._enable_dispatch()
            return
        get_policy_spec(policy)
        self._dispatcher = None
        self._dispatch_signature = None
        self.dispatch_policy = None
        self.dispatch_route_origin = None
        self.dispatch_route_source = None
        self.dispatch_route_sha256 = None
        self._select_named_policy(policy)

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
        return resolve_execution_plan(
            get_policy_spec(self._active_policy_id),
            self._execution_context(value, valid_token_mask),
            requested_policy=self.requested_policy,
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
            self._active_policy_id,
            self.requested_policy,
            self.dispatch_policy,
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
        description = plan.describe(
            dispatch_source=self.dispatch_route_source,
            dispatch_table_sha256=self.dispatch_route_sha256,
            dispatch_policy=self.dispatch_policy,
            route_origin=self.dispatch_route_origin,
            causal=bool(self.config.causal),
        )
        if self._last_execution_observation is not None:
            description["observed_execution"] = self._last_execution_observation
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
            _ExecutionObservation()
            if observe
            and self._execution_observation_enabled
            and not torch.compiler.is_compiling()
            else None
        )
        if not self.layers:
            value = self.final_norm(value)
        else:
            normalized = self.layers[0].norm1(value)
            for layer_index, layer in enumerate(self.layers):
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one residual boundary and its following LayerNorm."""

        if plan.residual_norm_backend == ResidualNormBackend.COMPILED:
            value, normalized, actual_backend = residual_layer_norm(
                value,
                update,
                layer_norm,
            )
        elif plan.residual_norm_backend == ResidualNormBackend.TRITON:
            value, normalized, actual_backend = triton_residual_layer_norm(
                value,
                update,
                layer_norm,
            )
        else:
            value = residual_add(value, update, valid_token_mask)
            normalized = layer_norm(value)
            actual_backend = "torch"
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
            if (
                get_policy_spec(self._active_policy_id).runtime
                is not RuntimeWrapper.EAGER
            ):
                raise RuntimeError(
                    "a Solution-owned runtime cannot run under torch.compile"
                )
            valid_token_mask = self._effective_valid_token_mask(x, valid_token_mask)
            plan = self._execution_plan(x, valid_token_mask)
        else:
            self._resolve_dispatch(x)
            valid_token_mask = self._effective_valid_token_mask(x, valid_token_mask)
            plan = self._cached_execution_plan(x, valid_token_mask)

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
                self._last_execution_observation["runtime_wrappers"] = [
                    "batch_tiled_cuda_graph"
                ]
            return output
        if plan.use_compiled_forward:
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
            )
            if (
                self._execution_observation_enabled
                and self._last_execution_observation is not None
            ):
                self._last_execution_observation["runtime_wrappers"] = [
                    "compiled_forward"
                ]
            return output
        if plan.use_cuda_graph:
            if self._cuda_graph_replay is None:
                self._cuda_graph_replay = CudaGraphReplay()
            output = self._cuda_graph_replay.run(
                lambda value, mask: self._forward_eager(value, mask, plan),
                x,
                valid_token_mask,
            )
            if (
                self._execution_observation_enabled
                and self._last_execution_observation is not None
            ):
                self._last_execution_observation["runtime_wrappers"] = ["cuda_graph"]
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
