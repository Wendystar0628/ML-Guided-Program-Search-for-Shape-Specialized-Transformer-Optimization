"""Transformer implementation optimized for the project performance mainline."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .cuda_graph import CudaGraphReplay
from .dispatch import OfflineDispatcher
from .execution_plan import (
    ExecutionContext,
    ExecutionPlan,
    LayerExecutionPlan,
    resolve_execution_plan,
)
from .kernels import (
    can_use_s512_native_half_softmax,
    can_use_triton_attention_preprocess,
    can_use_triton_attention_softmax,
    can_use_triton_online_attention,
    can_use_triton_qkv_layout,
    can_use_triton_residual,
    can_use_wide_exact_gelu,
    s512_scale_mask_native_half_softmax,
    triton_online_attention,
    triton_qkv_to_bhsd,
    triton_residual_add_padding,
    triton_scale_mask_softmax,
    triton_scale_mask_to_fp32,
    wide_linear_exact_gelu,
)
from .policies import get_policy_spec


@dataclass(slots=True)
class _ExecutionObservation:
    """Actual eager branches taken by one complete model forward."""

    qkv_layouts: list[str] = field(default_factory=list)
    attention_backends: list[str] = field(default_factory=list)
    ffn_backends: list[str] = field(default_factory=list)
    residual_backends: list[str] = field(default_factory=list)

    def describe(self, expected_layers: int) -> dict[str, Any]:
        """Return compact evidence suitable for the worker JSON result."""

        branch_lists = (
            self.qkv_layouts,
            self.attention_backends,
            self.ffn_backends,
            self.residual_backends,
        )
        return {
            "complete": all(len(values) == expected_layers for values in branch_lists),
            "layer_count": expected_layers,
            "qkv_layouts": list(self.qkv_layouts),
            "attention_backends": list(self.attention_backends),
            "ffn_backends": list(self.ffn_backends),
            "residual_backends": list(self.residual_backends),
        }


class _SelfAttention(nn.Module):
    """Reference-compatible attention with a one-time packed QKV projection."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads_view(self, value: torch.Tensor) -> torch.Tensor:
        """Expose the head dimension without materializing a layout copy."""

        batch_size, sequence_length, _ = value.shape
        return value.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _split_heads_contiguous(self, value: torch.Tensor) -> torch.Tensor:
        """Materialize the official head layout as a conservative candidate."""

        return self._split_heads_view(value).contiguous()

    def _can_use_fp32_sdpa(
        self,
        value: torch.Tensor,
        sequence_length: int,
        causal: bool,
    ) -> bool:
        return (
            value.is_cuda
            and value.dtype == torch.float32
            and not causal
            and sequence_length <= 128
        )

    def _project_heads(
        self,
        value: torch.Tensor,
        plan: LayerExecutionPlan,
        observation: _ExecutionObservation | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project QKV once through the implementation selected by the plan."""

        packed_qkv = self.qkv_proj(value)
        if plan.qkv_layout == "triton_single_pass" and can_use_triton_qkv_layout(
            packed_qkv,
            self.num_heads,
        ):
            packed_heads = triton_qkv_to_bhsd(packed_qkv, self.num_heads)
            query, key, projected_value = packed_heads.unbind(dim=0)
            if observation is not None:
                observation.qkv_layouts.append("triton_single_pass")
            return query, key, projected_value

        query, key, projected_value = packed_qkv.chunk(3, dim=-1)
        split_heads = (
            self._split_heads_contiguous
            if plan.qkv_layout == "torch_three_contiguous_copies"
            else self._split_heads_view
        )
        if observation is not None:
            if plan.qkv_layout == "torch_three_contiguous_copies":
                actual_layout = "torch_three_contiguous_copies"
            elif plan.qkv_layout == "triton_single_pass":
                actual_layout = "view_fallback"
            else:
                actual_layout = "torch_zero_copy_view"
            observation.qkv_layouts.append(actual_layout)
        return (
            split_heads(query),
            split_heads(key),
            split_heads(projected_value),
        )

    def _explicit_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        projected_value: torch.Tensor,
        valid_token_mask: torch.Tensor | None,
        score_mask: torch.Tensor | None,
        causal: bool,
        plan: LayerExecutionPlan,
        observation: _ExecutionObservation | None,
    ) -> torch.Tensor:
        """Preserve the official low-precision operation and accumulation order."""

        sequence_length = query.shape[-2]
        if plan.attention == "triton_two_pass_online_attention" and (
            can_use_triton_online_attention(
                query,
                key,
                projected_value,
                valid_token_mask,
            )
        ):
            if observation is not None:
                observation.attention_backends.append(
                    "triton_two_pass_online_attention"
                )
            return triton_online_attention(
                query,
                key,
                projected_value,
                valid_token_mask,
                scale=self.scale,
                causal=causal,
            )
        scores = torch.matmul(query, key.transpose(-2, -1))
        use_s512_native_softmax = (
            plan.attention == "explicit_qk_triton_scale_mask_native_half_softmax_pv"
            and can_use_s512_native_half_softmax(
                scores,
                valid_token_mask,
                self.head_dim,
            )
        )
        if use_s512_native_softmax:
            if observation is not None:
                observation.attention_backends.append(
                    "explicit_qk_triton_scale_mask_native_half_softmax_pv"
                )
            probabilities = s512_scale_mask_native_half_softmax(
                scores,
                valid_token_mask,
                head_dim=self.head_dim,
                scale=self.scale,
                causal=causal,
            )
            return torch.matmul(probabilities, projected_value)
        use_triton_softmax = (
            plan.attention == "explicit_qk_triton_softmax_pv"
            and can_use_triton_attention_softmax(
                scores,
                valid_token_mask,
                self.head_dim,
            )
        )
        if use_triton_softmax:
            if observation is not None:
                observation.attention_backends.append("explicit_qk_triton_softmax_pv")
            probabilities = triton_scale_mask_softmax(
                scores,
                valid_token_mask,
                head_dim=self.head_dim,
                scale=self.scale,
                causal=causal,
            )
            return torch.matmul(probabilities, projected_value)
        else:
            use_triton_preprocess = (
                plan.attention == "explicit_qk_triton_preprocess_native_softmax_pv"
                and can_use_triton_attention_preprocess(
                    scores,
                    valid_token_mask,
                    self.head_dim,
                )
            )
            if use_triton_preprocess:
                softmax_input = triton_scale_mask_to_fp32(
                    scores,
                    valid_token_mask,
                    head_dim=self.head_dim,
                    scale=self.scale,
                    causal=causal,
                )
            else:
                # A specialized route can be planned before QKV projection but
                # rejected later by its exact tensor guard. Build the ordinary
                # mask only on that fallback path so correctness never depends
                # on a looser, duplicated eligibility prediction.
                if score_mask is None:
                    score_mask = self._fallback_score_mask(
                        query,
                        valid_token_mask,
                        causal,
                    )
                scores.mul_(self.scale)

                if score_mask is not None:
                    scores.masked_fill_(score_mask, float("-inf"))
                softmax_input = scores

            use_native_dtype_softmax = (
                plan.attention != "explicit_reference_order"
                and scores.dtype in (torch.float16, torch.bfloat16)
                and sequence_length <= 512
            )
            if use_native_dtype_softmax:
                probabilities_fp32 = torch.softmax(
                    softmax_input,
                    dim=-1,
                    dtype=torch.float32,
                )
            else:
                probabilities_fp32 = torch.softmax(
                    softmax_input.float(),
                    dim=-1,
                )

            probabilities = probabilities_fp32.to(dtype=query.dtype)
            if observation is not None:
                if use_triton_preprocess:
                    actual_attention = "explicit_qk_triton_preprocess_native_softmax_pv"
                elif use_native_dtype_softmax:
                    actual_attention = "explicit_qk_native_fp32_dtype_softmax_pv"
                else:
                    actual_attention = "explicit_reference_order"
                observation.attention_backends.append(actual_attention)
            return torch.matmul(probabilities, projected_value)

    @staticmethod
    def _fallback_score_mask(
        query: torch.Tensor,
        valid_token_mask: torch.Tensor | None,
        causal: bool,
    ) -> torch.Tensor | None:
        """Materialize the reference mask only after a direct route rejects."""

        score_mask = None
        if valid_token_mask is not None:
            score_mask = (~valid_token_mask)[:, None, None, :]
        if causal:
            sequence_length = query.shape[-2]
            causal_mask = torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=query.device,
            ).triu(diagonal=1)
            causal_mask = causal_mask[None, None, :, :]
            score_mask = causal_mask if score_mask is None else score_mask | causal_mask
        return score_mask

    def forward(
        self,
        value: torch.Tensor,
        valid_token_mask: torch.Tensor | None = None,
        score_mask: torch.Tensor | None = None,
        causal: bool = False,
        plan: LayerExecutionPlan | None = None,
        observation: _ExecutionObservation | None = None,
    ) -> torch.Tensor:
        if plan is None:
            raise RuntimeError("attention execution requires a resolved layer plan")
        batch_size, sequence_length, _ = value.shape
        query, key, projected_value = self._project_heads(value, plan, observation)

        use_fp32_sdpa = plan.attention == "fp32_sdpa" and self._can_use_fp32_sdpa(
            value,
            sequence_length,
            causal,
        )
        if use_fp32_sdpa:
            if observation is not None:
                observation.attention_backends.append("fp32_sdpa")
            attention_mask = (
                None if valid_token_mask is None else valid_token_mask[:, None, None, :]
            )
            context = F.scaled_dot_product_attention(
                query,
                key,
                projected_value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            context = self._explicit_attention(
                query,
                key,
                projected_value,
                valid_token_mask,
                score_mask,
                causal,
                plan,
                observation,
            )

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.d_model)
        )
        output = self.out_proj(context)
        return output


class _TransformerBlock(nn.Module):
    """Reference-compatible pre-normalized Transformer block."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = _SelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def _ffn_hidden(
        self,
        normalized: torch.Tensor,
        plan: LayerExecutionPlan,
        observation: _ExecutionObservation | None,
    ) -> torch.Tensor:
        """Run the first FFN projection through the selected bounded candidate."""

        if plan.ffn == "torch_inplace_exact_gelu" and can_use_wide_exact_gelu(
            normalized,
            self.ffn_in.weight,
            self.ffn_in.bias,
        ):
            if observation is not None:
                observation.ffn_backends.append("torch_inplace_exact_gelu")
            return wide_linear_exact_gelu(
                normalized,
                self.ffn_in.weight,
                self.ffn_in.bias,
            )
        if observation is not None:
            observation.ffn_backends.append("torch_exact_gelu")
        return F.gelu(self.ffn_in(normalized), approximate="none")

    def _ffn_update(
        self,
        value: torch.Tensor,
        plan: LayerExecutionPlan,
        observation: _ExecutionObservation | None,
    ) -> torch.Tensor:
        """Run the exact FFN through the selected bounded implementation."""

        normalized = self.norm2(value)
        return self.ffn_out(self._ffn_hidden(normalized, plan, observation))

    def _packed_token_ffn(
        self,
        value: torch.Tensor,
        valid_token_mask: torch.Tensor,
        plan: LayerExecutionPlan,
        observation: _ExecutionObservation | None,
    ) -> torch.Tensor:
        """Run token-wise normalization and FFN only for valid token rows."""

        flat_value = value.reshape(-1, value.shape[-1])
        valid_indices = torch.nonzero(
            valid_token_mask.reshape(-1),
            as_tuple=False,
        ).flatten()
        packed_value = flat_value.index_select(0, valid_indices)
        if observation is not None:
            observation.ffn_backends.append("packed_valid_token_ffn")
        packed_update = self.ffn_out(
            F.gelu(self.ffn_in(self.norm2(packed_value)), approximate="none")
        )
        if plan.use_triton_residual:
            flat_update = torch.zeros_like(flat_value)
            flat_update.index_copy_(0, valid_indices, packed_update)
            update = flat_update.view_as(value)
            if can_use_triton_residual(value, update, valid_token_mask):
                if observation is not None:
                    observation.residual_backends.append("triton_residual_add_padding")
                return triton_residual_add_padding(
                    value,
                    update,
                    valid_token_mask,
                )

        packed_value = packed_value + packed_update
        if observation is not None:
            observation.residual_backends.append("packed_index_scatter_residual")
        output = torch.zeros_like(flat_value)
        output.index_copy_(0, valid_indices, packed_value)
        return output.view_as(value)

    def forward(
        self,
        value: torch.Tensor,
        valid_token_mask: torch.Tensor | None,
        score_mask: torch.Tensor | None,
        invalid_query_mask: torch.Tensor | None,
        causal: bool,
        plan: LayerExecutionPlan,
        observation: _ExecutionObservation | None = None,
    ) -> torch.Tensor:
        value = value + self.attention(
            self.norm1(value),
            valid_token_mask,
            score_mask,
            causal,
            plan,
            observation,
        )
        if plan.use_packed_ffn and valid_token_mask is not None:
            return self._packed_token_ffn(
                value,
                valid_token_mask,
                plan,
                observation,
            )

        ffn_update = self._ffn_update(value, plan, observation)
        if (
            plan.use_triton_residual
            and valid_token_mask is not None
            and can_use_triton_residual(value, ffn_update, valid_token_mask)
        ):
            if observation is not None:
                observation.residual_backends.append("triton_residual_add_padding")
            return triton_residual_add_padding(
                value,
                ffn_update,
                valid_token_mask,
            )

        value = value + ffn_update
        if observation is not None:
            observation.residual_backends.append("torch_residual_add")
        if invalid_query_mask is not None:
            value.masked_fill_(invalid_query_mask, 0)
        return value


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
        self._runtime_plan_signature: tuple[object, ...] | None = None
        self._runtime_plan: ExecutionPlan | None = None
        self._execution_observation_enabled = False
        self._last_execution_observation: dict[str, Any] | None = None

        causal_mask = None
        if config.causal:
            causal_mask = torch.ones(
                config.seq_len,
                config.seq_len,
                dtype=torch.bool,
            ).triu(diagonal=1)
        self.register_buffer("_causal_mask", causal_mask, persistent=False)

        self._dispatcher: OfflineDispatcher | None = None
        self._dispatch_signature: tuple[object, ...] | None = None
        self.dispatch_policy: str | None = None
        self.dispatch_route_origin: str | None = None
        self.dispatch_route_source: str | None = None
        self.dispatch_route_sha256: str | None = None
        self._enable_dispatch()

    def _enable_dispatch(self) -> None:
        """Restore deterministic offline dispatch as the default runtime mode."""

        self._dispatcher = OfflineDispatcher()
        self._dispatch_signature = None
        self._select_named_policy("auto", requested_policy="dispatch")
        self.dispatch_policy = "auto"
        self.dispatch_route_origin = "fallback"
        self.dispatch_route_source = self._dispatcher.source
        self.dispatch_route_sha256 = self._dispatcher.table_sha256
        if next(self.parameters()).is_cuda:
            self._resolve_dispatch()

    def _apply(self, function: Any, recurse: bool = True) -> UserOptimizedTransformer:
        """Invalidate captured parameter addresses after a module transform."""

        result = super()._apply(function, recurse=recurse)
        self._cuda_graph_replay = None
        self._runtime_plan_signature = None
        self._runtime_plan = None
        self._dispatch_signature = None
        if self._dispatcher is not None and next(self.parameters()).is_cuda:
            self._resolve_dispatch()
        return result

    def _resolve_dispatch(self, value: torch.Tensor | None = None) -> None:
        """Resolve an offline route once per static input signature."""

        if self._dispatcher is None:
            return
        if value is None:
            parameter = next(self.parameters())
            device = parameter.device
            dtype = parameter.dtype
            shape = (
                self.config.batch_size,
                self.config.seq_len,
                self.config.d_model,
            )
        else:
            device = value.device
            dtype = value.dtype
            shape = tuple(value.shape)
        signature = (device.type, device.index, dtype, shape)
        if signature == self._dispatch_signature:
            return
        resolution = self._dispatcher.resolve_result(
            self.config,
            device=device,
            dtype=dtype,
            shape=shape,
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
        """Select one registered policy and invalidate its previous plan."""

        spec = get_policy_spec(policy)
        self._cuda_graph_replay = None
        self._runtime_plan_signature = None
        self._runtime_plan = None
        self._last_execution_observation = None
        self._active_policy_id = spec.policy_id
        self.requested_policy = requested_policy or spec.policy_id

    def configure_runtime_policy(
        self,
        *,
        policy: str,
    ) -> None:
        """Select a registered policy or restore deterministic dispatch."""

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
        """Collect immutable facts without changing the selected policy."""

        parameter = next(self.parameters())
        if value is None:
            device = parameter.device
            dtype = parameter.dtype
            shape = (
                self.config.batch_size,
                self.config.seq_len,
                self.config.d_model,
            )
            input_contiguous = True
            has_valid_token_mask = True
            mask_compatible = True
            # Execution-path reporting describes the inference benchmark route.
            grad_enabled = False
        else:
            device = value.device
            dtype = value.dtype
            shape = tuple(value.shape)
            input_contiguous = value.is_contiguous()
            has_valid_token_mask = valid_token_mask is not None
            mask_compatible = valid_token_mask is None or (
                valid_token_mask.is_cuda == value.is_cuda
                and valid_token_mask.device == value.device
                and valid_token_mask.dtype == torch.bool
                and valid_token_mask.is_contiguous()
                and tuple(valid_token_mask.shape) == tuple(value.shape[:2])
            )
            grad_enabled = torch.is_grad_enabled()

        batch_size, sequence_length, d_model = shape
        return ExecutionContext(
            batch_size=batch_size,
            sequence_length=sequence_length,
            d_model=d_model,
            num_heads=self.config.num_heads,
            ffn_dim=self.config.ffn_dim,
            num_layers=self.config.num_layers,
            causal=self._causal_mask is not None,
            device=device,
            dtype=dtype,
            training=self.training,
            grad_enabled=grad_enabled,
            input_contiguous=input_contiguous,
            has_valid_token_mask=has_valid_token_mask,
            mask_compatible=mask_compatible,
        )

    def _execution_plan(
        self,
        value: torch.Tensor | None = None,
        valid_token_mask: torch.Tensor | None = None,
    ) -> ExecutionPlan:
        """Resolve the single plan used for reporting, masks and wrappers."""

        return resolve_execution_plan(
            get_policy_spec(self._active_policy_id),
            self._execution_context(value, valid_token_mask),
            requested_policy=self.requested_policy,
            dispatch_policy=self.dispatch_policy,
        )

    def _cached_execution_plan(
        self,
        value: torch.Tensor,
        valid_token_mask: torch.Tensor | None,
    ) -> ExecutionPlan:
        """Reuse a plan while every fact that affects eligibility is static."""

        mask_signature = None
        if valid_token_mask is not None:
            mask_signature = (
                valid_token_mask.device,
                valid_token_mask.dtype,
                tuple(valid_token_mask.shape),
                valid_token_mask.is_contiguous(),
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
        )
        if signature != self._runtime_plan_signature:
            self._runtime_plan = self._execution_plan(value, valid_token_mask)
            self._runtime_plan_signature = signature
        assert self._runtime_plan is not None
        return self._runtime_plan

    def describe_execution_path(self) -> dict[str, Any]:
        """Describe the last executed plan, or a pure preview before first use."""

        plan = (
            self._runtime_plan
            if self._runtime_plan is not None
            else self._execution_plan()
        )
        description = plan.describe(
            dispatch_source=self.dispatch_route_source,
            dispatch_table_sha256=self.dispatch_route_sha256,
            dispatch_policy=self.dispatch_policy,
            route_origin=self.dispatch_route_origin,
            causal=self._causal_mask is not None,
        )
        if self._last_execution_observation is not None:
            description["observed_execution"] = {
                key: list(value) if isinstance(value, list) else value
                for key, value in self._last_execution_observation.items()
            }
        return description

    def set_execution_observation(self, enabled: bool) -> None:
        """Enable branch observation for an eager correctness forward only."""

        self._execution_observation_enabled = bool(enabled)
        if enabled:
            self._last_execution_observation = None

    def _forward_eager(
        self,
        x: torch.Tensor,
        valid_token_mask: torch.Tensor | None = None,
        plan: ExecutionPlan | None = None,
    ) -> torch.Tensor:
        if plan is None:
            plan = self._cached_execution_plan(x, valid_token_mask)
        observation = (
            _ExecutionObservation()
            if self._execution_observation_enabled and not torch.compiler.is_compiling()
            else None
        )
        invalid_key_mask = None
        invalid_query_mask = None
        if valid_token_mask is not None:
            invalid_token_mask = ~valid_token_mask
            invalid_key_mask = invalid_token_mask[:, None, None, :]
            invalid_query_mask = invalid_token_mask[..., None]

        causal = self._causal_mask is not None
        score_mask = None if plan.direct_score_masking else invalid_key_mask
        if self._causal_mask is not None and not plan.direct_score_masking:
            sequence_length = x.shape[1]
            causal_score_mask = self._causal_mask[
                None,
                None,
                :sequence_length,
                :sequence_length,
            ]
            score_mask = (
                causal_score_mask
                if score_mask is None
                else score_mask | causal_score_mask
            )

        for layer, layer_plan in zip(self.layers, plan.layers, strict=True):
            x = layer(
                x,
                valid_token_mask,
                score_mask,
                invalid_query_mask,
                causal,
                layer_plan,
                observation,
            )
        x = self.final_norm(x)
        if invalid_query_mask is not None:
            x.masked_fill_(invalid_query_mask, 0)
        if observation is not None:
            self._last_execution_observation = observation.describe(len(self.layers))
        return x

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if torch.compiler.is_compiling():
            plan = self._execution_plan(x, valid_token_mask)
            return self._forward_eager(x, valid_token_mask, plan)
        self._resolve_dispatch(x)
        plan = self._cached_execution_plan(x, valid_token_mask)
        if self._execution_observation_enabled:
            # Correctness observation executes the eager body once; graph capture
            # and replay remain untouched and are used again after observation ends.
            return self._forward_eager(x, valid_token_mask, plan)
        if plan.use_cuda_graph:
            if self._cuda_graph_replay is None:
                self._cuda_graph_replay = CudaGraphReplay()
            return self._cuda_graph_replay.run(
                self._forward_eager,
                x,
                valid_token_mask,
            )
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
    """Copy official weights and derive packed QKV tensors exactly once."""

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
