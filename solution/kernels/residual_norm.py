"""Lazy compiled residual-plus-LayerNorm primitive for an explicit policy."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import torch
import torch.nn.functional as F
from torch import nn

_KernelResult: TypeAlias = tuple[torch.Tensor, torch.Tensor]
_CompiledKernel: TypeAlias = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        tuple[int, ...],
        torch.Tensor | None,
        torch.Tensor | None,
        float,
    ],
    _KernelResult,
]

_compiled_kernel: _CompiledKernel | None = None
_compile_factory_failed = False
_warmed_signatures: set[tuple[object, ...]] = set()
_failed_signatures: set[tuple[object, ...]] = set()


def _residual_layer_norm_impl(
    value: torch.Tensor,
    update: torch.Tensor,
    normalized_shape: tuple[int, ...],
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> _KernelResult:
    residual = value + update
    normalized = F.layer_norm(residual, normalized_shape, weight, bias, eps)
    return residual, normalized


def _signature(
    value: torch.Tensor,
    update: torch.Tensor,
    layer_norm: nn.LayerNorm,
) -> tuple[object, ...]:
    return (
        value.device,
        value.dtype,
        tuple(value.shape),
        tuple(value.stride()),
        tuple(update.stride()),
        tuple(layer_norm.normalized_shape),
        layer_norm.eps,
        layer_norm.weight is not None,
        layer_norm.bias is not None,
    )


def _is_outer_compilation_active() -> bool:
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    return bool(callable(is_compiling) and is_compiling())


def _compiled_backend_available() -> bool:
    return bool(
        not _compile_factory_failed and callable(getattr(torch, "compile", None))
    )


def can_use_residual_layer_norm(
    value: torch.Tensor,
    update: torch.Tensor,
    layer_norm: nn.LayerNorm,
) -> bool:
    """Return whether the lazy compiled inference path is eligible."""

    if not isinstance(layer_norm, nn.LayerNorm):
        return False
    if not _compiled_backend_available() or _is_outer_compilation_active():
        return False
    if torch.is_grad_enabled() or value.device.type != "cuda":
        return False
    if value.dtype != torch.float32 or update.dtype != value.dtype:
        return False
    if value.shape != update.shape or value.device != update.device:
        return False
    if value.ndim < 2 or not value.is_contiguous() or not update.is_contiguous():
        return False
    normalized_shape = tuple(layer_norm.normalized_shape)
    if len(normalized_shape) != 1 or normalized_shape[0] != value.shape[-1]:
        return False
    for parameter in (layer_norm.weight, layer_norm.bias):
        if parameter is not None and (
            parameter.device != value.device or parameter.dtype != value.dtype
        ):
            return False

    signature = _signature(value, update, layer_norm)
    if signature in _failed_signatures:
        return False
    # A compiler cannot safely specialize a new shape while an outer CUDA
    # graph is being captured. CudaGraphReplay performs eager warmup first, so
    # a warmed signature remains capturable on the subsequent pass.
    if torch.cuda.is_current_stream_capturing():
        return signature in _warmed_signatures
    return True


def _get_compiled_kernel() -> _CompiledKernel:
    global _compile_factory_failed, _compiled_kernel

    if _compiled_kernel is not None:
        return _compiled_kernel
    if _compile_factory_failed:
        raise RuntimeError("compiled residual LayerNorm backend is unavailable")
    compile_function = getattr(torch, "compile", None)
    if not callable(compile_function):
        _compile_factory_failed = True
        raise TypeError("torch.compile is unavailable")
    try:
        _compiled_kernel = compile_function(
            _residual_layer_norm_impl,
            fullgraph=True,
            dynamic=False,
            mode="default",
        )
    except Exception as exc:
        _compile_factory_failed = True
        raise RuntimeError("failed to create compiled residual LayerNorm") from exc
    return _compiled_kernel


def residual_layer_norm(
    value: torch.Tensor,
    update: torch.Tensor,
    layer_norm: nn.LayerNorm,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Run the requested compiled backend and report the observed backend.

    Compilation happens lazily during the runner's correctness or warmup
    phase. This function belongs only to the explicit compiled policy, so an
    unavailable compiler or failed specialization is surfaced to the runner
    instead of silently executing a different route.
    """

    if not can_use_residual_layer_norm(value, update, layer_norm):
        raise RuntimeError(
            "compiled residual LayerNorm is ineligible for the requested inputs"
        )
    signature = _signature(value, update, layer_norm)
    compiled = _get_compiled_kernel()
    try:
        residual, normalized = compiled(
            value,
            update,
            tuple(layer_norm.normalized_shape),
            layer_norm.weight,
            layer_norm.bias,
            layer_norm.eps,
        )
    except Exception as exc:
        _failed_signatures.add(signature)
        raise RuntimeError("compiled residual LayerNorm execution failed") from exc
    _warmed_signatures.add(signature)
    return residual, normalized, "compiled_residual_layer_norm"


__all__ = ["can_use_residual_layer_norm", "residual_layer_norm"]
