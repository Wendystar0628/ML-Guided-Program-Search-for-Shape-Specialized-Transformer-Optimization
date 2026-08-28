"""General causal attention primitives for the official workload family."""

from __future__ import annotations

import torch
import torch.nn.functional as F

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_REFERENCE_QUERY_BLOCK = 128


def can_use_causal_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_token_mask: torch.Tensor | None = None,
    *,
    causal: bool = True,
) -> bool:
    """Return whether native SDPA is comparator-safe for this request.

    BF16 intentionally uses the explicit fallback. Native SDPA changes more
    low-precision rounding boundaries than the official implementation and can
    exceed the elementwise tolerance even though the mathematical operation is
    equivalent.
    """

    return bool(
        query.dtype != torch.bfloat16
        and _attention_inputs_are_legal(
            query,
            key,
            value,
            valid_token_mask,
            causal=causal,
        )
    )


def _attention_inputs_are_legal(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_token_mask: torch.Tensor | None,
    *,
    causal: bool,
) -> bool:
    """Validate shared shape, dtype, device, and mask invariants."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False
    if query.device != key.device or query.device != value.device:
        return False
    if query.dtype not in _SUPPORTED_DTYPES:
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if query.shape[:-1] != key.shape[:-1]:
        return False
    if query.shape[:3] != value.shape[:3]:
        return False
    if query.shape[-1] != key.shape[-1]:
        return False
    if query.shape[-2] <= 0 or key.shape[-2] <= 0:
        return False
    if causal and query.shape[-2] != key.shape[-2]:
        return False
    return valid_token_mask is None or bool(
        valid_token_mask.device == query.device
        and valid_token_mask.dtype == torch.bool
        and valid_token_mask.ndim == 2
        and valid_token_mask.shape == (query.shape[0], key.shape[-2])
    )


def causal_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_token_mask: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Run native SDPA without constructing a persistent square mask.

    ``valid_token_mask`` follows the official ``[B, S]`` convention where
    ``True`` marks a valid token. PyTorch combines the broadcast key mask with
    its implicit causal bias inside SDPA, so an all-true mask does not require
    a Python-side ``S x S`` causal tensor.
    """

    if not can_use_causal_sdpa(
        query,
        key,
        value,
        valid_token_mask,
        causal=causal,
    ):
        raise ValueError("attention tensors are incompatible with causal SDPA")

    attention_mask = None
    if valid_token_mask is not None:
        attention_mask = valid_token_mask[:, None, None, :]
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=causal,
        scale=scale,
    )


def reference_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_token_mask: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Run the official operation order without a full square causal mask.

    Query blocking limits the temporary score and mask extent to
    ``[B, H, 128, S]`` and ``[128, S]`` while preserving the reference's QK,
    scaling, FP32 softmax, probability cast, and PV boundaries.
    """

    if not _attention_inputs_are_legal(
        query,
        key,
        value,
        valid_token_mask,
        causal=causal,
    ):
        raise ValueError("attention tensors are incompatible with the safe path")

    effective_scale = query.shape[-1] ** -0.5 if scale is None else scale
    sequence_length = query.shape[-2]
    key_positions = torch.arange(sequence_length, device=query.device)
    key_transposed = key.transpose(-2, -1)
    invalid_keys = (
        None if valid_token_mask is None else (~valid_token_mask)[:, None, None, :]
    )
    chunks: list[torch.Tensor] = []
    for start in range(0, sequence_length, _REFERENCE_QUERY_BLOCK):
        stop = min(start + _REFERENCE_QUERY_BLOCK, sequence_length)
        scores = torch.matmul(query[..., start:stop, :], key_transposed)
        scores.mul_(effective_scale)
        if causal:
            query_positions = torch.arange(start, stop, device=query.device)
            causal_block = key_positions[None, :] > query_positions[:, None]
            scores.masked_fill_(causal_block, float("-inf"))
        if invalid_keys is not None:
            scores.masked_fill_(invalid_keys, float("-inf"))
        probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        chunks.append(torch.matmul(probabilities, value))
    return torch.cat(chunks, dim=-2)


def causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    valid_token_mask: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Use native SDPA when safe and preserve exact reference fallback."""

    if can_use_causal_sdpa(
        query,
        key,
        value,
        valid_token_mask,
        causal=causal,
    ):
        return causal_sdpa(
            query,
            key,
            value,
            valid_token_mask,
            scale=scale,
            causal=causal,
        )
    return reference_causal_attention(
        query,
        key,
        value,
        valid_token_mask,
        scale=scale,
        causal=causal,
    )


__all__ = [
    "can_use_causal_sdpa",
    "causal_attention",
    "causal_sdpa",
    "reference_causal_attention",
]
