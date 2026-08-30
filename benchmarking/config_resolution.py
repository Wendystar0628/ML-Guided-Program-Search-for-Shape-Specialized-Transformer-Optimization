"""Resolve the deployed or explicit program for one benchmark shape."""

from __future__ import annotations

from pathlib import Path

import torch

from deployment.environment import ImplementationScope
from deployment.registry import (
    EnvironmentFingerprint,
    ShapeFingerprint,
    resolve_deployed_config,
)
from solution.config import ConfigSpec, portable_config, portable_streamed_config

from .protocols import RunVariant, TransformerShape, load_json


def _streamed_fallback_config(device: str) -> ConfigSpec:
    """Prefer the bounded Shape 14 kernel when the local CUDA stack supports it."""

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and torch.cuda.is_available():
        from solution.shape14.defaults import conservative_streamed_config
        from solution.shape14.triton_streaming_dh64 import (
            triton_streaming_dh64_causal_attention_available,
        )

        if (
            triton_streaming_dh64_causal_attention_available()
            and torch.cuda.get_device_capability(resolved_device) >= (8, 0)
        ):
            return conservative_streamed_config()
    return portable_streamed_config()


def shape_fingerprint(
    shape: TransformerShape,
    variant: RunVariant,
) -> ShapeFingerprint:
    return ShapeFingerprint(
        batch_size=shape.batch_size,
        qkv_dim=shape.d_model,
        heads=shape.num_heads,
        seq_len=shape.seq_len,
        layers=shape.num_layers,
        causal=shape.causal,
        ffn_dim=shape.ffn_dim,
        dtype=variant.dtype,
        padding_ratio=variant.padding_ratio,
        input_scale=variant.input_scale,
    )


def resolve_config(
    path: Path | None,
    shape: TransformerShape,
    variant: RunVariant,
    device: str,
    *,
    project_root: Path,
) -> ConfigSpec:
    if path is not None:
        return ConfigSpec.from_dict(load_json(path))
    hardware = EnvironmentFingerprint.detect(
        torch.device(device),
        project_root=project_root,
        scope=(
            ImplementationScope.SHAPE14
            if shape.streamed
            else ImplementationScope.RESIDENT
        ),
    )
    deployed = resolve_deployed_config(
        hardware=hardware,
        shape=shape_fingerprint(shape, variant),
    )
    if deployed is not None:
        return deployed
    return _streamed_fallback_config(device) if shape.streamed else portable_config()


__all__ = ["resolve_config", "shape_fingerprint"]
