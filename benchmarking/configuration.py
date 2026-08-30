"""Resolve the deployed or explicit program for one benchmark shape."""

from __future__ import annotations

from pathlib import Path

import torch

from deployment.registry import (
    EnvironmentFingerprint,
    ShapeFingerprint,
    resolve_deployed_config,
)
from solution.config import ConfigSpec, portable_config, portable_streamed_config

from .protocols import RunVariant, TransformerShape, load_json


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
    )
    deployed = resolve_deployed_config(
        hardware=hardware,
        shape=shape_fingerprint(shape, variant),
    )
    if deployed is not None:
        return deployed
    return portable_streamed_config() if shape.streamed else portable_config()


__all__ = ["resolve_config", "shape_fingerprint"]
