from __future__ import annotations

import pytest
import torch
from torch import nn

from solution.operators.norm.triton_residual import (
    can_use_triton_residual_layer_norm,
    triton_residual_layer_norm,
)


def test_triton_residual_norm_rejects_cpu_inputs() -> None:
    value = torch.randn(4, 128)
    update = torch.randn_like(value)
    layer_norm = nn.LayerNorm(128)

    assert not can_use_triton_residual_layer_norm(value, update, layer_norm)
    with pytest.raises(RuntimeError, match="ineligible"):
        triton_residual_layer_norm(value, update, layer_norm)


def test_triton_residual_norm_rejects_unmeasured_width() -> None:
    value = torch.empty(1_000_000, 64, device="meta")
    update = torch.empty_like(value)
    layer_norm = nn.LayerNorm(64, device="meta")

    assert not can_use_triton_residual_layer_norm(value, update, layer_norm)
