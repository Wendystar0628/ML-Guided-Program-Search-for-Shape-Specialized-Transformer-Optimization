from __future__ import annotations

import pytest
import torch
from torch import nn

from solution.kernels import triton_mixed_residual_norm as mixed_norm_module
from solution.kernels.triton_mixed_residual_norm import (
    can_use_triton_mixed_residual_layer_norm,
    triton_mixed_residual_layer_norm,
)


def test_mixed_residual_norm_rejects_cpu_inputs() -> None:
    value = torch.randn(2, 4, 128)
    update = torch.randn(2, 4, 128, dtype=torch.float16)
    layer_norm = nn.LayerNorm(128)

    assert not can_use_triton_mixed_residual_layer_norm(
        value,
        update,
        layer_norm,
    )
    with pytest.raises(RuntimeError, match="ineligible"):
        triton_mixed_residual_layer_norm(
            value,
            update,
            layer_norm,
            final_boundary=False,
        )


def test_mixed_residual_norm_tile_guard_is_exact() -> None:
    exact = torch.empty(128, 128, 128, device="meta")
    wrong_batch = torch.empty(127, 128, 128, device="meta")
    wrong_width = torch.empty(128, 128, 64, device="meta")

    assert mixed_norm_module._is_shape06_graph_tile(exact)
    assert not mixed_norm_module._is_shape06_graph_tile(wrong_batch)
    assert not mixed_norm_module._is_shape06_graph_tile(wrong_width)
