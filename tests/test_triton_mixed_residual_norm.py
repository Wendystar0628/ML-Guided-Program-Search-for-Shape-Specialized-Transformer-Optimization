from __future__ import annotations

import pytest
import torch
from torch import nn

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
