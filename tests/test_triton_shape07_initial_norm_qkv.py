from __future__ import annotations

import pytest
import torch
from torch import nn

from solution.kernels.projection import (
    TRITON_INITIAL_NORM_QKV_NATIVE_BHSD_BACKEND,
    can_use_triton_initial_norm_qkv_native_bhsd,
    triton_initial_norm_qkv_native_bhsd,
)


def _cpu_inputs() -> tuple[torch.Tensor, nn.LayerNorm, torch.Tensor, torch.Tensor]:
    value = torch.zeros((64, 128, 32), dtype=torch.float32)
    layer_norm = nn.LayerNorm(32, dtype=torch.float32)
    qkv_weight = torch.zeros((96, 32), dtype=torch.float16)
    qkv_bias = torch.zeros(96, dtype=torch.float16)
    return value, layer_norm, qkv_weight, qkv_bias


def test_shape07_fusion_marker_is_primitive_specific() -> None:
    assert (
        TRITON_INITIAL_NORM_QKV_NATIVE_BHSD_BACKEND
        == "triton_initial_norm_qkv_native_bhsd"
    )


def test_shape07_fusion_rejects_cpu_without_fallback() -> None:
    inputs = _cpu_inputs()
    with torch.inference_mode():
        assert not can_use_triton_initial_norm_qkv_native_bhsd(*inputs)
        with pytest.raises(RuntimeError, match="ineligible"):
            triton_initial_norm_qkv_native_bhsd(*inputs)


def test_shape07_fusion_rejects_invalid_launch_before_eligibility() -> None:
    inputs = _cpu_inputs()
    with torch.inference_mode(), pytest.raises(ValueError, match="launch configuration"):
        triton_initial_norm_qkv_native_bhsd(*inputs, block_m=8)
