from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from solution.kernels.boundary import (
    TRITON_FFN_RESIDUAL_NORM_BACKEND,
    can_use_triton_ffn_residual_norm,
    triton_ffn_residual_norm,
)


def test_fused_ffn_residual_norm_rejects_cpu() -> None:
    width = 128
    source = torch.randn(2, 3, width, dtype=torch.float16)
    weight = torch.randn(width, width, dtype=torch.float16)
    bias = torch.randn(width, dtype=torch.float16)
    value = torch.randn(2, 3, width)
    assert not can_use_triton_ffn_residual_norm(
        source,
        weight,
        bias,
        weight,
        bias,
        value,
        nn.LayerNorm(width),
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
@pytest.mark.parametrize("final_boundary", (False, True))
def test_fused_ffn_residual_norm_matches_materialized_program(
    final_boundary: bool,
) -> None:
    torch.manual_seed(17)
    device = torch.device("cuda")
    batch, sequence, width = 2, 17, 128
    source = torch.randn(
        batch,
        sequence,
        width,
        device=device,
        dtype=torch.float16,
    )
    input_weight = torch.randn(
        width,
        width,
        device=device,
        dtype=torch.float16,
    )
    input_bias = torch.randn(width, device=device, dtype=torch.float16)
    output_weight = torch.randn_like(input_weight)
    output_bias = torch.randn_like(input_bias)
    value = torch.randn(batch, sequence, width, device=device)
    norm = nn.LayerNorm(width, device=device)

    residual, normalized, marker = triton_ffn_residual_norm(
        source,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
        value,
        norm,
        final_boundary=final_boundary,
        num_stages=1,
    )

    hidden = F.gelu(
        F.linear(source.float(), input_weight.float(), input_bias.float()),
        approximate="none",
    ).half()
    update = F.linear(
        hidden.float(),
        output_weight.float(),
        output_bias.float(),
    ).half()
    expected_residual = value + update.float()
    expected_normalized = F.layer_norm(
        expected_residual,
        (width,),
        norm.weight,
        norm.bias,
        norm.eps,
    ).to(torch.float32 if final_boundary else torch.float16)

    assert marker == TRITON_FFN_RESIDUAL_NORM_BACKEND
    assert torch.allclose(residual, expected_residual, atol=3e-3, rtol=2e-2)
    assert torch.allclose(normalized, expected_normalized, atol=3e-3, rtol=2e-2)
