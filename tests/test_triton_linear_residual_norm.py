from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from solution.kernels.boundary import (
    TRITON_LINEAR_RESIDUAL_NORM_BACKEND,
    can_use_triton_linear_residual_norm,
    triton_linear_residual_norm,
)


def test_linear_residual_norm_rejects_cpu() -> None:
    width = 32
    source = torch.randn(2, 3, width, dtype=torch.float16)
    weight = torch.randn(width, width, dtype=torch.float16)
    bias = torch.randn(width, dtype=torch.float16)
    value = torch.randn(2, 3, width)
    assert not can_use_triton_linear_residual_norm(
        source,
        weight,
        bias,
        value,
        nn.LayerNorm(width),
        block_rows=16,
        num_warps=2,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
@pytest.mark.parametrize(
    ("width", "layout", "final_boundary", "block_rows", "num_warps"),
    [
        (32, "bsd", False, 16, 2),
        (32, "bhsd", True, 32, 4),
        (128, "bsd", False, 16, 4),
        (128, "bhsd", True, 32, 8),
    ],
)
def test_linear_residual_norm_matches_materialized_boundary(
    width: int,
    layout: str,
    final_boundary: bool,
    block_rows: int,
    num_warps: int,
) -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    batch, sequence = 2, 17
    source_bsd = torch.randn(
        batch,
        sequence,
        width,
        device=device,
        dtype=torch.float16,
    )
    if layout == "bhsd":
        heads = 4
        source = (
            source_bsd.view(batch, sequence, heads, width // heads)
            .transpose(1, 2)
            .contiguous()
        )
    else:
        source = source_bsd
    weight = torch.randn(width, width, device=device, dtype=torch.float16)
    bias = torch.randn(width, device=device, dtype=torch.float16)
    value = torch.randn(batch, sequence, width, device=device)
    norm = nn.LayerNorm(width, device=device)

    residual, normalized, marker = triton_linear_residual_norm(
        source,
        weight,
        bias,
        value,
        norm,
        final_boundary=final_boundary,
        block_rows=block_rows,
        num_warps=num_warps,
    )

    update = F.linear(source_bsd, weight, bias).to(torch.float16)
    expected_residual = value + update.float()
    expected_normalized = F.layer_norm(
        expected_residual,
        (width,),
        norm.weight,
        norm.bias,
        norm.eps,
    ).to(torch.float32 if final_boundary else torch.float16)
    assert marker == TRITON_LINEAR_RESIDUAL_NORM_BACKEND
    assert torch.allclose(residual, expected_residual, atol=2e-3, rtol=2e-2)
    assert torch.allclose(normalized, expected_normalized, atol=2e-3, rtol=2e-2)
