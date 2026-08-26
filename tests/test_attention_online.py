from __future__ import annotations

import pytest
import torch

from solution.kernels.attention_online import (
    can_use_triton_online_attention,
    triton_online_attention,
)


def test_online_attention_rejects_cpu_tensors() -> None:
    query = torch.empty((1, 8, 16, 64), dtype=torch.float16)
    key = torch.empty_like(query)
    value = torch.empty_like(query)

    with torch.inference_mode():
        assert not can_use_triton_online_attention(query, key, value, None)


def test_online_attention_keeps_caller_controlled_fallback() -> None:
    query = torch.empty((1, 8, 16, 64), dtype=torch.float16)
    key = torch.empty_like(query)
    value = torch.empty_like(query)

    with torch.inference_mode(), pytest.raises(
        ValueError,
        match="not supported",
    ):
        triton_online_attention(
            query,
            key,
            value,
            None,
            scale=0.125,
            causal=False,
        )
