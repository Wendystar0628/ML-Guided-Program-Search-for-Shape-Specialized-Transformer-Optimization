from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from solution.kernels import (
    can_split_qkv,
    can_use_causal_sdpa,
    can_use_linear_exact_gelu,
    can_use_residual_add,
    causal_sdpa,
    linear_exact_gelu,
    residual_add,
    split_qkv,
)


def test_split_qkv_is_a_shape_independent_view() -> None:
    packed = torch.arange(2 * 5 * 3 * 8, dtype=torch.float32).view(2, 5, 24)

    query, key, value = split_qkv(packed, num_heads=2)

    assert query.shape == key.shape == value.shape == (2, 2, 5, 4)
    assert query.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    assert can_split_qkv(packed, num_heads=2)
    assert not can_split_qkv(packed, num_heads=3)


def test_causal_sdpa_matches_an_explicit_float32_reference() -> None:
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(2, 4, 7, 8, generator=generator)
    key = torch.randn(2, 4, 7, 8, generator=generator)
    value = torch.randn(2, 4, 7, 8, generator=generator)

    actual = causal_sdpa(query, key, value)
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    causal_mask = torch.triu(torch.ones(7, 7, dtype=torch.bool), diagonal=1)
    scores.masked_fill_(causal_mask, float("-inf"))
    expected = torch.matmul(torch.softmax(scores, dim=-1), value)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_causal_sdpa_validates_shape_dtype_and_optional_token_mask() -> None:
    query = torch.randn(1, 2, 4, 8)
    mask = torch.tensor([[True, True, True, False]])

    assert can_use_causal_sdpa(query, query, query, mask)
    assert not can_use_causal_sdpa(query, query, query, mask[:, :3])
    with pytest.raises(ValueError, match="incompatible"):
        causal_sdpa(query, query, query, mask[:, :3])


def test_linear_exact_gelu_preserves_exact_gelu_semantics() -> None:
    value = torch.randn(3, 5)
    weight = torch.randn(7, 5)
    bias = torch.randn(7)
    original = value.clone()

    with torch.inference_mode():
        actual, backend = linear_exact_gelu(value, weight, bias)
        assert can_use_linear_exact_gelu(value, weight, bias)
    expected = F.gelu(F.linear(value, weight, bias), approximate="none")

    assert backend == "inplace_exact_gelu"
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(value, original)


def test_residual_add_masks_only_invalid_query_tokens() -> None:
    value = torch.ones(2, 3, 4)
    update = torch.full_like(value, 2.0)
    valid = torch.tensor([[True, False, True], [False, True, True]])

    actual = residual_add(value, update, valid)

    assert can_use_residual_add(value, update, valid)
    assert torch.all(actual[valid] == 3.0)
    assert torch.all(actual[~valid] == 0.0)
