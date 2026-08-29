from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from solution.operators.attention.triton_s1024_dh32 import (
    can_use_triton_shape13_causal_attention,
    triton_shape13_causal_attention,
    triton_shape13_causal_attention_available,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_shape13_triton_attention_availability_is_explicit() -> None:
    assert isinstance(triton_shape13_causal_attention_available(), bool)


def test_shape13_triton_attention_rejects_cpu_without_fallback() -> None:
    query = torch.randn(2, 4, 8, 32, dtype=torch.float16)

    assert not can_use_triton_shape13_causal_attention(query, query, query)
    with pytest.raises(RuntimeError, match="ineligible"):
        triton_shape13_causal_attention(query, query, query)


def test_shape13_triton_attention_rejects_masks_and_training() -> None:
    query = torch.empty(64, 4, 1024, 32, device="meta", dtype=torch.float16)
    mask = torch.ones(64, 1024, device="meta", dtype=torch.bool)

    assert not can_use_triton_shape13_causal_attention(
        query,
        query,
        query,
        mask,
    )
    assert not can_use_triton_shape13_causal_attention(
        query,
        query,
        query,
        training=True,
    )


def test_shape13_triton_attention_has_a_traceable_fake_implementation() -> None:
    if not triton_shape13_causal_attention_available():
        pytest.skip("optional Triton runtime is unavailable")
    script = """
import torch
from solution.operators.attention.triton_s1024_dh32 import (
    TRITON_SHAPE13_CAUSAL_ATTENTION_BACKEND,
    prevalidated_triton_shape13_causal_attention,
    triton_shape13_causal_attention_available,
)

assert triton_shape13_causal_attention_available()
query = torch.empty(64, 4, 1024, 32, device="meta", dtype=torch.float16)
output, backend = prevalidated_triton_shape13_causal_attention(query, query, query)
assert output.device.type == "meta"
assert output.shape == query.shape
assert output.dtype == query.dtype
assert output.is_contiguous()
assert backend == TRITON_SHAPE13_CAUSAL_ATTENTION_BACKEND
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
