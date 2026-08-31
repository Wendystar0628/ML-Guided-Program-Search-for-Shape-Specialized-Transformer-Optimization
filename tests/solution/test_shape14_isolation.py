from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from solution.shape14.executor import execute_streamed
from solution.transformer import _is_official_shape14

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHAPE14_KERNEL_MODULE = "solution.shape14.triton_streaming_dh64"


def test_only_the_exact_official_shape14_uses_streamed_deployment_scope() -> None:
    config = SimpleNamespace(
        num_heads=16,
        num_layers=2,
        ffn_dim=1024,
        causal=True,
    )

    assert _is_official_shape14(config, (32, 100000, 1024))
    assert not _is_official_shape14(config, (1, 100000, 1024))


def test_resident_solution_import_does_not_load_shape14_kernel() -> None:
    script = f"""
import sys
import torch
import solution
import solution.kernels.attention as attention
import solution.transformer
import autotune.search_engine
from solution.plan_builder import HardwareCapabilities

HardwareCapabilities.detect(torch.device("cpu"))
assert {SHAPE14_KERNEL_MODULE!r} not in sys.modules
assert "autotune.shape14_search_space" not in sys.modules
assert not hasattr(attention, "triton_streaming_dh64_causal_attention_bsd")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_streamed_executor_preserves_order_and_mask_slices() -> None:
    value = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
    valid_token_mask = torch.tensor(
        [
            [True, True, True],
            [True, False, False],
            [True, True, False],
            [False, False, False],
        ]
    )
    observed_masks: list[torch.Tensor] = []

    def forward_chunk(
        chunk: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        assert mask is not None
        observed_masks.append(mask.clone())
        return chunk + 1

    output = execute_streamed(
        value,
        valid_token_mask,
        microbatch_size=2,
        forward_chunk=forward_chunk,
    )

    torch.testing.assert_close(output, value + 1)
    torch.testing.assert_close(
        torch.cat(observed_masks),
        valid_token_mask,
    )


@pytest.mark.parametrize("microbatch_size", [0, True, 3])
def test_streamed_executor_rejects_invalid_microbatches(
    microbatch_size: int,
) -> None:
    value = torch.zeros(4, 2, 1)

    with pytest.raises(ValueError):
        execute_streamed(
            value,
            None,
            microbatch_size=microbatch_size,
            forward_chunk=lambda chunk, mask: chunk,
        )
