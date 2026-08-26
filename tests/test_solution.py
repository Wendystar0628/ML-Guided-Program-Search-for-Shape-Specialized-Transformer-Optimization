"""Correctness tests for the current optimization target."""

from __future__ import annotations

import inspect
import os
import py_compile
from pathlib import Path

import pytest
import torch

from official import torch_transformer_benchmark as official
from runner.execution import load_solution_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_official_entrypoint_uses_current_solution() -> None:
    import torch_transformer_benchmark as entrypoint

    assert (
        entrypoint.official.UserOptimizedTransformer
        is entrypoint.solution.UserOptimizedTransformer
    )


def test_solution_loader_ignores_stale_helper_bytecode(tmp_path: Path) -> None:
    solution_root = tmp_path / "solution"
    solution_root.mkdir()
    helper = solution_root / "helper.py"
    helper.write_text("marker = 111\n", encoding="utf-8")
    fixed_time = 1_700_000_000
    os.utime(helper, (fixed_time, fixed_time))
    py_compile.compile(str(helper), doraise=True)
    helper.write_text("marker = 222\n", encoding="utf-8")
    os.utime(helper, (fixed_time, fixed_time))
    (solution_root / "transformer.py").write_text(
        "from torch import nn\n"
        "from .helper import marker\n"
        "class UserOptimizedTransformer(nn.Module):\n"
        "    pass\n",
        encoding="utf-8",
    )

    module = load_solution_module(tmp_path)

    assert module.marker == 222


def _copy_official_weights(
    solution_module: object,
    baseline: torch.nn.Module,
    solution: torch.nn.Module,
) -> None:
    weight_loader = getattr(solution_module, "copy_model_weights", None)
    if weight_loader is None:
        official.copy_model_weights(baseline, solution, strict=True)
    else:
        weight_loader(baseline, solution, strict=True)


@pytest.mark.parametrize(
    ("causal", "use_padding"),
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
    ids=(
        "noncausal-no-padding",
        "noncausal-padding",
        "causal-no-padding",
        "causal-padding",
    ),
)
def test_solution_matches_baseline_for_all_mask_modes(
    causal: bool,
    use_padding: bool,
) -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution_class = solution_module.UserOptimizedTransformer
    signature = inspect.signature(solution_class.forward)
    assert list(signature.parameters) == ["self", "x", "valid_token_mask"]
    assert signature.parameters["valid_token_mask"].default is None

    config = official.TransformerConfig(
        batch_size=2,
        seq_len=4,
        d_model=8,
        num_heads=2,
        ffn_dim=16,
        num_layers=1,
        causal=causal,
    )
    torch.manual_seed(2026)
    baseline = official.BaselineTransformer(config).eval()
    solution = solution_class(config).eval()
    _copy_official_weights(solution_module, baseline, solution)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(99)
    inputs = torch.randn(2, 4, 8, generator=generator)
    valid_mask = None
    if use_padding:
        valid_mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False]],
            dtype=torch.bool,
        )

    with torch.inference_mode():
        if valid_mask is None:
            reference = baseline(inputs)
            solution_output = solution(inputs)
        else:
            reference = baseline(inputs, valid_mask)
            solution_output = solution(inputs, valid_mask)

    comparison = official.compare_outputs(
        reference,
        solution_output,
        rtol=0.01,
        atol=0.001,
    )
    assert comparison.passed
    assert solution_output.shape == inputs.shape
    assert solution_output.dtype == inputs.dtype
    assert solution_output.device == inputs.device
