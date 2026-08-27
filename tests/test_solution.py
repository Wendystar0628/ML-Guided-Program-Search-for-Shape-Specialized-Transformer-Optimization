"""Correctness tests for the current optimization target."""

from __future__ import annotations

import inspect
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from official import torch_transformer_benchmark as official
from runner.execution import load_solution_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_official_entrypoint_runs_current_solution_in_a_subprocess() -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "torch_transformer_benchmark.py"),
        "--batch-size",
        "1",
        "--seq-len",
        "4",
        "--d-model",
        "8",
        "--heads",
        "2",
        "--ffn-dim",
        "16",
        "--layers",
        "1",
        "--causal",
        "--padding-ratio",
        "0.5",
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--accuracy-trials",
        "1",
        "--warmup",
        "0",
        "--repeats",
        "1",
        "--benchmark-rounds",
        "1",
        "--matmul-precision",
        "highest",
        "--no-allow-tf32",
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "summary: PASS" in completed.stdout
    assert "speedup  :" in completed.stdout


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


def _config(*, causal: bool, num_layers: int = 2) -> official.TransformerConfig:
    return official.TransformerConfig(
        batch_size=2,
        seq_len=4,
        d_model=8,
        num_heads=2,
        ffn_dim=16,
        num_layers=num_layers,
        causal=causal,
    )


def _build_models(
    *, causal: bool, num_layers: int = 2
) -> tuple[object, torch.nn.Module, torch.nn.Module]:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution_class = solution_module.UserOptimizedTransformer
    signature = inspect.signature(solution_class.forward)
    assert list(signature.parameters) == ["self", "x", "valid_token_mask"]
    assert signature.parameters["valid_token_mask"].default is None

    config = _config(causal=causal, num_layers=num_layers)
    torch.manual_seed(2026)
    baseline = official.BaselineTransformer(config).eval()
    solution = solution_class(config).eval()
    solution_module.copy_model_weights(baseline, solution, strict=True)
    return solution_module, baseline, solution


def test_strict_weight_hook_packs_qkv_and_preserves_the_baseline() -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    config = _config(causal=False)
    torch.manual_seed(2026)
    baseline = official.BaselineTransformer(config).eval()
    before = {key: value.clone() for key, value in baseline.state_dict().items()}
    solution = solution_module.UserOptimizedTransformer(config).eval()

    solution_module.copy_model_weights(baseline, solution, strict=True)

    after = baseline.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key], after[key]) for key in before)
    solution_state = solution.state_dict()
    for layer_index in range(config.num_layers):
        source_prefix = f"layers.{layer_index}.attention"
        target_prefix = f"{source_prefix}.qkv_proj"
        expected_weight = torch.cat(
            [
                before[f"{source_prefix}.{projection}.weight"]
                for projection in ("q_proj", "k_proj", "v_proj")
            ],
            dim=0,
        )
        expected_bias = torch.cat(
            [
                before[f"{source_prefix}.{projection}.bias"]
                for projection in ("q_proj", "k_proj", "v_proj")
            ],
            dim=0,
        )
        assert torch.equal(solution_state[f"{target_prefix}.weight"], expected_weight)
        assert torch.equal(solution_state[f"{target_prefix}.bias"], expected_bias)


def test_strict_weight_hook_rejects_incompatible_models() -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    baseline = official.BaselineTransformer(_config(causal=False, num_layers=1)).eval()
    solution = solution_module.UserOptimizedTransformer(
        _config(causal=False, num_layers=2)
    ).eval()

    with pytest.raises(RuntimeError, match="strict weight mapping failed"):
        solution_module.copy_model_weights(baseline, solution, strict=True)


@pytest.mark.parametrize(
    ("causal", "mask_kind"),
    [
        (False, "none"),
        (False, "all-true"),
        (False, "padding"),
        (False, "scattered"),
        (True, "all-true"),
        (True, "padding"),
        (True, "scattered"),
    ],
    ids=(
        "noncausal-none",
        "noncausal-all-true",
        "noncausal-padding",
        "noncausal-scattered",
        "causal-all-true",
        "causal-padding",
        "causal-scattered",
    ),
)
def test_solution_matches_baseline_without_mutating_inputs(
    causal: bool,
    mask_kind: str,
) -> None:
    _, baseline, solution = _build_models(causal=causal)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(99)
    inputs = torch.randn(2, 4, 8, generator=generator)
    valid_mask: torch.Tensor | None
    if mask_kind == "none":
        valid_mask = None
    elif mask_kind == "all-true":
        valid_mask = torch.ones(2, 4, dtype=torch.bool)
    elif mask_kind == "padding":
        valid_mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False]],
            dtype=torch.bool,
        )
    else:
        valid_mask = torch.tensor(
            [[True, False, True, False], [False, True, True, False]],
            dtype=torch.bool,
        )
    input_snapshot = inputs.clone()
    mask_snapshot = None if valid_mask is None else valid_mask.clone()

    with torch.inference_mode():
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
    assert torch.isfinite(solution_output).all()
    assert torch.equal(inputs, input_snapshot)
    if valid_mask is not None and mask_snapshot is not None:
        assert torch.equal(valid_mask, mask_snapshot)
        invalid_output = solution_output.masked_select(~valid_mask[..., None])
        assert torch.count_nonzero(invalid_output) == 0


def test_causal_solution_accepts_a_sequence_shorter_than_config() -> None:
    _, baseline, solution = _build_models(causal=True)
    inputs = torch.randn(2, 2, 8)
    valid_mask = torch.tensor(
        [[True, False], [True, True]],
        dtype=torch.bool,
    )

    with torch.inference_mode():
        reference = baseline(inputs, valid_mask)
        solution_output = solution(inputs, valid_mask)

    comparison = official.compare_outputs(
        reference,
        solution_output,
        rtol=0.01,
        atol=0.001,
    )
    assert comparison.passed


@pytest.mark.parametrize(
    "policy",
    [
        "triton",
        "preprocess",
        "s512-native-softmax",
        "long-tail-online",
        "wide-triton-inplace",
        "cuda-graph",
        "balanced-cuda-graph",
        "padding",
        "packed",
    ],
)
def test_specialized_gpu_policy_reports_cpu_fallback(
    policy: str,
) -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config(causal=False)).eval()
    solution.configure_runtime_policy(policy=policy)

    execution_path = solution.describe_execution_path()

    assert execution_path["requested_policy"] == policy
    assert execution_path["selected_policy"] == "torch_fallback"
    assert execution_path["fallback_reason"]
    assert execution_path["required_components"]
    assert execution_path["missing_components"]


def test_module_transform_invalidates_cuda_graph_state() -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config(causal=False)).eval()
    solution._cuda_graph_replay = object()
    solution._dispatch_signature = ("fixture",)

    solution.to(dtype=torch.float64)

    assert solution._cuda_graph_replay is None
    assert solution._dispatch_signature is None


def test_balanced_cuda_graph_uses_one_plan_and_invalidates_capture() -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config(causal=False)).eval()
    solution._cuda_graph_replay = object()

    solution.configure_runtime_policy(policy="balanced-cuda-graph")

    assert solution._cuda_graph_replay is None
    execution_path = solution.describe_execution_path()
    assert execution_path["requested_policy"] == "balanced-cuda-graph"
    assert execution_path["requested_attention"] == "auto"
    assert not any(
        hasattr(layer.attention, "attention_policy") for layer in solution.layers
    )


def test_reference_policy_is_an_isolated_attention_control() -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config(causal=False)).eval()
    solution.configure_runtime_policy(policy="reference")

    execution_path = solution.describe_execution_path()

    assert execution_path["selected_policy"] == "reference"
    assert execution_path["resolved_qkv_layout"] == "torch_zero_copy_view"
    assert execution_path["resolved_attention"] == "explicit_reference_order"


def test_long_tail_online_does_not_install_mutable_layer_policies() -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    config = official.TransformerConfig(
        batch_size=1,
        seq_len=2048,
        d_model=512,
        num_heads=8,
        ffn_dim=2048,
        num_layers=4,
        causal=False,
    )
    solution = solution_module.UserOptimizedTransformer(config).eval()
    solution.configure_runtime_policy(policy="long-tail-online")

    execution_path = solution.describe_execution_path()

    assert execution_path["requested_policy"] == "long-tail-online"
    assert execution_path["selected_policy"] == "torch_fallback"
    assert all(
        not hasattr(layer.attention, "attention_policy") for layer in solution.layers
    )


def test_dispatch_is_the_default_and_falls_back_to_auto() -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config(causal=False)).eval()

    execution_path = solution.describe_execution_path()

    assert execution_path["requested_policy"] == "dispatch"
    assert execution_path["selected_policy"] == "auto"
    assert execution_path["dispatch_policy"] == "auto"
    assert execution_path["route_origin"] == "fallback"
