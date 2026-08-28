"""Correctness and public-interface tests for the migrated solution."""

from __future__ import annotations

import inspect
import os
import py_compile
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from official import torch_transformer_benchmark as official
from runner.execution import load_solution_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(*, num_layers: int = 2) -> official.TransformerConfig:
    return official.TransformerConfig(
        batch_size=2,
        seq_len=4,
        d_model=8,
        num_heads=2,
        ffn_dim=8,
        num_layers=num_layers,
        causal=True,
    )


def _build_models(
    *, num_layers: int = 2
) -> tuple[object, torch.nn.Module, torch.nn.Module]:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution_class = solution_module.UserOptimizedTransformer
    signature = inspect.signature(solution_class.forward)
    assert list(signature.parameters) == ["self", "x", "valid_token_mask"]
    assert signature.parameters["valid_token_mask"].default is None

    config = _config(num_layers=num_layers)
    torch.manual_seed(2026)
    baseline = official.BaselineTransformer(config).eval()
    solution = solution_class(config).eval()
    solution_module.copy_model_weights(baseline, solution, strict=True)
    return solution_module, baseline, solution


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
        "8",
        "--layers",
        "1",
        "--causal",
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


def test_strict_weight_hook_packs_qkv_without_mutating_baseline() -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    config = _config()
    torch.manual_seed(2026)
    baseline = official.BaselineTransformer(config).eval()
    before = {key: value.clone() for key, value in baseline.state_dict().items()}
    solution = solution_module.UserOptimizedTransformer(config).eval()

    solution_module.copy_model_weights(baseline, solution, strict=True)

    assert all(torch.equal(before[key], baseline.state_dict()[key]) for key in before)
    solution_state = solution.state_dict()
    for layer_index in range(config.num_layers):
        source_prefix = f"layers.{layer_index}.attention"
        expected_weight = torch.cat(
            [
                before[f"{source_prefix}.{projection}.weight"]
                for projection in ("q_proj", "k_proj", "v_proj")
            ],
            dim=0,
        )
        assert torch.equal(
            solution_state[f"{source_prefix}.qkv_proj.weight"], expected_weight
        )


def test_strict_weight_hook_rejects_incompatible_models() -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    baseline = official.BaselineTransformer(_config(num_layers=1)).eval()
    solution = solution_module.UserOptimizedTransformer(_config(num_layers=2)).eval()

    with pytest.raises(RuntimeError, match="strict weight mapping failed"):
        solution_module.copy_model_weights(baseline, solution, strict=True)


@pytest.mark.parametrize(
    "valid_mask",
    [
        torch.ones(2, 4, dtype=torch.bool),
        torch.tensor(
            [[True, True, False, False], [True, True, True, False]],
            dtype=torch.bool,
        ),
        torch.tensor(
            [[True, False, True, False], [False, True, True, False]],
            dtype=torch.bool,
        ),
    ],
    ids=("all-valid", "padding", "scattered"),
)
def test_causal_solution_matches_official_baseline_without_input_mutation(
    valid_mask: torch.Tensor,
) -> None:
    _, baseline, solution = _build_models()
    inputs = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(99))
    input_snapshot = inputs.clone()
    mask_snapshot = valid_mask.clone()

    with torch.inference_mode():
        reference = baseline(inputs, valid_mask)
        actual = solution(inputs, valid_mask)

    comparison = official.compare_outputs(
        reference,
        actual,
        rtol=0.02,
        atol=0.002,
    )
    assert comparison.passed
    assert actual.shape == inputs.shape
    assert torch.isfinite(actual).all()
    assert torch.equal(inputs, input_snapshot)
    assert torch.equal(valid_mask, mask_snapshot)
    assert torch.count_nonzero(actual.masked_select(~valid_mask[..., None])) == 0


@pytest.mark.parametrize(
    ("policy", "selected", "attention", "wrapper", "batch", "block"),
    [
        ("auto", "auto", "causal_sdpa", "eager", "full", "torch"),
        ("safe", "safe", "safe_streaming", "eager", "full", "torch"),
        (
            "causal-sdpa",
            "causal-sdpa",
            "causal_sdpa",
            "eager",
            "full",
            "torch",
        ),
        (
            "inplace-block",
            "inplace-block",
            "causal_sdpa",
            "eager",
            "full",
            "inplace_exact_gelu",
        ),
        ("graph", "safe", "safe_streaming", "eager", "full", "torch"),
        ("batch-tiled", "safe", "safe_streaming", "eager", "full", "torch"),
    ],
)
def test_explicit_policy_reports_one_honest_execution_plan(
    policy: str,
    selected: str,
    attention: str,
    wrapper: str,
    batch: str,
    block: str,
) -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config()).eval()
    solution.configure_runtime_policy(policy=policy)

    path = solution.describe_execution_path()

    assert path["requested_policy"] == policy
    assert path["selected_policy"] == selected
    assert path["attention_backend"] == attention
    assert path["runtime_wrapper"] == wrapper
    assert path["batch_strategy"] == batch
    assert path["block_backend"] == block


def test_old_shape_specific_policy_names_are_rejected() -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config()).eval()

    with pytest.raises(ValueError, match="unknown runtime policy"):
        solution.configure_runtime_policy(policy="removed-policy")


def test_module_transform_invalidates_cached_runtime_state() -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config()).eval()
    solution._cuda_graph_replay = object()
    solution._runtime_plan = object()
    solution._dispatch_signature = ("fixture",)

    solution.to(dtype=torch.float64)

    assert solution._cuda_graph_replay is None
    assert solution._runtime_plan is None
    assert solution._dispatch_signature is None


def test_execution_observation_does_not_bypass_the_graph_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config()).eval()
    plan = SimpleNamespace(use_batch_tiling=False, use_cuda_graph=True)
    replay_calls: list[str] = []

    class FakeGraphReplay:
        def run(self, function, value, valid_mask):  # type: ignore[no-untyped-def]
            replay_calls.append("replay")
            return function(value, valid_mask)

    def fake_forward_eager(value, _valid_mask, _plan):  # type: ignore[no-untyped-def]
        solution._last_execution_observation = {
            "attention_backends": ["causal_sdpa"],
            "block_backends": ["torch"],
        }
        return value + 1

    monkeypatch.setattr(solution_module, "CudaGraphReplay", FakeGraphReplay)
    monkeypatch.setattr(solution, "_resolve_dispatch", lambda _value: None)
    monkeypatch.setattr(
        solution,
        "_cached_execution_plan",
        lambda _value, _valid_mask: plan,
    )
    monkeypatch.setattr(solution, "_forward_eager", fake_forward_eager)
    solution._cuda_graph_replay = object()
    solution.set_execution_observation(True)

    inputs = torch.zeros(2, 4, 8)
    output = solution(inputs)

    assert replay_calls == ["replay"]
    assert torch.equal(output, inputs + 1)
    assert solution._last_execution_observation["runtime_wrappers"] == ["cuda_graph"]


def test_dispatch_is_the_default_and_falls_back_to_auto() -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config()).eval()

    path = solution.describe_execution_path()

    assert path["requested_policy"] == "dispatch"
    assert path["selected_policy"] == "auto"
    assert path["dispatch_policy"] == "auto"
    assert path["route_origin"] == "fallback"
