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
from runner.model_runtime import load_solution_module
from solution.cuda_graph import CudaGraphReplay

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
    ("policy", "selected", "attention", "wrapper", "residual_norm"),
    [
        ("eager-sdpa", "eager-sdpa", "causal_sdpa", "eager", "torch"),
        ("safe", "safe", "safe_streaming", "eager", "torch"),
        ("graph", "safe", "safe_streaming", "eager", "torch"),
        ("graph-fused-norm", "safe", "safe_streaming", "eager", "torch"),
        (
            "mixed-fp16-efficient",
            "safe",
            "safe_streaming",
            "eager",
            "torch",
        ),
        (
            "mixed-fp16-cudnn",
            "safe",
            "safe_streaming",
            "eager",
            "torch",
        ),
        (
            "mixed-fp16-core-efficient",
            "safe",
            "safe_streaming",
            "eager",
            "torch",
        ),
        (
            "mixed-fp16-core-efficient-triton-norm",
            "safe",
            "safe_streaming",
            "eager",
            "torch",
        ),
        (
            "mixed-fp16-core-cudnn",
            "safe",
            "safe_streaming",
            "eager",
            "torch",
        ),
        (
            "graph-mixed-fp16-efficient",
            "safe",
            "safe_streaming",
            "eager",
            "torch",
        ),
        (
            "graph-mixed-fp16-efficient-compiled-norm",
            "safe",
            "safe_streaming",
            "eager",
            "torch",
        ),
        (
            "graph-mixed-fp16-core-efficient-compiled-norm",
            "safe",
            "safe_streaming",
            "eager",
            "torch",
        ),
        (
            "batch-tiled-mixed-fp16-core-efficient-compiled-norm",
            "safe",
            "safe_streaming",
            "eager",
            "torch",
        ),
        (
            "batch-tiled-mixed-fp16-core-efficient-triton-mixed-norm",
            "safe",
            "safe_streaming",
            "eager",
            "torch",
        ),
    ],
)
def test_explicit_policy_reports_one_honest_execution_plan(
    policy: str,
    selected: str,
    attention: str,
    wrapper: str,
    residual_norm: str,
) -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config()).eval()
    solution.configure_runtime_policy(policy=policy)

    path = solution.describe_execution_path()

    assert path["requested_policy"] == policy
    assert path["selected_policy"] == selected
    assert path["attention_backend"] == attention
    assert path["runtime_wrapper"] == wrapper
    assert path["residual_norm_backend"] == residual_norm
    assert "batch_strategy" not in path
    assert "batch_tile_size" not in path
    assert "layer_backends" not in path


def test_unknown_policy_name_is_rejected() -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config()).eval()

    with pytest.raises(ValueError, match="unknown runtime policy"):
        solution.configure_runtime_policy(policy="removed-policy")


def test_module_transform_invalidates_cached_runtime_state() -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config()).eval()
    solution._cuda_graph_replay = object()
    solution._batch_tiled_graph_replay = object()
    solution._runtime_plan = object()
    solution._dispatch_signature = ("fixture",)

    solution.to(dtype=torch.float64)

    assert solution._cuda_graph_replay is None
    assert solution._batch_tiled_graph_replay is None
    assert solution._runtime_plan is None
    assert solution._dispatch_signature is None


def test_execution_plan_cache_tracks_cudnn_sdpa_runtime_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config()).eval()
    inputs = torch.zeros(2, 4, 8)
    runtime = {"cudnn": False}
    resolutions: list[bool] = []
    original_resolve = solution._execution_plan

    monkeypatch.setattr(
        torch.backends.cuda,
        "cudnn_sdp_enabled",
        lambda: runtime["cudnn"],
    )

    def counted_resolve(value, valid_mask):  # type: ignore[no-untyped-def]
        resolutions.append(runtime["cudnn"])
        return original_resolve(value, valid_mask)

    monkeypatch.setattr(solution, "_execution_plan", counted_resolve)
    with torch.inference_mode():
        solution._cached_execution_plan(inputs, None)
        solution._cached_execution_plan(inputs, None)
        runtime["cudnn"] = True
        solution._cached_execution_plan(inputs, None)

    assert resolutions == [False, True]


def test_execution_observation_does_not_bypass_the_graph_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config()).eval()
    plan = SimpleNamespace(
        use_batch_tiled_cuda_graph=False,
        use_compiled_forward=False,
        use_cuda_graph=True,
    )
    replay_calls: list[str] = []

    class FakeGraphReplay:
        def run(self, function, value, valid_mask):  # type: ignore[no-untyped-def]
            replay_calls.append("replay")
            return function(value, valid_mask)

    def fake_forward_eager(value, _valid_mask, _plan):  # type: ignore[no-untyped-def]
        solution._last_execution_observation = {
            "attention_backends": ["causal_sdpa"],
            "residual_norm_backends": ["torch", "torch"],
            "expected_layers": 1,
            "complete": True,
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


def test_execution_observation_reports_batch_tiled_graph_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config()).eval()
    plan = SimpleNamespace(
        use_batch_tiled_cuda_graph=True,
        batch_tile_size=128,
        use_cuda_graph=False,
    )
    replay_tiles: list[int] = []

    class FakeBatchTiledGraphReplay:
        def __init__(self, tile_size: int) -> None:
            replay_tiles.append(tile_size)

        def run(self, function, value, valid_mask):  # type: ignore[no-untyped-def]
            return function(value, valid_mask)

    def fake_forward_eager(value, _valid_mask, _plan):  # type: ignore[no-untyped-def]
        solution._last_execution_observation = {
            "attention_backends": ["mixed_fp16_efficient"],
            "residual_norm_backends": [
                "compiled_residual_layer_norm",
                "compiled_residual_layer_norm",
            ],
            "expected_layers": 1,
            "complete": True,
        }
        return value + 1

    monkeypatch.setattr(
        solution_module,
        "BatchTiledGraphReplay",
        FakeBatchTiledGraphReplay,
    )
    monkeypatch.setattr(solution, "_resolve_dispatch", lambda _value: None)
    monkeypatch.setattr(
        solution,
        "_cached_execution_plan",
        lambda _value, _valid_mask: plan,
    )
    monkeypatch.setattr(solution, "_forward_eager", fake_forward_eager)
    solution.set_execution_observation(True)

    inputs = torch.zeros(2, 4, 8)
    output = solution(inputs)

    assert replay_tiles == [128]
    assert torch.equal(output, inputs + 1)
    assert solution._last_execution_observation["runtime_wrappers"] == [
        "batch_tiled_cuda_graph"
    ]


def test_mixed_residual_stream_keeps_branch_updates_in_fp16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config()).eval()
    branch_inputs: list[torch.dtype] = []
    boundary_calls: list[tuple[torch.dtype, torch.dtype, bool]] = []

    def attention_update(
        _layer,
        normalized,
        _valid_mask,
        _causal,
        _plan,
        _observation,
    ):  # type: ignore[no-untyped-def]
        branch_inputs.append(normalized.dtype)
        return torch.ones_like(normalized, dtype=torch.float16)

    def ffn_update(
        _layer,
        normalized,
        _plan,
        _observation,
    ):  # type: ignore[no-untyped-def]
        branch_inputs.append(normalized.dtype)
        return torch.ones_like(normalized, dtype=torch.float16)

    def mixed_residual_norm(
        value,
        update,
        layer_norm,
        *,
        final_boundary,
    ):  # type: ignore[no-untyped-def]
        boundary_calls.append((value.dtype, update.dtype, final_boundary))
        residual = value + update.float()
        normalized = layer_norm(residual).to(
            torch.float32 if final_boundary else torch.float16
        )
        return residual, normalized, "triton_mixed_residual_layer_norm"

    monkeypatch.setattr(
        solution_module._TransformerBlock,
        "attention_update",
        attention_update,
    )
    monkeypatch.setattr(
        solution_module._TransformerBlock,
        "ffn_update",
        ffn_update,
    )
    monkeypatch.setattr(
        solution_module,
        "triton_mixed_residual_layer_norm",
        mixed_residual_norm,
    )
    plan = SimpleNamespace(
        residual_norm_backend="triton_mixed_residual_layer_norm",
    )

    output = solution._forward_eager(torch.zeros(2, 4, 8), None, plan)

    assert branch_inputs == [torch.float16] * 4
    assert boundary_calls == [
        (torch.float32, torch.float16, False),
        (torch.float32, torch.float16, False),
        (torch.float32, torch.float16, False),
        (torch.float32, torch.float16, True),
    ]
    assert output.dtype == torch.float32


def test_execution_observation_reports_compiled_forward_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config()).eval()
    plan = SimpleNamespace(
        use_batch_tiled_cuda_graph=False,
        use_compiled_forward=True,
        use_cuda_graph=False,
        compile_mode="max-autotune-no-cudagraphs",
    )
    compiled_calls: list[tuple[object, str]] = []

    class FakeCompiledForward:
        def run(
            self,
            function,
            value,
            valid_mask,
            *,
            plan_key,
            compile_mode,
        ):  # type: ignore[no-untyped-def]
            compiled_calls.append((plan_key, compile_mode))
            return function(value, valid_mask)

    def fake_forward_eager(
        value,
        _valid_mask,
        _plan,
        *,
        observe=True,
    ):  # type: ignore[no-untyped-def]
        if observe:
            solution._last_execution_observation = {
                "attention_backends": ["mixed_fp16_efficient"],
                "residual_norm_backends": ["torch", "torch"],
                "expected_layers": 1,
                "complete": True,
            }
        return value + 1

    monkeypatch.setattr(solution, "_resolve_dispatch", lambda _value: None)
    monkeypatch.setattr(
        solution,
        "_cached_execution_plan",
        lambda _value, _valid_mask: plan,
    )
    monkeypatch.setattr(solution, "_forward_eager", fake_forward_eager)
    solution._compiled_forward = FakeCompiledForward()
    solution.set_execution_observation(True)

    inputs = torch.zeros(2, 4, 8)
    output = solution(inputs)

    assert compiled_calls == [(plan, "max-autotune-no-cudagraphs")]
    assert torch.equal(output, inputs + 1)
    assert solution._last_execution_observation["runtime_wrappers"] == [
        "compiled_forward"
    ]


def test_all_valid_mask_cache_does_not_go_stale_for_inference_tensors() -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config()).eval()
    inputs = torch.zeros(2, 4, 8)

    with torch.inference_mode():
        mask = torch.ones(2, 4, dtype=torch.bool)
        assert solution._effective_valid_token_mask(inputs, mask) is None
        mask[0, 0] = False
        assert solution._effective_valid_token_mask(inputs, mask) is mask


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones(2, 4),
        torch.ones(2, 4, 1, dtype=torch.bool),
        torch.ones(1, 4, dtype=torch.bool),
    ],
)
def test_only_legal_boolean_batch_sequence_masks_can_be_elided(
    mask: torch.Tensor,
) -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config()).eval()
    inputs = torch.zeros(2, 4, 8)

    assert solution._effective_valid_token_mask(inputs, mask) is mask


def test_execution_observation_reports_complete_layer_coverage() -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config()).eval()
    solution.configure_runtime_policy(policy="safe")
    solution.set_execution_observation(True)

    with torch.inference_mode():
        solution(torch.randn(2, 4, 8), torch.ones(2, 4, dtype=torch.bool))

    observation = solution.describe_execution_path()["observed_execution"]
    assert observation == {
        "attention_backends": ["safe_streaming", "safe_streaming"],
        "residual_norm_backends": ["torch", "torch", "torch", "torch"],
        "expected_layers": 2,
        "complete": True,
    }


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_mixed_cudnn_policy_reports_the_backend_that_passed_comparator() -> None:
    if not torch.backends.cudnn.is_available():
        pytest.skip("cuDNN is unavailable")
    solution_module = load_solution_module(PROJECT_ROOT)
    config = official.TransformerConfig(
        batch_size=1,
        seq_len=1024,
        d_model=64,
        num_heads=1,
        ffn_dim=64,
        num_layers=1,
        causal=True,
    )
    torch.manual_seed(2031)
    baseline = official.BaselineTransformer(config).eval().cuda()
    solution = solution_module.UserOptimizedTransformer(config).eval().cuda()
    solution_module.copy_model_weights(baseline, solution, strict=True)
    solution.configure_runtime_policy(policy="mixed-fp16-cudnn")
    solution.set_execution_observation(True)
    inputs = torch.randn(1, 1024, 64, device="cuda")
    valid_mask = torch.ones(1, 1024, dtype=torch.bool, device="cuda")
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
    path = solution.describe_execution_path()
    assert comparison.passed
    assert torch.equal(inputs, input_snapshot)
    assert torch.equal(valid_mask, mask_snapshot)
    assert path["selected_policy"] == "mixed-fp16-cudnn"
    assert path["attention_backend"] == "mixed_fp16_cudnn"
    assert path["observed_execution"] == {
        "attention_backends": ["mixed_fp16_cudnn"],
        "residual_norm_backends": ["torch", "torch"],
        "expected_layers": 1,
        "complete": True,
    }


def test_graph_policy_fails_clearly_under_torch_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config()).eval()
    solution.configure_runtime_policy(policy="graph")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    with pytest.raises(RuntimeError, match="cannot run under torch.compile"):
        solution(torch.zeros(2, 4, 8))


def test_cuda_graph_replay_rejects_a_second_signature_without_recapture() -> None:
    replay = CudaGraphReplay()
    first = torch.zeros(1, 4, 8)
    replay._signature = replay._input_signature(first, None)
    replay._graph = SimpleNamespace()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="one static input signature"):
        replay.run(lambda value, _mask: value, torch.zeros(2, 4, 8), None)


def test_dispatch_is_the_default_and_falls_back_to_eager_sdpa() -> None:
    solution_module = load_solution_module(PROJECT_ROOT)
    solution = solution_module.UserOptimizedTransformer(_config()).eval()

    path = solution.describe_execution_path()

    assert path["requested_policy"] == "dispatch"
    assert path["selected_policy"] == "eager-sdpa"
    assert path["dispatch_policy"] == "eager-sdpa"
    assert path["route_origin"] == "fallback"
