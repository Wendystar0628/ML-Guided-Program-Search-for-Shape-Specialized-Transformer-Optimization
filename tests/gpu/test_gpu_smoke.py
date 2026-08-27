"""Real-device smoke tests for the deployable CUDA execution paths.

These tests intentionally run the production benchmark implementation so the
official comparator, CUDA Event timer, explicit worker policy, and execution
evidence are exercised together. They do not assert a speedup or run a Formal
measurement protocol.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from runner.candidates import candidate_spec_for_policy
from runner.contracts import MeasurementProtocol, WorkloadCase
from runner.execution import execute_benchmark
from runner.result_contracts import WorkerRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("real CUDA smoke tests require a CUDA-capable device")


@pytest.fixture(autouse=True)
def _release_cuda_cache() -> None:
    yield
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _smoke_protocol() -> MeasurementProtocol:
    protocol = MeasurementProtocol(
        preset="smoke",
        accuracy_trials=1,
        warmup=1,
        repeats=2,
        rounds=1,
        timeout_seconds=120.0,
    )
    protocol.validate()
    return protocol


def _run_policy(case: WorkloadCase, policy: str) -> dict[str, object]:
    result = execute_benchmark(
        WorkerRequest(
            run_kind="benchmark",
            project_root=PROJECT_ROOT,
            case=case,
            protocol=_smoke_protocol(),
            device="cuda:0",
            target="solution",
            solution_policy=policy,
        )
    )
    assert result["outcome"] == "success", result.get("failure")
    correctness = result["correctness"]
    assert isinstance(correctness, dict)
    assert correctness["passed"] is True
    assert correctness["failed_elements"] == 0

    execution_path = result["execution_path"]
    assert isinstance(execution_path, dict)
    assert execution_path["requested_policy"] == policy
    spec = candidate_spec_for_policy(case, policy, deployable_only=True)
    assert spec is not None
    assert spec.evidence_matches(execution_path), execution_path

    performance = result["performance"]
    assert isinstance(performance, dict)
    assert performance["timer"] == "cuda_event"
    target = performance["target"]
    assert isinstance(target, dict)
    assert math.isfinite(float(target["median_ms"]))
    assert float(target["median_ms"]) > 0
    assert math.isfinite(float(target["p90_ms"]))
    assert float(target["p90_ms"]) > 0
    return result


def test_fp16_preprocess_handles_causal_padding_with_official_comparator() -> None:
    pytest.importorskip("triton")
    case = WorkloadCase(
        case_id="gpu_preprocess_causal_padding",
        batch_size=1,
        seq_len=64,
        d_model=256,
        num_heads=8,
        ffn_dim=512,
        num_layers=1,
        dtype="float16",
        causal=True,
        padding_ratio=0.5,
    )

    result = _run_policy(case, "preprocess")

    execution_path = result["execution_path"]
    assert isinstance(execution_path, dict)
    assert execution_path["selected_policy"] == "preprocess"
    assert execution_path["selected_attention_backend"] == (
        "triton_preprocess_native_softmax"
    )
    assert execution_path["token_mask_preprocessing"] == (
        "triton_direct_causal_and_key_mask"
    )


def test_bfloat16_padding_fusion_runs_on_real_cuda() -> None:
    pytest.importorskip("triton")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("this CUDA device does not support bfloat16")
    case = WorkloadCase(
        case_id="gpu_bf16_padding_fusion",
        batch_size=2,
        seq_len=64,
        d_model=128,
        num_heads=4,
        ffn_dim=256,
        num_layers=1,
        dtype="bfloat16",
        causal=False,
        padding_ratio=0.5,
    )

    result = _run_policy(case, "padding")

    execution_path = result["execution_path"]
    assert isinstance(execution_path, dict)
    assert execution_path["selected_policy"] == "padding"
    assert execution_path["block_fusion"] == ("triton_residual_add_padding_when_masked")


def test_fp16_solution_cuda_graph_captures_and_replays() -> None:
    case = WorkloadCase(
        case_id="gpu_launch_cuda_graph",
        batch_size=1,
        seq_len=64,
        d_model=256,
        num_heads=8,
        ffn_dim=1024,
        num_layers=4,
        dtype="float16",
        causal=False,
        padding_ratio=0.0,
    )

    result = _run_policy(case, "cuda-graph")

    execution_path = result["execution_path"]
    assert isinstance(execution_path, dict)
    assert execution_path["selected_policy"] == "cuda-graph"
    assert execution_path["runtime_wrapper"] == "solution_eager_cuda_graph"
    assert execution_path["shape_route"] == "launch_fp16_eager_cuda_graph"
