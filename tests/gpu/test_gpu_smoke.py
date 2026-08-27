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
from runner.contracts import (
    MeasurementProtocol,
    WorkloadCase,
    load_json,
    load_workload_set,
)
from runner.execution import execute_benchmark
from runner.result_contracts import WorkerRequest
from runner.supervisor import run_managed_benchmark
from runner.verified_hardware import (
    VerifiedHardwareError,
    collect_runtime_identity,
    expected_hardware_identity,
    validate_hardware_identity,
)

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


@pytest.mark.parametrize(
    ("case_id", "expected_policy"),
    (
        ("launch_s64_fp16", "cuda-graph"),
        ("balanced_s128_fp32", "auto"),
        ("balanced_s128_fp16", "balanced-cuda-graph"),
        ("attention_s2048_fp16", "long-tail-online"),
        ("mask_s512_full_fp16", "s512-native-softmax"),
        ("wide_s256_bf16", "wide-triton-inplace"),
    ),
)
def test_default_dispatch_runs_deployed_policy_through_a_managed_worker(
    tmp_path: Path,
    case_id: str,
    expected_policy: str,
) -> None:
    """Exercise the checked-in device route through the real process boundary."""

    bundle_root = PROJECT_ROOT / "verified_hardware" / "nvidia_geforce_rtx_4080"
    expected_identity = expected_hardware_identity(
        load_json(bundle_root / "profile.json")
    )
    actual_identity = collect_runtime_identity("cuda:0")
    try:
        validate_hardware_identity(expected_identity, actual_identity)
    except VerifiedHardwareError as exc:
        pytest.skip(f"checked-in RTX 4080 Bundle does not target this GPU: {exc}")

    workload_set = load_workload_set(PROJECT_ROOT, "transformer_core_v1")
    case = next(case for case in workload_set.cases if case.case_id == case_id)
    result, result_path = run_managed_benchmark(
        PROJECT_ROOT,
        workload_set_id=workload_set.workload_set_id,
        workload_sha256=workload_set.sha256,
        case=case,
        protocol=_smoke_protocol(),
        device="cuda:0",
        target="solution",
        solution_policy="dispatch",
        result_dir=tmp_path,
    )

    assert result_path.parent == tmp_path
    assert result["outcome"] == "success", result.get("failure")
    assert result["correctness"]["passed"] is True
    execution_path = result["execution_path"]
    assert isinstance(execution_path, dict)
    assert execution_path["route_origin"] == "calibrated"
    assert execution_path["dispatch_policy"] == expected_policy
    spec = candidate_spec_for_policy(case, expected_policy, deployable_only=True)
    assert spec is not None
    assert spec.dispatch_evidence_matches(execution_path), execution_path
