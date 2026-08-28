"""Short real-CUDA checks for the official-shape execution paths."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from runner.contracts import MeasurementProtocol, RunVariant
from runner.execution import execute_benchmark
from runner.result_contracts import WorkerRequest
from tests.support.runner_fixtures import official_shape

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="real CUDA smoke tests require a GPU",
    ),
]


def _run_policy(policy: str) -> dict[str, object]:
    request = WorkerRequest(
        run_kind="benchmark",
        project_root=PROJECT_ROOT,
        shape=official_shape("official_02"),
        variant=RunVariant(),
        protocol=MeasurementProtocol(
            preset="smoke",
            seed=17,
            accuracy_trials=1,
            warmup=1,
            repeats=2,
            rounds=1,
            timeout_seconds=120.0,
        ),
        device="cuda:0",
        target="solution",
        solution_policy=policy,
    )
    return execute_benchmark(request)


@pytest.mark.parametrize(
    "policy",
    ["safe", "causal-sdpa", "graph", "inplace-block"],
)
def test_official_02_policy_executes_and_passes_the_official_comparator(
    policy: str,
) -> None:
    result = _run_policy(policy)

    assert result["outcome"] == "success", result.get("failure")
    assert result["correctness"]["passed"] is True
    assert result["execution_path"]["selected_policy"] == policy
    assert result["performance"]["timer"] == "cuda_event"
    assert result["performance"]["target"]["median_ms"] > 0
