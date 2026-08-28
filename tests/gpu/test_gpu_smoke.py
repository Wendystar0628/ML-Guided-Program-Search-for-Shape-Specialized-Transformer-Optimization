"""Short real-CUDA checks for the official-shape execution paths."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from runner.candidates import candidate_spec
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


def _run_policy(
    policy: str,
    *,
    case_id: str = "official_02",
) -> dict[str, object]:
    request = WorkerRequest(
        run_kind="benchmark",
        project_root=PROJECT_ROOT,
        shape=official_shape(case_id),
        variant=RunVariant(),
        protocol=MeasurementProtocol(
            preset="smoke",
            seed=17,
            accuracy_trials=2,
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


def _assert_policy_executed(result: dict[str, object], policy: str) -> None:
    assert result["outcome"] == "success", result.get("failure")
    assert result["correctness"]["passed"] is True
    execution_path = result["execution_path"]
    assert execution_path["selected_policy"] == policy
    candidate_id = {
        "auto": "eager-auto",
        "safe": "eager-safe",
    }.get(policy, policy)
    candidate = candidate_spec(candidate_id)
    assert candidate is not None
    assert candidate.evidence_matches(execution_path)
    assert result["performance"]["timer"] == "cuda_event"
    assert result["performance"]["target"]["median_ms"] > 0


@pytest.mark.parametrize(
    "policy",
    ["auto", "safe", "graph", "inplace-block"],
)
def test_official_02_policy_executes_and_passes_the_official_comparator(
    policy: str,
) -> None:
    result = _run_policy(policy)

    _assert_policy_executed(result, policy)


def test_official_13_inplace_path_executes_with_observed_evidence() -> None:
    result = _run_policy("inplace-block", case_id="official_13")

    _assert_policy_executed(result, "inplace-block")
