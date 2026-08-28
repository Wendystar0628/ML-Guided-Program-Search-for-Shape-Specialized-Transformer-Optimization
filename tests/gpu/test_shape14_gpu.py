"""Opt-in real-GPU coverage for the full logical Shape-14 stream."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from runner.candidates import candidate_specs_for_execution_mode
from runner.contracts import MeasurementProtocol, RunVariant
from runner.result_contracts import WorkerRequest
from runner.streamed_execution import execute_streamed_benchmark
from runner.workload_execution import plan_workload_execution
from tests.support.runner_fixtures import official_shape

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RUN_SHAPE14_GPU = os.environ.get("RUN_SHAPE14_GPU") == "1"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="real Shape-14 coverage requires a CUDA GPU",
    ),
    pytest.mark.skipif(
        not _RUN_SHAPE14_GPU,
        reason="set RUN_SHAPE14_GPU=1 to run the long Shape-14 smoke test",
    ),
]


def test_shape14_completes_every_microbatch_with_registered_policy() -> None:
    device = torch.device("cuda:0")
    if torch.cuda.get_device_properties(device).total_memory < 8 * 1024**3:
        pytest.skip("Shape-14 B=1 validation requires at least 8 GiB device memory")

    shape = official_shape("official_14")
    variant = RunVariant()
    protocol = MeasurementProtocol(
        preset="smoke",
        seed=17,
        accuracy_trials=1,
        warmup=0,
        repeats=1,
        rounds=1,
        timeout_seconds=300.0,
    )
    plan = plan_workload_execution(shape, variant)
    request = WorkerRequest(
        run_kind="benchmark",
        project_root=PROJECT_ROOT,
        shape=shape,
        variant=variant,
        protocol=protocol,
        device="cuda:0",
        target="solution",
        comparison_mode="target_only",
        solution_policy="screen",
    )

    result = execute_streamed_benchmark(request, plan)

    assert result["outcome"] == "success", result.get("failure")
    assert result["correctness"]["passed"] is True
    assert result["correctness"]["validation_level"] == "provisional"
    assert result["performance"]["comparison_mode"] == "target_only"
    assert "baseline" not in result["performance"]
    assert "speedup" not in result["performance"]
    workload = result["workload_execution"]
    timing_size = workload["timing_microbatch_size"]
    expected_microbatches = shape.batch_size // timing_size
    assert timing_size in plan.timing_microbatch_candidates
    assert workload["microbatches_per_sample"] == expected_microbatches
    assert workload["completed_microbatches"] == expected_microbatches
    assert workload["end_to_end_microbatches"] == expected_microbatches
    supported = {
        spec.solution_policy
        for spec in candidate_specs_for_execution_mode(
            shape,
            variant,
            plan.execution_mode,
        )
    }
    assert workload["selection"]["policy"] in supported
    assert (
        result["execution_path"]["selected_policy"] == workload["selection"]["policy"]
    )
    assert result["execution_path"].get("dispatch_policy") is None
