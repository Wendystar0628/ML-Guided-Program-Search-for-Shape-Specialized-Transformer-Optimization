"""Boundary tests for worker IPC and compact benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from runner import supervisor
from runner.contracts import ContractError, RunVariant
from runner.result_contracts import (
    WorkerRequest,
    compact_correctness,
    compact_performance,
    validate_benchmark_performance,
    validate_execution_path,
    validate_workload_execution,
)
from tests.support.runner_fixtures import tiny_protocol, tiny_shape


def test_worker_request_round_trips_shape_variant_and_policy(tmp_path: Path) -> None:
    request = WorkerRequest(
        run_kind="benchmark",
        project_root=tmp_path,
        shape=tiny_shape(),
        variant=RunVariant(),
        protocol=tiny_protocol(),
        device="cuda:0",
        target="solution",
        comparison_mode="paired",
        solution_policy="eager-sdpa",
    )

    parsed = WorkerRequest.from_dict(request.as_dict())

    assert parsed == request
    assert parsed.shape == tiny_shape()
    assert parsed.variant == RunVariant()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda request: request.update(solution_policy=""),
        lambda request: request.update(comparison_mode="baseline_only"),
        lambda request: request.update(comparison_mode="unknown"),
        lambda request: request.update(unexpected=True),
        lambda request: request.pop("comparison_mode"),
        lambda request: request.pop("shape"),
        lambda request: request.pop("variant"),
    ],
    ids=(
        "blank-policy",
        "target-mode-mismatch",
        "unknown-comparison-mode",
        "unknown-field",
        "missing-comparison-mode",
        "missing-shape",
        "missing-variant",
    ),
)
def test_worker_request_rejects_malformed_documents(
    tmp_path: Path,
    mutation: Any,
) -> None:
    request = WorkerRequest(
        run_kind="benchmark",
        project_root=tmp_path,
        shape=tiny_shape(),
        variant=RunVariant(),
        protocol=tiny_protocol(),
        device="cuda:0",
        target="solution",
        comparison_mode="paired",
        solution_policy="eager-sdpa",
    ).as_dict()
    mutation(request)

    with pytest.raises(ContractError):
        WorkerRequest.from_dict(request)


def test_managed_benchmark_forwards_explicit_variant_and_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_request: dict[str, Any] = {}

    def fake_worker(
        _project_root: Path,
        request: dict[str, Any],
        _timeout_seconds: float,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        captured_request.update(request)
        return {
            "outcome": "unsupported",
            "solution_source_sha256": None,
            "environment": None,
            "correctness": None,
            "performance": None,
            "execution_path": None,
            "failure": {
                "stage": "fixture",
                "type": "FixtureUnsupported",
                "message": "fixture",
            },
        }

    monkeypatch.setattr(supervisor, "_run_worker", fake_worker)
    monkeypatch.setattr(
        supervisor,
        "validate_official_snapshot",
        lambda _root: {"combined_sha256": "fixture-official"},
    )
    supervisor.run_managed_benchmark(
        tmp_path,
        workload_set_id="official_transformer_v1",
        workload_sha256="fixture-workload",
        shape=tiny_shape(),
        variant=RunVariant(),
        protocol=tiny_protocol(),
        device="cuda:0",
        solution_policy="eager-sdpa",
        result_dir=tmp_path / "results",
    )

    assert captured_request["shape"] == tiny_shape().as_dict()
    assert captured_request["variant"] == RunVariant().as_dict()
    assert captured_request["comparison_mode"] == "paired"
    assert captured_request["solution_policy"] == "eager-sdpa"


def test_shared_performance_contract_rejects_invalid_percentiles() -> None:
    performance = {
        "comparison_mode": "paired",
        "timer": "cuda_event",
        "sample_count": 2,
        "baseline": {"median_ms": 2.0, "p90_ms": 1.0},
        "target": {"median_ms": 1.0, "p90_ms": 1.1},
        "speedup": 2.0,
    }

    parsed, error = validate_benchmark_performance(
        performance,
        target="solution",
        comparison_mode="paired",
        repeats=2,
        rounds=1,
        expected_timer="cuda_event",
    )

    assert parsed is None
    assert error == "baseline_p90_below_median"


@pytest.mark.parametrize(
    (
        "target",
        "comparison_mode",
        "performance",
        "expected_baseline",
        "expected_target",
    ),
    [
        (
            "baseline",
            "baseline_only",
            {
                "comparison_mode": "baseline_only",
                "timer": "cuda_event",
                "sample_count": 2,
                "baseline": {"median_ms": 2.0, "p90_ms": 2.1},
            },
            2.0,
            None,
        ),
        (
            "solution",
            "paired",
            {
                "comparison_mode": "paired",
                "timer": "cuda_event",
                "sample_count": 2,
                "baseline": {"median_ms": 2.0, "p90_ms": 2.1},
                "target": {"median_ms": 1.0, "p90_ms": 1.1},
                "speedup": 2.0,
            },
            2.0,
            1.0,
        ),
        (
            "solution",
            "target_only",
            {
                "comparison_mode": "target_only",
                "timer": "cuda_event",
                "sample_count": 2,
                "target": {"median_ms": 1.0, "p90_ms": 1.1},
            },
            None,
            1.0,
        ),
    ],
    ids=("baseline-only", "paired", "target-only"),
)
def test_performance_modes_accept_only_their_required_sides(
    target: str,
    comparison_mode: Any,
    performance: dict[str, Any],
    expected_baseline: float | None,
    expected_target: float | None,
) -> None:
    parsed, error = validate_benchmark_performance(
        performance,
        target=target,
        comparison_mode=comparison_mode,
        repeats=2,
        rounds=1,
        expected_timer="cuda_event",
    )

    assert error is None
    assert parsed is not None
    assert (
        None if parsed.baseline is None else parsed.baseline.median_ms
    ) == expected_baseline
    assert (
        None if parsed.target is None else parsed.target.median_ms
    ) == expected_target
    assert parsed.speedup == (2.0 if comparison_mode == "paired" else None)


@pytest.mark.parametrize(
    ("comparison_mode", "forbidden_key", "forbidden_value", "expected_error"),
    [
        (
            "baseline_only",
            "target",
            {"median_ms": 1.0, "p90_ms": 1.0},
            "baseline_only_has_target",
        ),
        ("baseline_only", "speedup", 2.0, "baseline_only_has_speedup"),
        (
            "target_only",
            "baseline",
            {"median_ms": 2.0, "p90_ms": 2.0},
            "target_only_has_baseline",
        ),
        ("target_only", "speedup", 2.0, "target_only_has_speedup"),
    ],
)
def test_unpaired_performance_modes_reject_forbidden_fields(
    comparison_mode: Any,
    forbidden_key: str,
    forbidden_value: Any,
    expected_error: str,
) -> None:
    measured_key = "baseline" if comparison_mode == "baseline_only" else "target"
    performance = {
        "comparison_mode": comparison_mode,
        "timer": "cuda_event",
        "sample_count": 1,
        measured_key: {"median_ms": 1.0, "p90_ms": 1.0},
        forbidden_key: forbidden_value,
    }

    parsed, error = validate_benchmark_performance(
        performance,
        target="baseline" if comparison_mode == "baseline_only" else "solution",
        comparison_mode=comparison_mode,
        repeats=1,
        rounds=1,
        expected_timer="cuda_event",
    )

    assert parsed is None
    assert error == expected_error


@pytest.mark.parametrize("forbidden_key", ["baseline", "speedup"])
def test_target_only_rejects_forbidden_fields_even_when_null(
    forbidden_key: str,
) -> None:
    performance = {
        "comparison_mode": "target_only",
        "timer": "cuda_event",
        "sample_count": 1,
        "target": {"median_ms": 1.0, "p90_ms": 1.0},
        forbidden_key: None,
    }

    parsed, error = validate_benchmark_performance(
        performance,
        target="solution",
        comparison_mode="target_only",
        repeats=1,
        rounds=1,
        expected_timer="cuda_event",
    )

    assert parsed is None
    assert error == f"target_only_has_{forbidden_key}"


def test_target_only_compaction_omits_baseline_and_speedup() -> None:
    compact = compact_performance(
        {
            "timer": "cuda_event",
            "target": {
                "sample_count": 2,
                "median_ms": 1.0,
                "p90_ms": 1.1,
            },
        },
        target="solution",
        comparison_mode="target_only",
    )

    assert compact == {
        "comparison_mode": "target_only",
        "timer": "cuda_event",
        "sample_count": 2,
        "target": {"median_ms": 1.0, "p90_ms": 1.1},
    }


def test_provisional_reference_correctness_fields_survive_compaction() -> None:
    correctness = {
        "passed": True,
        "trial_count": 1,
        "failed_elements": 0,
        "max_abs_error": 0.0001,
        "max_relative_error": 0.001,
        "reference_kind": "internal_query_block",
        "reference_scope": "full_single_sample",
        "validation_level": "provisional",
        "compared_elements": 102_400_000,
        "reference_latency_ms": 35_462.0,
    }

    assert compact_correctness(correctness) == correctness


def test_execution_path_contract_uses_one_common_truth_shape() -> None:
    path = {
        "requested_policy": "eager-sdpa",
        "selected_policy": "eager-sdpa",
        "qkv_projection": "packed",
        "attention_backend": "causal_sdpa",
        "runtime_wrapper": "eager",
        "residual_norm_backend": "torch",
        "causal_mask": "native",
        "valid_token_mask": "none",
        "fallback_reasons": [],
        "execution_mode": "eager",
    }

    assert validate_execution_path(path) is None
    del path["attention_backend"]
    assert validate_execution_path(path) == "missing_attention_backend"


def test_streamed_workload_contract_links_selection_to_measured_schedule() -> None:
    workload = {
        "mode": "batch_streamed",
        "validation_microbatch_size": 1,
        "timing_microbatch_candidates": [1, 2, 4],
        "timing_microbatch_size": 4,
        "microbatch_count": 8,
        "selection": {
            "method": "runtime_policy_and_microbatch_screen",
            "policy": "mixed-fp16-cudnn",
            "timing_microbatch_size": 4,
            "microbatch_count": 8,
            "estimated_logical_batch_ms": 12.0,
            "evidence_sha256": "a" * 64,
        },
        "candidate_screening": [
            {
                "policy": "mixed-fp16-cudnn",
                "comparator_passed": True,
                "timing_schedules": [
                    {
                        "timing_microbatch_size": 4,
                        "microbatch_count": 8,
                        "passed": True,
                    }
                ],
            }
        ],
    }

    assert validate_workload_execution(workload) is None
    workload["selection"]["timing_microbatch_size"] = 2
    assert (
        validate_workload_execution(workload)
        == "streamed_selection_microbatch_mismatch"
    )
