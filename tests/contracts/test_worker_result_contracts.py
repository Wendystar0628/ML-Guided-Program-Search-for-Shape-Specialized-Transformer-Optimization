"""Boundary tests for worker IPC and compact benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from runner import supervisor
from runner.contracts import ContractError, RunVariant
from runner.result_contracts import (
    WorkerRequest,
    validate_benchmark_performance,
    validate_execution_path,
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
        solution_policy="auto",
    )

    parsed = WorkerRequest.from_dict(request.as_dict())

    assert parsed == request
    assert parsed.shape == tiny_shape()
    assert parsed.variant == RunVariant()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda request: request.update(solution_policy=""),
        lambda request: request.update(unexpected=True),
        lambda request: request.pop("shape"),
        lambda request: request.pop("variant"),
    ],
    ids=("blank-policy", "unknown-field", "missing-shape", "missing-variant"),
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
        solution_policy="auto",
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
        solution_policy="auto",
        result_dir=tmp_path / "results",
    )

    assert captured_request["shape"] == tiny_shape().as_dict()
    assert captured_request["variant"] == RunVariant().as_dict()
    assert captured_request["solution_policy"] == "auto"


def test_shared_performance_contract_rejects_invalid_percentiles() -> None:
    performance = {
        "timer": "cuda_event",
        "sample_count": 2,
        "baseline": {"median_ms": 2.0, "p90_ms": 1.0},
        "target": {"median_ms": 1.0, "p90_ms": 1.1},
        "speedup": 2.0,
    }

    parsed, error = validate_benchmark_performance(
        performance,
        target="solution",
        repeats=2,
        rounds=1,
        expected_timer="cuda_event",
    )

    assert parsed is None
    assert error == "baseline_p90_below_median"


def test_execution_path_contract_uses_one_common_truth_shape() -> None:
    path = {
        "requested_policy": "auto",
        "selected_policy": "auto",
        "qkv_projection": "packed",
        "attention_backend": "causal_sdpa",
        "runtime_wrapper": "eager",
        "block_backend": "torch",
        "causal_mask": "native",
        "valid_token_mask": "none",
        "fallback_reasons": [],
        "execution_mode": "eager",
    }

    assert validate_execution_path(path) is None
    del path["attention_backend"]
    assert validate_execution_path(path) == "missing_attention_backend"
