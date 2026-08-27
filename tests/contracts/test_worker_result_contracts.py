"""Boundary tests for explicit worker requests and shared result validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from runner import supervisor
from runner.contracts import ContractError, MeasurementProtocol, WorkloadCase
from runner.result_contracts import WorkerRequest, validate_benchmark_performance


def _case() -> WorkloadCase:
    return WorkloadCase(
        case_id="fixture",
        batch_size=1,
        seq_len=16,
        d_model=32,
        num_heads=4,
        ffn_dim=64,
        num_layers=1,
        dtype="float16",
        causal=False,
        padding_ratio=0.0,
    )


def test_worker_request_round_trips_an_explicit_solution_policy(
    tmp_path: Path,
) -> None:
    request = WorkerRequest(
        run_kind="benchmark",
        project_root=tmp_path,
        case=_case(),
        protocol=MeasurementProtocol.for_preset("smoke"),
        device="cuda:0",
        target="solution",
        solution_policy="preprocess",
    )

    parsed = WorkerRequest.from_dict(request.as_dict())

    assert parsed == request
    assert parsed.solution_policy == "preprocess"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda request: request.update(solution_policy=""),
        lambda request: request.update(unexpected=True),
        lambda request: request.pop("case"),
    ],
    ids=("blank-policy", "unknown-field", "missing-case"),
)
def test_worker_request_rejects_malformed_documents(
    tmp_path: Path,
    mutation: Any,
) -> None:
    request = WorkerRequest(
        run_kind="benchmark",
        project_root=tmp_path,
        case=_case(),
        protocol=MeasurementProtocol.for_preset("smoke"),
        device="cuda:0",
        target="solution",
        solution_policy="auto",
    ).as_dict()
    mutation(request)

    with pytest.raises(ContractError):
        WorkerRequest.from_dict(request)


def test_managed_benchmark_forwards_explicit_policy(
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
                "exit_code": None,
            },
        }

    monkeypatch.setattr(supervisor, "_run_worker", fake_worker)
    monkeypatch.setattr(
        supervisor,
        "validate_official_snapshot",
        lambda _root: {"sha256": "fixture-official"},
    )
    supervisor.run_managed_benchmark(
        tmp_path,
        workload_set_id="fixture",
        workload_sha256="fixture-workload",
        case=_case(),
        protocol=MeasurementProtocol.for_preset("smoke"),
        device="cuda:0",
        solution_policy="preprocess",
        result_dir=tmp_path / "results",
    )

    assert captured_request["solution_policy"] == "preprocess"


def test_shared_performance_contract_rejects_invalid_percentiles() -> None:
    performance = {
        "timer": "cuda_event",
        "sample_count": 2,
        "baseline": {
            "median_ms": 2.0,
            "p90_ms": 1.0,
            "round_medians_ms": [2.0],
        },
        "target": {
            "median_ms": 1.0,
            "p90_ms": 1.1,
            "round_medians_ms": [1.0],
        },
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
