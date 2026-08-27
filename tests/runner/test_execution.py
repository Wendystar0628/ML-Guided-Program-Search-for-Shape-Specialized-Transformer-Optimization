"""Tests for execution-path guards and timing aggregation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from official import torch_transformer_benchmark as official
from runner.contracts import ContractError, MeasurementProtocol, WorkloadCase
from runner.execution import (
    PreparedExecution,
    _validate_cuda_graph_composition,
    _validate_profile_execution_path,
    execute_benchmark,
    execute_profile,
    prepare_execution,
    run_performance,
)
from runner.result_contracts import WorkerRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _tiny_case() -> WorkloadCase:
    return WorkloadCase(
        case_id="execution_fixture",
        batch_size=1,
        seq_len=2,
        d_model=4,
        num_heads=1,
        ffn_dim=8,
        num_layers=1,
        dtype="float32",
        causal=False,
        padding_ratio=0.0,
        input_scale=1.0,
    )


def _request(run_kind: str, protocol: MeasurementProtocol) -> WorkerRequest:
    return WorkerRequest(
        run_kind=run_kind,  # type: ignore[arg-type]
        project_root=PROJECT_ROOT,
        case=_tiny_case(),
        protocol=protocol,
        device="cpu",
        target="baseline",
    )


@pytest.mark.parametrize(
    "protocol",
    [
        MeasurementProtocol(preset="smoke", compile_solution=True),
        MeasurementProtocol(preset="smoke", cuda_graph_solution=True),
    ],
)
def test_solution_graph_rejects_compile_or_outer_graph(
    protocol: MeasurementProtocol,
) -> None:
    with pytest.raises(ContractError, match="Solution CUDA Graph"):
        _validate_cuda_graph_composition(
            {"runtime_wrapper": "solution_eager_cuda_graph"},
            protocol,
        )


def test_operator_profile_rejects_the_solution_graph_wrapper() -> None:
    with pytest.raises(ContractError, match="hides per-operator"):
        _validate_profile_execution_path(
            {"runtime_wrapper": "solution_eager_cuda_graph"}
        )


def test_prepare_execution_returns_one_frozen_shared_context() -> None:
    prepared = prepare_execution(
        _request(
            "benchmark",
            MeasurementProtocol(
                preset="smoke",
                accuracy_trials=1,
                warmup=0,
                repeats=1,
                rounds=1,
            ),
        ),
        expected_run_kind="benchmark",
    )

    assert isinstance(prepared, PreparedExecution)
    assert prepared.target_model is prepared.baseline
    assert prepared.device.type == "cpu"
    with pytest.raises(FrozenInstanceError):
        prepared.device = torch.device("cuda")  # type: ignore[misc]


@pytest.mark.parametrize(
    ("run_kind", "execute"),
    [("benchmark", execute_benchmark), ("profile", execute_profile)],
)
def test_cpu_formal_is_rejected_by_the_shared_preparation_path(
    run_kind: str,
    execute: Any,
) -> None:
    response = execute(
        _request(
            run_kind,
            MeasurementProtocol(
                preset="formal",
                accuracy_trials=1,
                warmup=0,
                repeats=1,
                rounds=1,
            ),
        )
    )

    assert response["outcome"] == "unsupported"
    assert response["failure"]["stage"] == "device"
    assert response["failure"]["message"] == (
        "CPU execution is supported only by the smoke preset"
    )


def test_performance_alternates_order_and_uses_all_raw_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = WorkloadCase(
        case_id="timing_fixture",
        batch_size=1,
        seq_len=2,
        d_model=4,
        num_heads=1,
        ffn_dim=8,
        num_layers=1,
        dtype="float32",
        causal=False,
        padding_ratio=0.0,
        input_scale=1.5,
    )
    protocol = MeasurementProtocol(
        preset="smoke",
        seed=17,
        accuracy_trials=1,
        warmup=2,
        repeats=2,
        rounds=3,
    )
    config = official.TransformerConfig(
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        d_model=case.d_model,
        num_heads=case.num_heads,
        ffn_dim=case.ffn_dim,
        num_layers=case.num_layers,
        causal=case.causal,
    )
    fixed_inputs = torch.zeros(1, 2, 4)
    fixed_mask = torch.ones(1, 2, dtype=torch.bool)
    events: list[tuple[str, str, int]] = []
    generated_arguments: dict[str, Any] = {}

    class MarkerModel(nn.Module):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

    baseline = MarkerModel("baseline")
    solution = MarkerModel("solution")
    batches = {
        "baseline": iter(([1.0, 100.0], [2.0, 3.0], [4.0, 5.0])),
        "solution": iter(([1.0, 9.0], [2.0, 8.0], [3.0, 7.0])),
    }

    def fake_generate_random_case(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        generated_arguments.update(kwargs)
        return fixed_inputs, fixed_mask

    def fake_warmup(
        model: MarkerModel,
        inputs: torch.Tensor,
        valid_mask: torch.Tensor,
        iterations: int,
        device: torch.device,
    ) -> None:
        assert inputs is fixed_inputs
        assert valid_mask is fixed_mask
        assert device.type == "cpu"
        events.append(("warmup", model.name, iterations))

    def fake_benchmark_once(
        model: MarkerModel,
        inputs: torch.Tensor,
        valid_mask: torch.Tensor,
        iterations: int,
        device: torch.device,
    ) -> list[float]:
        assert inputs is fixed_inputs
        assert valid_mask is fixed_mask
        assert device.type == "cpu"
        events.append(("timing", model.name, iterations))
        return list(next(batches[model.name]))

    monkeypatch.setattr(official, "generate_random_case", fake_generate_random_case)
    monkeypatch.setattr(official, "warmup_model", fake_warmup)
    monkeypatch.setattr(official, "benchmark_once", fake_benchmark_once)

    performance = run_performance(
        baseline,
        solution,
        config,
        case,
        protocol,
        torch.device("cpu"),
        torch.float32,
    )

    assert generated_arguments["seed"] == 100_017
    assert generated_arguments["input_scale"] == 1.5
    assert events == [
        ("warmup", "baseline", 2),
        ("warmup", "solution", 2),
        ("timing", "baseline", 2),
        ("timing", "solution", 2),
        ("timing", "solution", 2),
        ("timing", "baseline", 2),
        ("timing", "baseline", 2),
        ("timing", "solution", 2),
    ]
    assert performance["baseline"]["median_ms"] == 3.5
    assert performance["baseline"]["p90_ms"] == pytest.approx(52.5)
    assert performance["baseline"]["round_medians_ms"] == [50.5, 2.5, 4.5]
    assert performance["baseline"]["sample_count"] == 6
    assert performance["target"]["median_ms"] == 5.0
    assert performance["target"]["p90_ms"] == pytest.approx(8.5)
    assert performance["target"]["round_medians_ms"] == [5.0, 5.0, 5.0]
    assert performance["target"]["sample_count"] == 6
    assert performance["speedup"] == pytest.approx(0.7)
    assert "samples_ms" not in str(performance)
    assert "solution" not in performance
