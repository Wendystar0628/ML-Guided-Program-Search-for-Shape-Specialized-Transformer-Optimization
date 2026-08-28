"""Tests for execution-path guards and timing aggregation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from official import torch_transformer_benchmark as official
from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    RunVariant,
    TransformerShape,
)
from runner.execution import (
    PreparedExecution,
    _validate_cuda_graph_composition,
    _validate_profile_execution_path,
    execute_benchmark,
    execute_profile,
    measure_solution_peak_allocated_bytes,
    prepare_execution,
    run_performance,
)
from runner.result_contracts import WorkerRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _tiny_shape() -> TransformerShape:
    return TransformerShape(
        case_id="execution_fixture",
        batch_size=1,
        seq_len=2,
        d_model=4,
        num_heads=1,
        ffn_dim=4,
        num_layers=1,
        causal=True,
    )


def _request(run_kind: str, protocol: MeasurementProtocol) -> WorkerRequest:
    return WorkerRequest(
        run_kind=run_kind,  # type: ignore[arg-type]
        project_root=PROJECT_ROOT,
        shape=_tiny_shape(),
        variant=RunVariant(),
        protocol=protocol,
        device="cpu",
        target="baseline",
        comparison_mode="baseline_only" if run_kind == "benchmark" else None,
    )


def test_solution_graph_rejects_torch_compile() -> None:
    with pytest.raises(ContractError, match="graph policy"):
        _validate_cuda_graph_composition(
            {"runtime_wrapper": "cuda_graph"},
            MeasurementProtocol(preset="smoke", compile_solution=True),
        )


def test_operator_profile_rejects_the_solution_graph_wrapper() -> None:
    with pytest.raises(ContractError, match="hides per-operator"):
        _validate_profile_execution_path({"runtime_wrapper": "cuda_graph"})


def test_solution_peak_measurement_is_cuda_only() -> None:
    config = official.TransformerConfig(
        batch_size=1,
        seq_len=2,
        d_model=4,
        num_heads=1,
        ffn_dim=4,
        num_layers=1,
        causal=True,
    )
    baseline = nn.Identity()
    solution = nn.Identity()

    assert (
        measure_solution_peak_allocated_bytes(
            baseline,
            solution,
            config,
            RunVariant(),
            MeasurementProtocol(preset="smoke"),
            torch.device("cpu"),
            torch.float32,
        )
        is None
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
    assert prepared.execution_path == {
        "requested_policy": "official-baseline",
        "selected_policy": "official-baseline",
        "qkv_projection": "separate",
        "attention_backend": "official_explicit",
        "runtime_wrapper": "eager",
        "residual_norm_backend": "torch",
        "causal_mask": "per_forward",
        "valid_token_mask": "direct_key_mask",
        "fallback_reasons": [],
        "execution_mode": "eager",
    }
    with pytest.raises(FrozenInstanceError):
        prepared.device = torch.device("cuda")  # type: ignore[misc]


def test_eager_solution_observation_preserves_execution_mode() -> None:
    response = execute_benchmark(
        WorkerRequest(
            run_kind="benchmark",
            project_root=PROJECT_ROOT,
            shape=_tiny_shape(),
            variant=RunVariant(),
            protocol=MeasurementProtocol(
                preset="smoke",
                accuracy_trials=1,
                warmup=0,
                repeats=1,
                rounds=1,
            ),
            device="cpu",
            target="solution",
            comparison_mode="paired",
            solution_policy="safe",
        )
    )

    assert response["outcome"] == "success"
    assert response["execution_path"]["execution_mode"] == "eager"
    assert response["execution_path"]["observed_execution"]["complete"] is True


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
    shape = TransformerShape(
        case_id="timing_fixture",
        batch_size=1,
        seq_len=2,
        d_model=4,
        num_heads=1,
        ffn_dim=4,
        num_layers=1,
        causal=True,
    )
    variant = RunVariant(input_scale=1.5)
    protocol = MeasurementProtocol(
        preset="smoke",
        seed=17,
        accuracy_trials=1,
        warmup=2,
        repeats=2,
        rounds=3,
    )
    config = official.TransformerConfig(
        batch_size=shape.batch_size,
        seq_len=shape.seq_len,
        d_model=shape.d_model,
        num_heads=shape.num_heads,
        ffn_dim=shape.ffn_dim,
        num_layers=shape.num_layers,
        causal=shape.causal,
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
        variant,
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
    assert performance["baseline"]["sample_count"] == 6
    assert performance["target"]["median_ms"] == 5.0
    assert performance["target"]["p90_ms"] == pytest.approx(8.5)
    assert performance["target"]["sample_count"] == 6
    assert performance["speedup"] == pytest.approx(0.7)
    assert "samples_ms" not in str(performance)
    assert "solution" not in performance


def test_compile_mode_is_reported_separately_from_runtime_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        official,
        "maybe_compile",
        lambda model, _enabled, _mode: model,
    )

    prepared = prepare_execution(
        _request(
            "benchmark",
            MeasurementProtocol(
                preset="smoke",
                accuracy_trials=1,
                warmup=0,
                repeats=1,
                rounds=1,
                compile_baseline=True,
            ),
        )
    )

    assert prepared.execution_path["runtime_wrapper"] == "eager"
    assert prepared.execution_path["execution_mode"] == "torch_compile"


def test_execution_plan_failure_reports_its_own_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_plan(_path: dict[str, Any]) -> None:
        raise ContractError("fixture plan failure")

    monkeypatch.setattr(
        "runner.execution._validate_profile_execution_path",
        reject_plan,
    )

    response = execute_profile(
        _request(
            "profile",
            MeasurementProtocol(
                preset="smoke",
                accuracy_trials=1,
                warmup=0,
                repeats=1,
                rounds=1,
            ),
        )
    )

    assert response["outcome"] == "runtime_error"
    assert response["failure"]["stage"] == "execution_plan"
