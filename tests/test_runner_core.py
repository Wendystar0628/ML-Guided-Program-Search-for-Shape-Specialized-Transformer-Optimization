"""High-value regression tests for the performance development loop."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from official import torch_transformer_benchmark as official
from runner.contracts import (
    MeasurementProtocol,
    WorkloadCase,
    load_workload_set,
    validate_official_snapshot,
)
from runner.execution import run_performance
from runner.supervisor import run_managed_benchmark

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_OFFICIAL_SHA256 = (
    "1630fe39ebc845beeaef73aaaf2d47e061fc56fd20777706c3ddc961664c266b"
)
WORKLOAD_SET_ID = "provisional_reference_v1"


def _tiny_case(*, causal: bool = True, padding_ratio: float = 0.5) -> WorkloadCase:
    return WorkloadCase(
        case_id="tiny_cpu_smoke",
        batch_size=1,
        seq_len=2,
        d_model=4,
        num_heads=1,
        ffn_dim=8,
        num_layers=1,
        dtype="float32",
        causal=causal,
        padding_ratio=padding_ratio,
    )


def _tiny_protocol() -> MeasurementProtocol:
    return MeasurementProtocol(
        preset="smoke",
        seed=17,
        accuracy_trials=1,
        warmup=0,
        repeats=1,
        rounds=1,
        matmul_precision="highest",
        allow_tf32=False,
        timeout_seconds=60.0,
    )


def _copy_runtime_project(tmp_path: Path) -> Path:
    project = tmp_path / "managed-project"
    project.mkdir()
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    for directory in ("official", "runner", "solution"):
        shutil.copytree(
            PROJECT_ROOT / directory,
            project / directory,
            ignore=ignored,
        )
    return project


def test_official_snapshot_and_workload_are_usable() -> None:
    metadata = validate_official_snapshot(PROJECT_ROOT)
    snapshot_path = PROJECT_ROOT / metadata["snapshot_path"]

    assert metadata["sha256"] == EXPECTED_OFFICIAL_SHA256
    assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == (
        EXPECTED_OFFICIAL_SHA256
    )

    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    cases = workload["cases"]
    assert len(cases) == 4
    assert len({case.case_id for case in cases}) == 4
    assert {(case.causal, case.padding_ratio > 0.0) for case in cases} == {
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    }


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
    assert performance["baseline"]["samples_ms"] == [
        1.0,
        100.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]
    assert performance["solution"]["samples_ms"] == [
        1.0,
        9.0,
        2.0,
        8.0,
        3.0,
        7.0,
    ]
    assert performance["baseline"]["median_ms"] == 3.5
    assert performance["solution"]["median_ms"] == 5.0
    assert performance["speedup"] == pytest.approx(0.7)


def test_managed_cpu_solution_smoke_persists_result(tmp_path: Path) -> None:
    project = _copy_runtime_project(tmp_path)
    protocol = _tiny_protocol()

    result, result_path = run_managed_benchmark(
        project,
        workload_set_id="tiny_test_fixture",
        case=_tiny_case(),
        protocol=protocol,
        device="cpu",
    )

    assert result["status"] == "success"
    assert result["correctness"]["passed"] is True
    assert result["failure"] is None
    performance = result["performance"]
    assert len(performance["baseline"]["samples_ms"]) == 1
    assert len(performance["solution"]["samples_ms"]) == 1
    assert performance["baseline"]["median_ms"] > 0
    assert performance["solution"]["median_ms"] > 0
    assert math.isfinite(performance["speedup"])
    assert performance["speedup"] > 0

    assert result_path.parent == project / "results" / "runs"
    assert json.loads(result_path.read_text(encoding="utf-8")) == result
