"""Focused tests for the bounded candidate-screening loop."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from runner import tuning
from runner.contracts import ContractError, MeasurementProtocol, WorkloadCase


def _case(
    *,
    case_id: str = "fixture",
    padding_ratio: float = 0.0,
    wide: bool = False,
) -> WorkloadCase:
    return WorkloadCase(
        case_id=case_id,
        batch_size=2,
        seq_len=32,
        d_model=1024 if wide else 64,
        num_heads=8,
        ffn_dim=4096 if wide else 128,
        num_layers=1,
        dtype="bfloat16" if wide else "float16",
        causal=False,
        padding_ratio=padding_ratio,
    )


def test_candidates_add_only_relevant_specialized_routes() -> None:
    common = {item.candidate_id for item in tuning.candidates_for_case(_case())}
    padded = {
        item.candidate_id
        for item in tuning.candidates_for_case(_case(padding_ratio=0.5))
    }
    full_mask = {
        item.candidate_id
        for item in tuning.candidates_for_case(_case(case_id="mask_s512_full_fp16"))
    }
    wide = {item.candidate_id for item in tuning.candidates_for_case(_case(wide=True))}
    launch = {
        item.candidate_id
        for item in tuning.candidates_for_case(
            WorkloadCase(
                case_id="launch",
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
        )
    }
    long_attention = {
        item.candidate_id
        for item in tuning.candidates_for_case(
            WorkloadCase(
                case_id="long",
                batch_size=1,
                seq_len=2048,
                d_model=512,
                num_heads=8,
                ffn_dim=2048,
                num_layers=4,
                dtype="float16",
                causal=False,
                padding_ratio=0.0,
            )
        )
    }
    exact_wide = {
        item.candidate_id
        for item in tuning.candidates_for_case(
            WorkloadCase(
                case_id="wide",
                batch_size=16,
                seq_len=256,
                d_model=1024,
                num_heads=8,
                ffn_dim=4096,
                num_layers=6,
                dtype="bfloat16",
                causal=False,
                padding_ratio=0.0,
            )
        )
    }

    assert "padding-packed" not in common
    assert "padding-fused" not in common
    assert "compile-max-autotune" not in common
    assert "padding-packed" in padded
    assert "padding-fused" in padded
    assert "padding-packed" in full_mask
    assert "padding-fused" in full_mask
    assert "compile-max-autotune" in wide
    assert "attention-preprocess" in launch
    assert "padding-fused" in launch
    assert "eager-cudagraph" in launch
    assert "launch-cudagraph" in launch
    assert "long-pv" not in long_attention
    assert "long-tail-online" in long_attention
    assert "attention-preprocess" in long_attention
    assert "wide-gelu-epilogue" not in wide
    assert "wide-gelu-epilogue" not in exact_wide
    assert "wide-triton-inplace" in exact_wide


def test_solution_policy_restores_the_parent_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(tuning.SOLUTION_POLICY_ENV, "previous")

    with tuning.solution_policy("triton"):
        assert os.environ[tuning.SOLUTION_POLICY_ENV] == "triton"

    assert os.environ[tuning.SOLUTION_POLICY_ENV] == "previous"
    with (
        pytest.raises(ContractError, match="unsupported solution policy"),
        tuning.solution_policy("unknown"),
    ):
        pass

    monkeypatch.delenv(tuning.SOLUTION_POLICY_ENV, raising=False)
    with (
        pytest.raises(RuntimeError, match="fixture failure"),
        tuning.solution_policy("packed"),
    ):
        assert os.environ[tuning.SOLUTION_POLICY_ENV] == "packed"
        raise RuntimeError("fixture failure")
    assert tuning.SOLUTION_POLICY_ENV not in os.environ


def test_tuning_case_runs_serial_candidates_and_selects_correct_winner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run_managed_benchmark(
        project_root: Path,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Path]:
        del project_root
        protocol = kwargs["protocol"]
        policy = os.environ[tuning.SOLUTION_POLICY_ENV]
        calls.append(
            {
                "policy": policy,
                "compile_solution": protocol.compile_solution,
                "compile_mode": protocol.compile_mode,
                "sweep_id": kwargs["sweep_id"],
            }
        )
        speedup = 1.25 if protocol.compile_solution else 1.05
        result = {
            "outcome": "success",
            "correctness": {"passed": True},
            "performance": {
                "baseline": {"median_ms": 2.0},
                "target": {"median_ms": 2.0 / speedup, "p90_ms": 1.8},
                "speedup": speedup,
            },
            "execution_path": {
                "requested_policy": policy,
                "selected_policy": policy,
            },
            "source": {"solution_sha256": "fixture-solution-hash"},
        }
        return result, tmp_path / f"{policy}-{protocol.compile_mode}.json"

    monkeypatch.setattr(tuning, "run_managed_benchmark", fake_run_managed_benchmark)
    summary = tuning.run_tuning_case(
        tmp_path,
        workload_set_id="fixture",
        workload_sha256="fixture-hash",
        case=_case(),
        base_protocol=MeasurementProtocol.for_preset("smoke"),
        device="cuda:0",
        requested_candidates=("eager-torch", "compile-default"),
    )

    assert calls == [
        {
            "policy": "torch",
            "compile_solution": False,
            "compile_mode": "default",
            "sweep_id": calls[0]["sweep_id"],
        },
        {
            "policy": "auto",
            "compile_solution": True,
            "compile_mode": "default",
            "sweep_id": calls[0]["sweep_id"],
        },
    ]
    assert summary["tuning_id"] == calls[0]["sweep_id"]
    assert summary["winner"]["candidate_id"] == "compile-default"
    assert len(summary["observations"]) == 2
    summary_path = Path(summary["summary_path"])
    assert summary_path.is_file()
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary


def test_select_candidates_rejects_a_route_that_does_not_fit_the_case() -> None:
    with pytest.raises(ContractError, match="not available"):
        tuning.select_candidates(_case(), ("padding-packed",))


def test_fallback_candidate_is_reported_but_not_ranked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_managed_benchmark(
        project_root: Path,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Path]:
        del project_root, kwargs
        policy = os.environ[tuning.SOLUTION_POLICY_ENV]
        selected = "torch_fallback" if policy == "triton" else policy
        speedup = 2.0 if policy == "triton" else 1.1
        return (
            {
                "outcome": "success",
                "correctness": {"passed": True},
                "performance": {
                    "baseline": {"median_ms": 2.0},
                    "target": {"median_ms": 2.0 / speedup, "p90_ms": 2.0},
                    "speedup": speedup,
                },
                    "execution_path": {
                    "requested_policy": policy,
                    "selected_policy": selected,
                    "resolved_qkv_layout": (
                        "torch_three_contiguous_copies"
                        if policy == "torch"
                        else "view_fallback"
                        ),
                    },
                    "source": {"solution_sha256": "fixture-solution-hash"},
            },
            tmp_path / f"{policy}.json",
        )

    monkeypatch.setattr(tuning, "run_managed_benchmark", fake_run_managed_benchmark)
    summary = tuning.run_tuning_case(
        tmp_path,
        workload_set_id="fixture",
        workload_sha256="fixture-hash",
        case=_case(),
        base_protocol=MeasurementProtocol.for_preset("smoke"),
        device="cuda:0",
        requested_candidates=("eager-triton", "eager-torch"),
    )

    assert summary["observations"][0]["policy_applied"] is False
    assert summary["winner"]["candidate_id"] == "eager-torch"


def test_padding_candidate_requires_the_triton_fusion_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_managed_benchmark(
        project_root: Path,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Path]:
        del project_root, kwargs
        policy = os.environ[tuning.SOLUTION_POLICY_ENV]
        return (
            {
                "outcome": "success",
                "correctness": {"passed": True},
                "performance": {
                    "baseline": {"median_ms": 2.0},
                    "target": {"median_ms": 1.0, "p90_ms": 1.1},
                    "speedup": 2.0,
                },
                "execution_path": {
                    "requested_policy": policy,
                    "selected_policy": policy,
                    "block_fusion": "torch_residual_fallback",
                },
                "source": {"solution_sha256": "fixture-solution-hash"},
            },
            tmp_path / "padding-fallback.json",
        )

    monkeypatch.setattr(tuning, "run_managed_benchmark", fake_run_managed_benchmark)
    summary = tuning.run_tuning_case(
        tmp_path,
        workload_set_id="fixture",
        workload_sha256="fixture-hash",
        case=_case(padding_ratio=0.5),
        base_protocol=MeasurementProtocol.for_preset("smoke"),
        device="cuda:0",
        requested_candidates=("padding-fused",),
    )

    assert summary["observations"][0]["policy_applied"] is False
    assert summary["winner"] is None
