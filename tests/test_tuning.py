"""Focused tests for official-shape candidate tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from project_identity import solution_implementation_hash
from runner import tuning
from runner.contracts import ContractError, RunVariant
from tests.support.runner_fixtures import official_shape, tiny_protocol


def _candidate_ids(case_id: str) -> list[str]:
    return [
        candidate.candidate_id
        for candidate in tuning.candidates_for_shape(
            official_shape(case_id),
            RunVariant(),
        )
    ]


def _execution_path(candidate_id: str) -> dict[str, Any]:
    paths = {
        "eager-auto": {
            "requested_policy": "auto",
            "selected_policy": "auto",
        },
        "eager-safe": {
            "requested_policy": "safe",
            "selected_policy": "safe",
            "attention_backend": "safe_streaming",
            "runtime_wrapper": "eager",
            "batch_strategy": "full",
            "block_backend": "torch",
        },
        "causal-sdpa": {
            "requested_policy": "causal-sdpa",
            "selected_policy": "causal-sdpa",
            "attention_backend": "causal_sdpa",
            "runtime_wrapper": "eager",
            "batch_strategy": "full",
            "block_backend": "torch",
        },
    }
    path = paths[candidate_id]
    if candidate_id == "causal-sdpa":
        path["observed_execution"] = {
            "complete": True,
            "attention_backends": ["causal_sdpa"],
            "block_backends": ["torch"],
        }
    return path


def test_candidates_are_small_and_specific_to_official_shape_families() -> None:
    launch = _candidate_ids("official_02")
    extreme_batch = _candidate_ids("official_06")
    long_sequence = _candidate_ids("official_13")

    assert launch == [
        "eager-auto",
        "eager-safe",
        "causal-sdpa",
        "graph",
        "inplace-block",
    ]
    assert "batch-tiled" in extreme_batch
    assert "graph" not in extreme_batch
    assert "causal-sdpa" in long_sequence
    assert "graph" not in long_sequence


def test_runtime_variant_is_not_hidden_inside_the_shape() -> None:
    shape = official_shape("official_02")

    assert tuning.candidates_for_shape(shape, RunVariant(padding_ratio=0.5)) == ()


def test_select_candidates_requires_explicit_valid_order() -> None:
    shape = official_shape("official_02")
    variant = RunVariant()

    selected = tuning.select_candidates(
        shape,
        variant,
        ["causal-sdpa", "eager-auto"],
    )
    assert [item.candidate_id for item in selected] == [
        "causal-sdpa",
        "eager-auto",
    ]
    with pytest.raises(ContractError, match="at least one"):
        tuning.select_candidates(shape, variant, [])
    with pytest.raises(ContractError, match="not available"):
        tuning.select_candidates(shape, variant, ["batch-tiled"])


def test_deployed_policy_maps_back_to_one_candidate() -> None:
    shape = official_shape("official_06")
    variant = RunVariant()

    assert (
        tuning.deployable_candidate_id_for_policy(shape, variant, "batch-tiled")
        == "batch-tiled"
    )
    assert tuning.deployable_candidate_id_for_policy(shape, variant, "graph") is None


def test_tuning_runs_serial_candidates_and_selects_the_measured_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solution_root = tmp_path / "solution"
    solution_root.mkdir()
    (solution_root / "transformer.py").write_text("VALUE = 1\n", encoding="utf-8")
    implementation_hash = solution_implementation_hash(solution_root)
    calls: list[str] = []

    def fake_run(
        _project_root: Path,
        *,
        candidate_id: str,
        solution_policy: str,
        result_dir: Path,
        **_kwargs: Any,
    ) -> tuple[dict[str, Any], Path]:
        calls.append(candidate_id)
        speedup = {"eager-auto": 1.0, "causal-sdpa": 1.4}[candidate_id]
        target = 2.0 / speedup
        result = {
            "run_id": f"run-{candidate_id}",
            "outcome": "success",
            "correctness": {
                "passed": True,
                "failed_elements": 0,
                "max_abs_error": 0.0,
            },
            "performance": {
                "baseline": {"median_ms": 2.0, "p90_ms": 2.0},
                "target": {"median_ms": target, "p90_ms": target},
                "speedup": speedup,
            },
            "source": {
                "solution_sha256": implementation_hash,
                "official_sha256": "fixture-official",
            },
            "execution_path": _execution_path(candidate_id),
        }
        return result, result_dir / f"{candidate_id}.json"

    monkeypatch.setattr(tuning, "run_managed_benchmark", fake_run)

    summary = tuning.run_tuning_case(
        tmp_path,
        workload_set_id="official_transformer_v1",
        workload_sha256="fixture-workload",
        shape=official_shape("official_02"),
        variant=RunVariant(),
        base_protocol=tiny_protocol(),
        device="cpu",
        requested_candidates=["eager-auto", "causal-sdpa"],
    )

    assert calls == ["eager-auto", "causal-sdpa"]
    assert summary["complete"] is True
    assert summary["winner"]["candidate_id"] == "causal-sdpa"
    assert summary["deployable_winner"]["candidate_id"] == "causal-sdpa"
    assert Path(summary["summary_path"]).exists()


def test_candidate_fallback_is_observed_but_cannot_win() -> None:
    candidate = next(
        item
        for item in tuning.candidates_for_shape(
            official_shape("official_02"), RunVariant()
        )
        if item.candidate_id == "causal-sdpa"
    )
    result = {
        "run_id": "fallback",
        "outcome": "success",
        "correctness": {"passed": True, "failed_elements": 0},
        "performance": {
            "baseline": {"median_ms": 2.0},
            "target": {"median_ms": 1.0, "p90_ms": 1.0},
            "speedup": 2.0,
        },
        "source": {
            "solution_sha256": "fixture-solution",
            "official_sha256": "fixture-official",
        },
        "execution_path": {
            "requested_policy": "causal-sdpa",
            "selected_policy": "safe",
        },
    }

    observation = tuning._observation(candidate, result, Path("fallback.json"))

    assert observation["policy_applied"] is False


def test_graph_candidate_requires_replay_and_underlying_backend_evidence() -> None:
    candidate = tuning.candidate_spec("graph")
    assert candidate is not None
    path = {
        "requested_policy": "graph",
        "selected_policy": "graph",
        "attention_backend": "causal_sdpa",
        "runtime_wrapper": "cuda_graph",
        "batch_strategy": "full",
        "block_backend": "torch",
        "observed_execution": {
            "attention_backends": ["causal_sdpa"],
            "block_backends": ["torch"],
            "runtime_wrappers": ["cuda_graph"],
        },
    }

    assert candidate.evidence_matches(path)
    del path["observed_execution"]["runtime_wrappers"]
    assert not candidate.evidence_matches(path)
