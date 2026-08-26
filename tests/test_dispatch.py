"""Tests for offline calibration promotion and deterministic dispatch."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from runner.contracts import ContractError, solution_implementation_hash
from runner.route_promotion import (
    build_promoted_route_document,
    promote_tuning_summary,
    select_deployable_winner,
)
from solution.dispatch import (
    OfflineDispatcher,
    make_route_key,
    resolve_route,
    validate_route_table,
)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=1,
        seq_len=2048,
        d_model=512,
        num_heads=8,
        ffn_dim=2048,
        num_layers=4,
        causal=False,
    )


def _formal_summary() -> dict[str, object]:
    return {
        "schema_version": 1,
        "tuning_id": "fixture-tuning",
        "complete": True,
        "protocol": {
            "preset": "formal",
            "seed": 1234,
            "accuracy_trials": 5,
            "rtol": 0.01,
            "atol": 0.001,
            "warmup": 20,
            "repeats": 100,
            "rounds": 3,
            "matmul_precision": "high",
            "allow_tf32": True,
        },
        "source_consistent": True,
        "source_solution_sha256": "fixture-solution-hash",
        "implementation_consistent": True,
        "source_implementation_sha256": "fixture-implementation-hash",
        "device_profile": {
            "device_type": "cuda",
            "device_name": "Fixture GPU",
            "compute_capability": "8.9",
        },
        "workload": {
            "case": {
                "case_id": "attention_fixture",
                "batch_size": 1,
                "seq_len": 2048,
                "d_model": 512,
                "num_heads": 8,
                "ffn_dim": 2048,
                "num_layers": 4,
                "dtype": "float16",
                "causal": False,
                "padding_ratio": 0.0,
                "input_scale": 1.0,
            }
        },
        "observations": [
            {
                "candidate_id": "compile-fast",
                "solution_policy": "auto",
                "compile_solution": True,
                "cuda_graph_solution": False,
                "outcome": "success",
                "correctness_passed": True,
                "failed_elements": 0,
                "policy_applied": True,
                "conservative_speedup": 3.0,
                "baseline_round_medians_ms": [3.0, 3.0, 3.0],
                "target_round_medians_ms": [1.0, 1.0, 1.0],
                "target_median_ms": 0.9,
                "target_p90_ms": 1.0,
                "solution_sha256": "fixture-solution-hash",
                "execution_path": {"shape_route": "compile-control"},
            },
            {
                "candidate_id": "eager-auto",
                "solution_policy": "auto",
                "compile_solution": False,
                "cuda_graph_solution": False,
                "outcome": "success",
                "correctness_passed": True,
                "failed_elements": 0,
                "policy_applied": True,
                "conservative_speedup": 1.4,
                "baseline_round_medians_ms": [1.4, 1.4, 1.4],
                "target_round_medians_ms": [1.0, 1.0, 1.0],
                "target_median_ms": 6.5,
                "target_p90_ms": 6.8,
                "solution_sha256": "fixture-solution-hash",
                "execution_path": {"shape_route": "safe-auto"},
            },
            {
                "candidate_id": "long-pv",
                "solution_policy": "long-pv",
                "compile_solution": False,
                "cuda_graph_solution": False,
                "outcome": "success",
                "correctness_passed": True,
                "failed_elements": 0,
                "policy_applied": True,
                "conservative_speedup": 1.5,
                "baseline_round_medians_ms": [1.5, 1.5, 1.5],
                "target_round_medians_ms": [1.0, 1.0, 1.0],
                "target_median_ms": 6.1,
                "target_p90_ms": 6.4,
                "solution_sha256": "fixture-solution-hash",
                "execution_path": {"shape_route": "long-pv"},
            },
        ],
    }


def _candidate_observation(
    *,
    candidate_id: str,
    policy: str,
    speedup: float,
) -> dict[str, object]:
    observation = copy.deepcopy(_formal_summary()["observations"][2])  # type: ignore[index]
    observation["candidate_id"] = candidate_id
    observation["solution_policy"] = policy
    observation["conservative_speedup"] = speedup
    observation["baseline_round_medians_ms"] = [speedup, speedup, speedup]
    observation["execution_path"] = {"shape_route": candidate_id}
    return observation


def _existing_exact_route(policy: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "default_policy": "auto",
        "routes": [
            {
                "match": {
                    "dtype": "float16",
                    "B": 1,
                    "S": 2048,
                    "D": 512,
                    "heads": 8,
                    "ffn": 2048,
                    "layers": 4,
                    "causal": False,
                    "device_type": "cuda",
                    "device_name": "Fixture GPU",
                    "compute_capability": "8.9",
                },
                "policy": policy,
            }
        ],
    }


def test_route_table_resolves_an_exact_static_subset() -> None:
    table = validate_route_table(
        {
            "schema_version": 1,
            "default_policy": "auto",
            "routes": [
                {
                    "match": {
                        "device_type": "cuda",
                        "dtype": "float16",
                        "S": 2048,
                        "causal": False,
                    },
                    "policy": "long-pv",
                }
            ],
        }
    )
    matching = make_route_key(
        _config(),
        dtype=torch.float16,
        device_type="cuda",
    )
    fallback = make_route_key(
        _config(),
        dtype=torch.float32,
        device_type="cuda",
    )

    assert resolve_route(table, matching) == "long-pv"
    assert resolve_route(table, fallback) == "auto"


def test_dispatcher_loads_once_and_never_reads_mask_content(tmp_path) -> None:
    route_path = tmp_path / "dispatch_routes.json"
    route_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "default_policy": "auto",
                "routes": [
                    {
                        "match": {"device_type": "cpu", "S": 2048},
                        "policy": "reference",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dispatcher = OfflineDispatcher(route_path)
    route_path.write_text("not valid json", encoding="utf-8")

    assert (
        dispatcher.resolve(
            _config(),
            device="cpu",
            dtype=torch.float32,
            shape=(1, 2048, 512),
        )
        == "reference"
    )


def test_invalid_route_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown fields"):
        validate_route_table(
            {
                "schema_version": 1,
                "default_policy": "auto",
                "routes": [
                    {
                        "match": {"padding_ratio": 0.5},
                        "policy": "packed",
                    }
                ],
            }
        )


def test_formal_promotion_excludes_compile_and_writes_only_static_match() -> None:
    summary = _formal_summary()
    winner = select_deployable_winner(summary)
    document, promoted = build_promoted_route_document(None, summary)

    assert winner["candidate_id"] == "long-pv"
    assert promoted == winner
    route = document["routes"][0]
    assert route["policy"] == "long-pv"
    assert route["match"]["device_name"] == "Fixture GPU"
    assert route["match"]["B"] == 1
    assert "case_id" not in route["match"]
    assert "padding_ratio" not in route["match"]


def test_smoke_summary_cannot_change_the_dispatch_table() -> None:
    summary = _formal_summary()
    summary["protocol"] = {"preset": "smoke"}

    with pytest.raises(ContractError, match="formal"):
        build_promoted_route_document(None, summary)


def test_promotion_requires_paired_round_ranking() -> None:
    summary = _formal_summary()
    del summary["observations"][2]["conservative_speedup"]  # type: ignore[index]

    winner = select_deployable_winner(summary)

    assert winner["candidate_id"] == "eager-auto"


def test_promotion_rejects_inconsistent_observation_source() -> None:
    summary = _formal_summary()
    summary["observations"][2]["solution_sha256"] = "stale"  # type: ignore[index]

    with pytest.raises(ContractError, match="source hashes"):
        build_promoted_route_document(None, summary)


def test_promotion_rejects_a_stale_current_solution(tmp_path) -> None:
    solution_root = tmp_path / "solution"
    solution_root.mkdir()
    (solution_root / "transformer.py").write_text("VALUE = 1\n", encoding="utf-8")
    (solution_root / "dispatch_routes.json").write_text(
        json.dumps(
            {"schema_version": 1, "default_policy": "auto", "routes": []}
        ),
        encoding="utf-8",
    )
    summary = _formal_summary()
    summary["source_implementation_sha256"] = solution_implementation_hash(
        solution_root
    )
    for observation in summary["observations"]:  # type: ignore[union-attr]
        observation["solution_sha256"] = summary["source_solution_sha256"]
    (solution_root / "transformer.py").write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(ContractError, match="does not match"):
        promote_tuning_summary(tmp_path, summary)


def test_promotion_rejects_an_incomplete_tuning_summary() -> None:
    summary = _formal_summary()
    summary["complete"] = False

    with pytest.raises(ContractError, match="complete"):
        build_promoted_route_document(None, summary)


def test_specialized_route_requires_a_meaningful_gain_over_auto() -> None:
    summary = _formal_summary()
    specialized = summary["observations"][2]  # type: ignore[index]
    specialized["conservative_speedup"] = 1.41
    specialized["baseline_round_medians_ms"] = [1.41, 1.41, 1.41]

    with pytest.raises(ContractError, match="promotion margin"):
        build_promoted_route_document(None, summary)


def test_replacing_an_exact_route_requires_a_gain_over_the_incumbent() -> None:
    summary = _formal_summary()
    observations = summary["observations"]
    assert isinstance(observations, list)
    observations.append(
        _candidate_observation(
            candidate_id="incumbent-preprocess",
            policy="preprocess",
            speedup=1.49,
        )
    )

    with pytest.raises(ContractError, match="incumbent.*promotion margin"):
        build_promoted_route_document(
            _existing_exact_route("preprocess"),
            summary,
        )


def test_replacing_an_exact_route_requires_an_incumbent_observation() -> None:
    with pytest.raises(ContractError, match="missing a correct incumbent observation"):
        build_promoted_route_document(
            _existing_exact_route("preprocess"),
            _formal_summary(),
        )


def test_a_clear_gain_replaces_only_the_matching_incumbent_route() -> None:
    summary = _formal_summary()
    observations = summary["observations"]
    assert isinstance(observations, list)
    observations.append(
        _candidate_observation(
            candidate_id="incumbent-preprocess",
            policy="preprocess",
            speedup=1.45,
        )
    )
    existing = _existing_exact_route("preprocess")
    routes = existing["routes"]
    assert isinstance(routes, list)
    unrelated = {"match": {"B": 8}, "policy": "reference"}
    routes.append(unrelated)

    document, winner = build_promoted_route_document(existing, summary)

    assert winner["solution_policy"] == "long-pv"
    assert unrelated in document["routes"]
    assert {
        "match": _existing_exact_route("long-pv")["routes"][0]["match"],  # type: ignore[index]
        "policy": "long-pv",
    } in document["routes"]


def test_an_incumbent_winner_keeps_its_route_without_a_new_margin() -> None:
    summary = _formal_summary()
    incumbent = summary["observations"][2]  # type: ignore[index]
    incumbent["conservative_speedup"] = 1.41
    incumbent["baseline_round_medians_ms"] = [1.41, 1.41, 1.41]
    existing = _existing_exact_route("long-pv")

    document, winner = build_promoted_route_document(existing, summary)

    assert winner["solution_policy"] == "long-pv"
    assert document == existing


def test_promoted_exact_route_precedes_a_broad_fallback() -> None:
    summary = _formal_summary()
    observations = summary["observations"]
    assert isinstance(observations, list)
    observations.append(
        _candidate_observation(
            candidate_id="incumbent-reference",
            policy="reference",
            speedup=1.45,
        )
    )
    existing = {
        "schema_version": 1,
        "default_policy": "auto",
        "routes": [{"match": {"B": 1}, "policy": "reference"}],
    }

    document, _ = build_promoted_route_document(existing, summary)
    table = validate_route_table(document)
    key = make_route_key(
        _config(),
        dtype=torch.float16,
        device_type="cuda",
        device_name="Fixture GPU",
        compute_capability="8.9",
    )

    assert resolve_route(table, key) == "long-pv"


def test_a_broad_incumbent_requires_a_formal_observation() -> None:
    existing = {
        "schema_version": 1,
        "default_policy": "auto",
        "routes": [{"match": {"B": 1}, "policy": "preprocess"}],
    }

    with pytest.raises(ContractError, match="missing a correct incumbent observation"):
        build_promoted_route_document(existing, _formal_summary())


def test_a_broad_incumbent_is_protected_by_the_promotion_margin() -> None:
    summary = _formal_summary()
    observations = summary["observations"]
    assert isinstance(observations, list)
    observations.append(
        _candidate_observation(
            candidate_id="incumbent-preprocess",
            policy="preprocess",
            speedup=1.49,
        )
    )
    existing = {
        "schema_version": 1,
        "default_policy": "auto",
        "routes": [{"match": {"B": 1}, "policy": "preprocess"}],
    }

    with pytest.raises(ContractError, match="incumbent.*promotion margin"):
        build_promoted_route_document(existing, summary)


def test_exact_auto_can_override_a_broad_specialized_route() -> None:
    summary = _formal_summary()
    observations = summary["observations"]
    assert isinstance(observations, list)
    auto = observations[1]
    assert isinstance(auto, dict)
    auto["conservative_speedup"] = 1.6
    auto["baseline_round_medians_ms"] = [1.6, 1.6, 1.6]
    observations.append(
        _candidate_observation(
            candidate_id="incumbent-preprocess",
            policy="preprocess",
            speedup=1.5,
        )
    )
    existing = {
        "schema_version": 1,
        "default_policy": "auto",
        "routes": [{"match": {"B": 1}, "policy": "preprocess"}],
    }

    document, winner = build_promoted_route_document(existing, summary)
    table = validate_route_table(document)
    key = make_route_key(
        _config(),
        dtype=torch.float16,
        device_type="cuda",
        device_name="Fixture GPU",
        compute_capability="8.9",
    )

    assert winner["solution_policy"] == "auto"
    assert resolve_route(table, key) == "auto"
    assert document["routes"][0]["policy"] == "auto"


def test_promotion_preserves_unrelated_overlapping_route_order() -> None:
    summary = _formal_summary()
    observations = summary["observations"]
    assert isinstance(observations, list)
    observations.append(
        _candidate_observation(
            candidate_id="incumbent-reference",
            policy="reference",
            speedup=1.45,
        )
    )
    existing = {
        "schema_version": 1,
        "default_policy": "auto",
        "routes": [
            {"match": {"dtype": "float16"}, "policy": "reference"},
            {
                "match": {"B": 2, "dtype": "float16"},
                "policy": "torch",
            },
        ],
    }
    unrelated_key = {
        "B": 2,
        "dtype": "float16",
    }
    before = resolve_route(validate_route_table(existing), unrelated_key)

    document, _ = build_promoted_route_document(existing, summary)
    after = resolve_route(validate_route_table(document), unrelated_key)

    assert before == "reference"
    assert after == before
    assert document["routes"][1:] == existing["routes"]


def test_a_matching_broad_winner_does_not_add_a_redundant_exact_route() -> None:
    existing = {
        "schema_version": 1,
        "default_policy": "auto",
        "routes": [{"match": {"B": 1}, "policy": "long-pv"}],
    }

    document, winner = build_promoted_route_document(existing, _formal_summary())

    assert winner["solution_policy"] == "long-pv"
    assert document == existing
