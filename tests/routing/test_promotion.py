"""Formal measurement promotion and exact incumbent tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from runner.contracts import ContractError
from runner.route_promotion import (
    build_promoted_route_document,
    promote_tuning_summaries,
    select_deployable_winner,
)
from solution.dispatch import ROUTE_FIELDS
from tests.support.routing_fixtures import (
    bind_workload_summaries,
    candidate_execution_path,
    candidate_observation,
    exact_match,
    exact_route_document,
    formal_summary,
    promotion_project,
    s512_summary,
)


def test_formal_promotion_ignores_compile_and_writes_one_exact_route() -> None:
    summary = formal_summary()

    winner = select_deployable_winner(summary)
    document, promoted = build_promoted_route_document(None, summary)

    assert winner["candidate_id"] == "long-tail-online"
    assert promoted == winner
    assert len(document["routes"]) == 1
    assert set(document["routes"][0]["match"]) == ROUTE_FIELDS
    assert document["routes"][0]["policy"] == "long-tail-online"
    assert "case_id" not in document["routes"][0]["match"]
    assert "padding_ratio" not in document["routes"][0]["match"]


def test_smoke_summary_cannot_be_promoted() -> None:
    summary = formal_summary()
    summary["protocol"] = {"preset": "smoke"}

    with pytest.raises(ContractError, match="formal"):
        build_promoted_route_document(None, summary)


def test_promotion_recomputes_candidate_execution_evidence() -> None:
    summary = formal_summary()
    challenger = summary["observations"][2]
    challenger["policy_applied"] = True
    challenger["execution_path"] = {
        "requested_policy": "long-tail-online",
        "selected_policy": "long-tail-online",
        "shape_route": "long-tail-online",
    }

    winner = select_deployable_winner(summary)

    assert winner["candidate_id"] == "eager-auto"


def test_promotion_rejects_candidate_policy_identity_mismatch() -> None:
    summary = formal_summary()
    summary["observations"][2]["candidate_id"] = "eager-reference"

    winner = select_deployable_winner(summary)

    assert winner["candidate_id"] == "eager-auto"


def test_specialized_candidate_below_margin_keeps_exact_auto() -> None:
    summary = formal_summary()
    specialized = summary["observations"][2]
    specialized["conservative_speedup"] = 1.41
    specialized["baseline_round_medians_ms"] = [1.41, 1.41, 1.41]

    document, deployed = build_promoted_route_document(None, summary)

    assert deployed["solution_policy"] == "auto"
    assert document["routes"][0]["policy"] == "auto"
    assert set(document["routes"][0]["match"]) == ROUTE_FIELDS


def test_replacing_exact_incumbent_requires_its_formal_observation() -> None:
    with pytest.raises(ContractError, match="missing a correct incumbent observation"):
        build_promoted_route_document(
            exact_route_document("preprocess"),
            formal_summary(),
        )


def test_challenger_below_margin_keeps_exact_incumbent() -> None:
    summary = formal_summary()
    summary["observations"].append(
        candidate_observation(
            candidate_id="attention-preprocess",
            policy="preprocess",
            speedup=1.49,
        )
    )
    existing = exact_route_document("preprocess")

    document, deployed = build_promoted_route_document(existing, summary)

    assert deployed["solution_policy"] == "preprocess"
    assert document == existing


def test_clear_gain_replaces_only_the_matching_exact_incumbent() -> None:
    summary = formal_summary()
    summary["observations"].append(
        candidate_observation(
            candidate_id="attention-preprocess",
            policy="preprocess",
            speedup=1.45,
        )
    )
    existing = exact_route_document("preprocess")
    unrelated = {
        "match": exact_match(batch_size=8),
        "policy": "reference",
    }
    existing["routes"].append(unrelated)

    document, winner = build_promoted_route_document(existing, summary)

    assert winner["solution_policy"] == "long-tail-online"
    assert unrelated in document["routes"]
    assert {
        "match": exact_match(),
        "policy": "long-tail-online",
    } in document["routes"]


def test_incumbent_winner_keeps_its_existing_exact_route() -> None:
    summary = formal_summary()
    incumbent = summary["observations"][2]
    incumbent["conservative_speedup"] = 1.41
    incumbent["baseline_round_medians_ms"] = [1.41, 1.41, 1.41]
    existing = exact_route_document("long-tail-online")

    document, winner = build_promoted_route_document(existing, summary)

    assert winner["solution_policy"] == "long-tail-online"
    assert document == existing


def test_faster_auto_can_replace_an_exact_specialized_incumbent() -> None:
    summary = formal_summary()
    auto = summary["observations"][1]
    auto["conservative_speedup"] = 1.6
    auto["baseline_round_medians_ms"] = [1.6, 1.6, 1.6]
    summary["observations"].append(
        candidate_observation(
            candidate_id="attention-preprocess",
            policy="preprocess",
            speedup=1.5,
        )
    )

    document, winner = build_promoted_route_document(
        exact_route_document("preprocess"),
        summary,
    )

    assert winner["solution_policy"] == "auto"
    assert document["routes"][0]["policy"] == "auto"
    assert set(document["routes"][0]["match"]) == ROUTE_FIELDS


def test_shared_s512_route_requires_all_formal_summaries(tmp_path: Path) -> None:
    project_root = promotion_project(tmp_path)
    full = s512_summary("mask_s512_full_fp16", padding_ratio=0.0)
    padding = s512_summary("mask_s512_padding_fp16", padding_ratio=0.75)
    bind_workload_summaries(project_root, [full, padding])
    route_path = tmp_path / "routes.json"

    with pytest.raises(ContractError, match="shared runtime route"):
        promote_tuning_summaries(
            project_root,
            [full],
            route_path=route_path,
        )

    document, winners, _ = promote_tuning_summaries(
        project_root,
        [full, padding],
        route_path=route_path,
    )

    assert len(winners) == 2
    assert len(document["routes"]) == 1
    assert set(document["routes"][0]["match"]) == ROUTE_FIELDS
    assert document["routes"][0]["policy"] == "s512-native-softmax"


def test_shared_route_conflict_keeps_the_common_exact_incumbent(
    tmp_path: Path,
) -> None:
    project_root = promotion_project(tmp_path)
    full = s512_summary("mask_s512_full_fp16", padding_ratio=0.0)
    padding = s512_summary("mask_s512_padding_fp16", padding_ratio=0.75)
    padding_winner = padding["observations"][2]
    padding_winner["candidate_id"] = "padding-fused"
    padding_winner["solution_policy"] = "padding"
    padding_winner["execution_path"] = candidate_execution_path("padding")
    bind_workload_summaries(project_root, [full, padding])

    document, deployments, _ = promote_tuning_summaries(
        project_root,
        [full, padding],
        route_path=tmp_path / "routes.json",
    )

    assert [item["solution_policy"] for item in deployments] == ["auto", "auto"]
    assert len(document["routes"]) == 1
    assert document["routes"][0]["policy"] == "auto"
    assert set(document["routes"][0]["match"]) == ROUTE_FIELDS


def test_weak_workload_does_not_block_independent_exact_route(
    tmp_path: Path,
) -> None:
    project_root = promotion_project(tmp_path)
    weak = formal_summary()
    weak["workload"]["case"]["case_id"] = "weak_fixture"
    weak["tuning_id"] = "weak-tuning"
    weak_specialized = weak["observations"][2]
    weak_specialized["conservative_speedup"] = 1.41
    weak_specialized["baseline_round_medians_ms"] = [1.41, 1.41, 1.41]
    strong = copy.deepcopy(formal_summary())
    strong["workload"]["case"]["case_id"] = "strong_fixture"
    strong["workload"]["case"]["causal"] = True
    strong["tuning_id"] = "strong-tuning"
    bind_workload_summaries(project_root, [weak, strong])

    document, deployed, route_path = promote_tuning_summaries(
        project_root,
        [weak, strong],
        route_path=tmp_path / "routes.json",
    )

    policies_by_causal = {
        route["match"]["causal"]: route["policy"] for route in document["routes"]
    }
    assert policies_by_causal == {False: "auto", True: "long-tail-online"}
    assert [winner["solution_policy"] for winner in deployed] == [
        "auto",
        "long-tail-online",
    ]
    assert json.loads(route_path.read_text(encoding="utf-8")) == document
