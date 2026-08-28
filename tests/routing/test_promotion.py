from __future__ import annotations

import copy

import pytest

from runner.contracts import ContractError
from runner.route_promotion import build_promoted_route_document
from solution.dispatch import validate_route_table
from tests.support.routing_fixtures import formal_summary


def test_formal_winner_writes_one_exact_official_shape_route() -> None:
    document, winner = build_promoted_route_document(None, formal_summary())

    table = validate_route_table(document)
    assert winner["solution_policy"] == "causal-sdpa"
    assert len(table.routes) == 1
    assert table.routes[0][1] == "causal-sdpa"


def test_smoke_summary_cannot_be_promoted() -> None:
    summary = formal_summary()
    summary["protocol"]["preset"] = "smoke"

    with pytest.raises(ContractError, match="formal"):
        build_promoted_route_document(None, summary)


def test_compiled_formal_summary_cannot_be_promoted() -> None:
    summary = formal_summary()
    summary["protocol"]["compile_solution"] = True

    with pytest.raises(ContractError, match="uncompiled"):
        build_promoted_route_document(None, summary)


def test_graph_winner_can_be_promoted_when_replay_evidence_is_complete() -> None:
    document, winner = build_promoted_route_document(
        None,
        formal_summary(challenger_policy="graph"),
    )

    assert winner["solution_policy"] == "graph"
    assert document["routes"][0]["policy"] == "graph"


def test_candidate_fallback_cannot_be_promoted_as_applied() -> None:
    summary = formal_summary()
    challenger = summary["observations"][1]
    challenger["execution_path"] = {
        "requested_policy": "causal-sdpa",
        "selected_policy": "safe",
    }

    document, winner = build_promoted_route_document(None, summary)

    assert winner["solution_policy"] == "auto"
    assert document["routes"][0]["policy"] == "auto"


def test_challenger_below_gain_margin_keeps_auto() -> None:
    document, winner = build_promoted_route_document(
        None,
        formal_summary(challenger_speedup=1.01),
    )

    assert winner["solution_policy"] == "auto"
    assert document["routes"][0]["policy"] == "auto"


def test_removed_candidate_identity_is_never_accepted() -> None:
    summary = formal_summary()
    summary["observations"][1] = copy.deepcopy(summary["observations"][1])
    summary["observations"][1]["candidate_id"] = "removed-candidate"

    document, winner = build_promoted_route_document(None, summary)

    assert winner["solution_policy"] == "auto"
    assert document["routes"][0]["policy"] == "auto"
