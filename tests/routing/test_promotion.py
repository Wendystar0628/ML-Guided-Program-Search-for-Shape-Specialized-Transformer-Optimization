from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from route_contracts import MANIFEST_SCHEMA_VERSION, validate_route_table
from runner import route_promotion
from runner.contracts import ContractError
from runner.route_promotion import build_promoted_route_document
from tests.support.routing_fixtures import (
    exact_match,
    exact_route_document,
    formal_summary,
)
from tests.support.runner_fixtures import official_shape


def test_formal_winner_writes_one_exact_official_shape_route() -> None:
    document, winner = build_promoted_route_document(None, formal_summary())

    table = validate_route_table(document)
    assert winner["solution_policy"] == "graph"
    assert len(table.routes) == 1
    assert table.routes[0][1] == "graph"


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
        "requested_policy": "graph",
        "selected_policy": "safe",
    }

    document, winner = build_promoted_route_document(None, summary)

    assert winner["solution_policy"] == "eager-sdpa"
    assert document["routes"][0]["policy"] == "eager-sdpa"


def test_challenger_below_gain_margin_keeps_eager_sdpa() -> None:
    document, winner = build_promoted_route_document(
        None,
        formal_summary(challenger_speedup=1.01),
    )

    assert winner["solution_policy"] == "eager-sdpa"
    assert document["routes"][0]["policy"] == "eager-sdpa"


def test_promotion_ranks_target_latency_not_worker_baseline_speedup() -> None:
    document, winner = build_promoted_route_document(
        None,
        formal_summary(
            challenger_policy="graph",
            control_speedup=4.0,
            control_target_median_ms=2.0,
            challenger_speedup=1.1,
            challenger_target_median_ms=1.0,
        ),
    )

    assert winner["solution_policy"] == "graph"
    assert document["routes"][0]["policy"] == "graph"


def test_removed_candidate_identity_is_never_accepted() -> None:
    summary = formal_summary()
    summary["observations"][1] = copy.deepcopy(summary["observations"][1])
    summary["observations"][1]["candidate_id"] = "removed-candidate"

    document, winner = build_promoted_route_document(None, summary)

    assert winner["solution_policy"] == "eager-sdpa"
    assert document["routes"][0]["policy"] == "eager-sdpa"


def _manifest_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    official_hash = "2" * 64
    implementation_hash = "3" * 64
    shapes = tuple(official_shape(f"official_{index:02d}") for index in range(1, 15))
    monkeypatch.setattr(
        route_promotion,
        "load_workload_set",
        lambda _root, _set_id: SimpleNamespace(shapes=shapes),
    )
    monkeypatch.setattr(
        route_promotion,
        "official_snapshot_hash",
        lambda _root: official_hash,
    )
    monkeypatch.setattr(
        route_promotion,
        "solution_implementation_hash",
        lambda _root: implementation_hash,
    )
    summaries = [
        formal_summary(
            case_id=f"official_{index:02d}",
            official_hash=official_hash,
            implementation_hash=implementation_hash,
        )
        for index in range(1, 14)
    ]
    route_document = exact_route_document()
    route_document["routes"] = [
        {
            "match": exact_match(case_id=f"official_{index:02d}"),
            "policy": "graph",
        }
        for index in range(1, 14)
    ]
    return route_document, summaries


def test_manifest_v5_records_verified_and_shape_14_provisional_scopes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_document, summaries = _manifest_inputs(monkeypatch)

    manifest = route_promotion._build_verified_bundle_manifest(
        tmp_path,
        route_document,
        summaries,
        previous_manifest=None,
    )

    assert manifest["formal"] == {
        "protocol": summaries[0]["protocol"],
        "variant": summaries[0]["workload"]["variant"],
        "covered_case_ids": [f"official_{index:02d}" for index in range(1, 14)],
        "provisional_case_ids": ["official_14"],
        "excluded_case_ids": [],
    }


def test_manifest_v5_incremental_update_merges_covered_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_document, summaries = _manifest_inputs(monkeypatch)
    previous = route_promotion._build_verified_bundle_manifest(
        tmp_path,
        route_document,
        summaries,
        previous_manifest=None,
    )

    updated = route_promotion._build_verified_bundle_manifest(
        tmp_path,
        route_document,
        [summaries[0]],
        previous_manifest=previous,
    )

    assert (
        updated["formal"]["covered_case_ids"] == previous["formal"]["covered_case_ids"]
    )


def test_manifest_v5_rejects_old_manifest_on_incremental_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_document, summaries = _manifest_inputs(monkeypatch)
    previous = route_promotion._build_verified_bundle_manifest(
        tmp_path,
        route_document,
        summaries,
        previous_manifest=None,
    )
    previous["schema_version"] = MANIFEST_SCHEMA_VERSION - 1

    with pytest.raises(ContractError, match="complete calibration"):
        route_promotion._build_verified_bundle_manifest(
            tmp_path,
            route_document,
            [summaries[0]],
            previous_manifest=previous,
        )
