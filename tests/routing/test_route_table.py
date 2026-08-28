from __future__ import annotations

import copy

import pytest

from route_contracts import resolve_route, validate_route_table
from tests.support.routing_fixtures import exact_match, exact_route_document


def test_exact_official_shape_route_resolves() -> None:
    table = validate_route_table(exact_route_document("graph"))

    assert resolve_route(table, exact_match()) == "graph"
    assert resolve_route(table, exact_match(case_id="official_03")) == "eager-sdpa"


def test_route_table_rejects_removed_policy() -> None:
    document = exact_route_document()
    document["routes"][0]["policy"] = "removed-policy"

    with pytest.raises(ValueError, match="must be one of"):
        validate_route_table(document)


def test_route_table_requires_every_exact_hardware_and_shape_field() -> None:
    document = exact_route_document()
    document["routes"][0]["match"].pop("driver")

    with pytest.raises(ValueError, match="must be exact"):
        validate_route_table(document)


def test_route_table_rejects_duplicate_exact_matches() -> None:
    document = exact_route_document()
    document["routes"].append(copy.deepcopy(document["routes"][0]))

    with pytest.raises(ValueError, match="duplicates"):
        validate_route_table(document)


def test_route_table_rejects_removed_triton_identity() -> None:
    document = exact_route_document()
    document["routes"][0]["match"]["triton"] = "legacy"

    with pytest.raises(ValueError, match="unknown fields: triton"):
        validate_route_table(document)
