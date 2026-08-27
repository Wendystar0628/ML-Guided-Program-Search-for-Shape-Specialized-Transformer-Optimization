"""Route schema and exact-key resolution tests."""

from __future__ import annotations

import copy
import platform

import pytest
import torch
import triton

from solution.dispatch import (
    make_route_key,
    resolve_route,
    resolve_route_result,
    validate_route_table,
    validate_verified_route_table,
)
from tests.support.routing_fixtures import (
    exact_match,
    exact_route_document,
    transformer_config,
)


def _runtime_route_key(*, dtype: torch.dtype = torch.float16) -> dict[str, object]:
    return make_route_key(
        transformer_config(),
        dtype=dtype,
        device_type="cuda",
        device_name="Fixture GPU",
        compute_capability="8.9",
        platform_system=platform.system(),
        torch_version=str(torch.__version__),
        cuda_runtime=str(torch.version.cuda),
        triton_version=str(triton.__version__),
        driver="fixture-driver",
    )


def test_exact_route_resolves_and_runtime_mismatch_falls_back() -> None:
    table = validate_route_table(exact_route_document("preprocess"))

    assert resolve_route(table, _runtime_route_key()) == "preprocess"
    assert resolve_route(table, _runtime_route_key(dtype=torch.float32)) == "auto"


def test_route_resolution_reports_calibrated_or_fallback_origin() -> None:
    table = validate_route_table(exact_route_document("long-tail-online"))

    calibrated = resolve_route_result(table, _runtime_route_key())
    drifted = {**_runtime_route_key(), "cuda_runtime": "different-runtime"}
    fallback = resolve_route_result(table, drifted)

    assert (calibrated.policy, calibrated.origin) == (
        "long-tail-online",
        "calibrated",
    )
    assert (fallback.policy, fallback.origin) == ("auto", "fallback")


def test_route_key_contains_process_static_runtime_facts() -> None:
    key = make_route_key(
        transformer_config(),
        dtype=torch.float16,
        device_type="cuda",
    )

    assert key["platform_system"] == platform.system()
    assert key["torch"] == str(torch.__version__)
    assert key["triton"] == str(triton.__version__)
    if torch.version.cuda is not None:
        assert key["cuda_runtime"] == str(torch.version.cuda)


def test_route_table_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        validate_route_table(exact_route_document("unknown-policy"))


def test_route_table_rejects_partial_route() -> None:
    document = exact_route_document("auto")
    route = document["routes"][0]
    assert isinstance(route, dict)
    match = route["match"]
    assert isinstance(match, dict)
    del match["triton"]

    with pytest.raises(ValueError, match="exact|missing fields"):
        validate_route_table(document)


def test_route_table_rejects_duplicate_exact_match() -> None:
    document = exact_route_document("auto")
    routes = document["routes"]
    assert isinstance(routes, list)
    routes.append(copy.deepcopy(routes[0]))

    with pytest.raises(ValueError, match="duplicates"):
        validate_route_table(document)


def test_verified_route_rejects_mismatched_hardware_identity() -> None:
    expected_identity = {
        key: value
        for key, value in exact_match(device_name="Different GPU").items()
        if key not in {"dtype", "B", "S", "D", "heads", "ffn", "layers", "causal"}
    }

    with pytest.raises(ValueError, match="mismatched hardware identity"):
        validate_verified_route_table(
            exact_route_document("auto"),
            expected_identity=expected_identity,
        )
