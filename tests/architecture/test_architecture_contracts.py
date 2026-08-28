from __future__ import annotations

import ast
from pathlib import Path

import pytest

from policy_registry import (
    POLICY_SPECS,
    ROUTABLE_POLICY_IDS,
    policy_ids,
)
from route_contracts import ALLOWED_POLICIES
from runner.candidates import CANDIDATE_SPECS, deployable_policy_ids

pytestmark = pytest.mark.architecture

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_EXPLICIT_POLICIES = frozenset({"auto", "safe", "graph", "inplace-block"})
EXPECTED_ROUTABLE_POLICIES = frozenset({"auto", "graph", "inplace-block"})


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_solution_and_official_packages_do_not_depend_on_runner() -> None:
    for package in ("solution", "official"):
        for path in (PROJECT_ROOT / package).rglob("*.py"):
            assert "runner" not in _imports(path), path


def test_policy_candidate_and_route_contract_share_one_registry() -> None:
    assert policy_ids() == EXPECTED_EXPLICIT_POLICIES
    assert frozenset(POLICY_SPECS) == EXPECTED_EXPLICIT_POLICIES
    assert ROUTABLE_POLICY_IDS == EXPECTED_ROUTABLE_POLICIES
    assert ALLOWED_POLICIES == EXPECTED_ROUTABLE_POLICIES
    assert deployable_policy_ids() == EXPECTED_ROUTABLE_POLICIES


def test_each_policy_has_one_shape_independent_candidate_owner() -> None:
    owners = [spec.solution_policy for spec in CANDIDATE_SPECS.values()]

    assert set(owners) == EXPECTED_EXPLICIT_POLICIES
    assert len(owners) == len(set(owners))
    assert CANDIDATE_SPECS["eager-safe"].deployable is False


def test_candidates_derive_capabilities_from_policy_specs() -> None:
    for candidate in CANDIDATE_SPECS.values():
        assert (
            candidate.required_components
            == POLICY_SPECS[candidate.solution_policy].required_components
        )


def test_candidate_registry_contains_only_distinct_strategies() -> None:
    assert set(CANDIDATE_SPECS) == {
        "eager-auto",
        "eager-safe",
        "graph",
        "inplace-block",
    }


def test_removed_solution_local_policy_registry_does_not_return() -> None:
    assert not (PROJECT_ROOT / "solution" / "policies.py").exists()
