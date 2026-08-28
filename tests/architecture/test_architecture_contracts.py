from __future__ import annotations

import ast
from pathlib import Path

import pytest

from policy_registry import (
    POLICY_SPECS,
    ROUTABLE_POLICY_IDS,
    ResidualNormBackend,
    policy_ids,
)
from route_contracts import ALLOWED_POLICIES
from runner.candidates import (
    CANDIDATE_SPECS,
    deployable_policy_ids,
    exact_route_policy_ids,
)

pytestmark = pytest.mark.architecture

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_EXPLICIT_POLICIES = frozenset(
    {
        "eager-sdpa",
        "safe",
        "graph",
        "graph-fused-norm",
        "mixed-fp16-efficient",
        "mixed-fp16-cudnn",
        "mixed-fp16-core-efficient",
        "mixed-fp16-core-efficient-triton-norm",
        "mixed-fp16-core-cudnn",
        "graph-mixed-fp16-efficient",
        "graph-mixed-fp16-efficient-compiled-norm",
    }
)
EXPECTED_ROUTABLE_POLICIES = EXPECTED_EXPLICIT_POLICIES - {"safe"}


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
    assert exact_route_policy_ids() == EXPECTED_ROUTABLE_POLICIES - {
        "mixed-fp16-core-cudnn"
    }


def test_each_policy_has_one_shape_independent_candidate_owner() -> None:
    owners = [spec.solution_policy for spec in CANDIDATE_SPECS.values()]

    assert set(owners) == EXPECTED_EXPLICIT_POLICIES
    assert len(owners) == len(set(owners))
    assert CANDIDATE_SPECS["eager-safe"].deployable is False
    assert CANDIDATE_SPECS["mixed-fp16-core-cudnn"].deployable is True
    assert CANDIDATE_SPECS["mixed-fp16-core-cudnn"].exact_route_eligible is False


def test_candidates_derive_capabilities_from_policy_specs() -> None:
    for candidate in CANDIDATE_SPECS.values():
        assert (
            candidate.required_components
            == POLICY_SPECS[candidate.solution_policy].required_components
        )

    assert (
        POLICY_SPECS["graph-fused-norm"].residual_norm is ResidualNormBackend.COMPILED
    )
    assert (
        POLICY_SPECS["mixed-fp16-core-efficient-triton-norm"].residual_norm
        is ResidualNormBackend.TRITON
    )


def test_candidate_registry_contains_only_distinct_strategies() -> None:
    assert set(CANDIDATE_SPECS) == {
        "eager-sdpa",
        "eager-safe",
        "graph",
        "graph-fused-norm",
        "mixed-fp16-efficient",
        "mixed-fp16-cudnn",
        "mixed-fp16-core-efficient",
        "mixed-fp16-core-efficient-triton-norm",
        "mixed-fp16-core-cudnn",
        "graph-mixed-fp16-efficient",
        "graph-mixed-fp16-efficient-compiled-norm",
    }


def test_removed_solution_local_policy_registry_does_not_return() -> None:
    assert not (PROJECT_ROOT / "solution" / "policies.py").exists()
