from __future__ import annotations

import ast
from pathlib import Path

from runner.candidates import CANDIDATE_SPECS, deployable_policy_ids
from runner.route_promotion import DEPLOYABLE_POLICIES
from solution.dispatch import ALLOWED_POLICIES
from solution.policies import ROUTABLE_POLICY_IDS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_POLICIES = frozenset(
    {"auto", "safe", "causal-sdpa", "graph", "batch-tiled", "inplace-block"}
)


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


def test_policy_candidate_dispatch_and_promotion_share_one_registry() -> None:
    assert ROUTABLE_POLICY_IDS == EXPECTED_POLICIES
    assert ALLOWED_POLICIES == EXPECTED_POLICIES
    assert deployable_policy_ids() == EXPECTED_POLICIES
    assert DEPLOYABLE_POLICIES == EXPECTED_POLICIES


def test_each_policy_has_one_shape_independent_candidate_owner() -> None:
    owners = [spec.solution_policy for spec in CANDIDATE_SPECS.values()]

    assert set(owners) == EXPECTED_POLICIES
    assert len(owners) == len(set(owners))


def test_candidate_registry_contains_only_the_bounded_strategy_set() -> None:
    assert set(CANDIDATE_SPECS) == {
        "eager-auto",
        "eager-safe",
        "causal-sdpa",
        "graph",
        "batch-tiled",
        "inplace-block",
    }
