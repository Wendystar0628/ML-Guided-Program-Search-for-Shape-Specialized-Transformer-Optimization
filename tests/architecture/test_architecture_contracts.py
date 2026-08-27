"""Small structural guards for the project control-plane boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from runner.candidates import (
    CANDIDATE_SPECS,
    deployable_policy_ids,
)
from runner.route_promotion import DEPLOYABLE_EAGER_POLICIES
from solution.dispatch import ALLOWED_POLICIES
from solution.policies import (
    POLICY_SPECS,
    ROUTABLE_POLICY_IDS,
    ExecutionComponent,
    get_policy_spec,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_PACKAGES = frozenset({"official", "runner", "solution"})

pytestmark = pytest.mark.architecture


def _module_name(path: Path) -> str:
    relative = path.relative_to(PROJECT_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_import(module_name: str, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    package = module_name.split(".")[:-1]
    keep = len(package) - (node.level - 1)
    if keep < 0:
        return None
    prefix = package[:keep]
    suffix = [] if node.module is None else node.module.split(".")
    return ".".join((*prefix, *suffix))


def _project_imports(path: Path) -> set[str]:
    module_name = _module_name(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        names: list[str]
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_import(module_name, node)
            names = [] if resolved is None else [resolved]
        else:
            continue
        imported.update(
            name for name in names if name.split(".", maxsplit=1)[0] in PROJECT_PACKAGES
        )
    return imported


def _production_modules() -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for package in PROJECT_PACKAGES:
        for path in (PROJECT_ROOT / package).rglob("*.py"):
            modules[_module_name(path)] = path
    return modules


def test_solution_and_official_packages_point_only_downward() -> None:
    violations: list[str] = []
    for module, path in _production_modules().items():
        imports = _project_imports(path)
        if module == "official" or module.startswith("official."):
            forbidden = imports
        elif module == "solution" or module.startswith("solution."):
            forbidden = {
                imported
                for imported in imports
                if imported.startswith(("runner.", "official."))
                or imported in {"runner", "official"}
            }
        else:
            continue
        violations.extend(
            f"{module} imports {imported}" for imported in sorted(forbidden)
        )

    assert not violations, "invalid dependency direction:\n" + "\n".join(violations)


def test_project_import_graph_is_acyclic() -> None:
    modules = _production_modules()
    graph = {
        module: {
            imported
            for imported in _project_imports(path)
            if imported in modules and imported != module
        }
        for module, path in modules.items()
    }
    visiting: list[str] = []
    visited: set[str] = set()

    def visit(module: str) -> None:
        if module in visiting:
            start = visiting.index(module)
            cycle = " -> ".join((*visiting[start:], module))
            pytest.fail(f"project import cycle detected: {cycle}")
        if module in visited:
            return
        visiting.append(module)
        for dependency in sorted(graph[module]):
            visit(dependency)
        visiting.pop()
        visited.add(module)

    for module in sorted(graph):
        visit(module)


def test_candidate_policy_and_dispatch_registries_are_coherent() -> None:
    candidate_policies = {spec.solution_policy for spec in CANDIDATE_SPECS.values()}

    assert set(POLICY_SPECS) == ROUTABLE_POLICY_IDS
    assert ALLOWED_POLICIES == ROUTABLE_POLICY_IDS
    assert candidate_policies == ROUTABLE_POLICY_IDS
    assert DEPLOYABLE_EAGER_POLICIES == deployable_policy_ids()
    assert all(
        get_policy_spec(policy_id) is spec for policy_id, spec in POLICY_SPECS.items()
    )


def test_each_active_deployable_policy_has_one_candidate_owner() -> None:
    owners: dict[str, list[str]] = {}
    for spec in CANDIDATE_SPECS.values():
        if spec.deployable:
            owners.setdefault(spec.solution_policy, []).append(spec.candidate_id)

    assert set(owners) == deployable_policy_ids()
    assert all(len(candidate_ids) == 1 for candidate_ids in owners.values())


def test_specialized_policy_identity_is_defined_by_required_components() -> None:
    expected = {
        "triton": {
            ExecutionComponent.TRITON_QKV_LAYOUT,
            ExecutionComponent.TRITON_ATTENTION_SOFTMAX,
        },
        "preprocess": {ExecutionComponent.TRITON_ATTENTION_PREPROCESS},
        "s512-native-softmax": {
            ExecutionComponent.S512_NATIVE_HALF_SOFTMAX
        },
        "long-tail-online": {ExecutionComponent.TAIL_ONLINE_ATTENTION},
        "wide-triton-inplace": {
            ExecutionComponent.TRITON_QKV_LAYOUT,
            ExecutionComponent.WIDE_INPLACE_FFN,
        },
        "cuda-graph": {ExecutionComponent.CUDA_GRAPH},
        "balanced-cuda-graph": {ExecutionComponent.CUDA_GRAPH},
        "padding": {ExecutionComponent.TRITON_RESIDUAL},
        "packed": {ExecutionComponent.PACKED_FFN},
    }

    assert {
        policy_id: set(spec.required_components)
        for policy_id, spec in POLICY_SPECS.items()
        if spec.required_components
    } == expected
    assert POLICY_SPECS["triton"].allow_partial_application
    assert all(
        not spec.allow_partial_application
        for policy_id, spec in POLICY_SPECS.items()
        if policy_id != "triton"
    )
