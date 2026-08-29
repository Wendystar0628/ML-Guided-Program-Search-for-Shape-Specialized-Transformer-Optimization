from __future__ import annotations

import ast
from pathlib import Path

import pytest

from policy_registry import (
    POLICY_SPECS,
    ROUTABLE_POLICY_IDS,
    ExecutionComponent,
    PolicySpec,
    ResidualNormBackend,
    RuntimeWrapper,
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
        "graph-mixed-fp16-core-efficient-compiled-norm",
        "graph-fp16-shadow-efficient-compiled-norm",
        "graph-mixed-fp16-core-efficient-triton-mixed-norm-reuse-input",
        "graph-fp16-shadow-efficient-triton-mixed-norm-reuse-input",
        "batch-tiled-mixed-fp16-core-efficient-compiled-norm",
        "batch-tiled-shape06-triton-mixed-norm-fp16-shadow",
        "compiled-mixed-fp16-core-efficient",
        "compiled-shape08-fp16-shadow-weights",
        "compiled-shape11-dh8-triton-fp16-shadow",
        "compiled-shape13-triton-attention-fp16-shadow",
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
    shadow = CANDIDATE_SPECS["compiled-shape08-fp16-shadow-weights"]
    assert shadow.deployable is True
    assert shadow.exact_route_eligible is True


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
    assert (
        POLICY_SPECS[
            "batch-tiled-shape06-triton-mixed-norm-fp16-shadow"
        ].residual_norm
        is ResidualNormBackend.TRITON_MIXED
    )
    shape06 = POLICY_SPECS[
        "batch-tiled-shape06-triton-mixed-norm-fp16-shadow"
    ]
    assert shape06.triton_initial_fp16_norm is True
    assert (
        ExecutionComponent.TRITON_INITIAL_FP16_LAYER_NORM
        in shape06.required_components
    )
    assert (
        POLICY_SPECS["compiled-mixed-fp16-core-efficient"].compile_mode
        == "max-autotune"
    )
    assert (
        POLICY_SPECS[
            "compiled-shape13-triton-attention-fp16-shadow"
        ].compile_mode
        == "max-autotune-no-cudagraphs"
    )
    assert POLICY_SPECS["eager-sdpa"].compile_mode is None
    assert POLICY_SPECS[
        "graph-mixed-fp16-core-efficient-triton-mixed-norm-reuse-input"
    ].reuse_unchanged_input


def test_compile_mode_is_owned_only_by_compiled_policy_specs() -> None:
    assert (
        PolicySpec("compiled", runtime=RuntimeWrapper.COMPILED_FORWARD).compile_mode
        == "max-autotune"
    )
    with pytest.raises(ValueError, match="unsupported compiled-forward mode"):
        PolicySpec(
            "empty-mode",
            runtime=RuntimeWrapper.COMPILED_FORWARD,
            compile_mode="",
        )
    with pytest.raises(ValueError, match="unsupported compiled-forward mode"):
        PolicySpec(
            "unknown-mode",
            runtime=RuntimeWrapper.COMPILED_FORWARD,
            compile_mode="unknown",
        )
    with pytest.raises(ValueError, match="valid only"):
        PolicySpec("eager-with-mode", compile_mode="max-autotune")


def test_input_reuse_is_owned_only_by_cuda_graph_specs() -> None:
    assert PolicySpec(
        "versioned-graph",
        runtime=RuntimeWrapper.CUDA_GRAPH,
        reuse_unchanged_input=True,
    ).reuse_unchanged_input
    with pytest.raises(ValueError, match="valid only"):
        PolicySpec("eager-versioned", reuse_unchanged_input=True)


def test_attention_layout_is_bound_to_the_backend_contract() -> None:
    assert (
        PolicySpec(
            "dh8-bsd",
            attention="triton_dh8_causal_attention_bsd",
            attention_output_layout="bsd",
        ).attention_output_layout
        == "bsd"
    )
    with pytest.raises(ValueError, match="requires 'bsd' output layout"):
        PolicySpec("dh8-wrong-layout", attention="triton_dh8_causal_attention_bsd")
    with pytest.raises(ValueError, match="requires 'bhsd' output layout"):
        PolicySpec("sdpa-wrong-layout", attention_output_layout="bsd")


def test_initial_norm_is_bound_to_the_fixed_shape06_runtime_contract() -> None:
    with pytest.raises(ValueError, match="fixed 128-row batch-tiled"):
        PolicySpec("eager-initial-norm", triton_initial_fp16_norm=True)
    with pytest.raises(ValueError, match="fixed 128-row batch-tiled"):
        PolicySpec(
            "wrong-tile-initial-norm",
            attention="mixed_fp16_efficient",
            linear_compute="float16_shadow",
            residual_norm=ResidualNormBackend.TRITON_MIXED,
            runtime=RuntimeWrapper.BATCH_TILED_CUDA_GRAPH,
            batch_tile_size=64,
            triton_initial_fp16_norm=True,
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
        "graph-mixed-fp16-core-efficient-compiled-norm",
        "graph-fp16-shadow-efficient-compiled-norm",
        "graph-mixed-fp16-core-efficient-triton-mixed-norm-reuse-input",
        "graph-fp16-shadow-efficient-triton-mixed-norm-reuse-input",
        "batch-tiled-mixed-fp16-core-efficient-compiled-norm",
        "batch-tiled-shape06-triton-mixed-norm-fp16-shadow",
        "compiled-mixed-fp16-core-efficient",
        "compiled-shape08-fp16-shadow-weights",
        "compiled-shape11-dh8-triton-fp16-shadow",
        "compiled-shape13-triton-attention-fp16-shadow",
    }


def test_removed_solution_local_policy_registry_does_not_return() -> None:
    assert not (PROJECT_ROOT / "solution" / "policies.py").exists()
