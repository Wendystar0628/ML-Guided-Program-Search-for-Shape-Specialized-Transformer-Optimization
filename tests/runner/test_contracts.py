"""Project and protocol contract tests for the benchmark runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from project_identity import (
    canonical_json_sha256,
    sha256_file,
    solution_implementation_hash,
)
from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    RunVariant,
    TransformerShape,
    atomic_replace_json,
    atomic_write_json,
    load_json,
    load_workload_set,
    select_transformer_shape,
    validate_official_snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKLOAD_SET_ID = "official_transformer_v1"


def test_official_snapshot_binds_benchmark_and_shape_table() -> None:
    metadata = validate_official_snapshot(PROJECT_ROOT)
    benchmark = metadata["benchmark"]
    shapes = metadata["shapes"]

    assert sha256_file(PROJECT_ROOT / benchmark["path"]) == benchmark["sha256"]
    assert canonical_json_sha256(PROJECT_ROOT / shapes["path"]) == shapes["sha256"]
    assert len(metadata["combined_sha256"]) == 64


def test_official_workload_contract_has_exact_shapes_without_runtime_fields() -> None:
    expected = (
        ("official_01", 64, 128, 128, 4, 128, 4),
        ("official_02", 1, 128, 128, 4, 128, 4),
        ("official_03", 4, 128, 128, 4, 128, 4),
        ("official_04", 16, 128, 128, 4, 128, 4),
        ("official_05", 128, 128, 128, 4, 128, 4),
        ("official_06", 10000, 128, 128, 4, 128, 4),
        ("official_07", 64, 128, 32, 4, 32, 4),
        ("official_08", 64, 128, 1024, 4, 1024, 4),
        ("official_09", 64, 128, 128, 1, 128, 4),
        ("official_10", 64, 128, 128, 2, 128, 4),
        ("official_11", 64, 128, 128, 16, 128, 4),
        ("official_12", 64, 32, 128, 4, 128, 4),
        ("official_13", 64, 1024, 128, 4, 128, 4),
        ("official_14", 32, 100000, 1024, 16, 1024, 2),
    )

    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    actual = tuple(
        (
            shape.case_id,
            shape.batch_size,
            shape.seq_len,
            shape.d_model,
            shape.num_heads,
            shape.ffn_dim,
            shape.num_layers,
        )
        for shape in workload.shapes
    )
    assert actual == expected
    assert all(shape.causal for shape in workload.shapes)
    assert select_transformer_shape(workload, "official_13").seq_len == 1024
    shape_keys = set(workload.shapes[0].as_dict())
    assert {"dtype", "padding_ratio", "input_scale"}.isdisjoint(shape_keys)


def test_shape_and_variant_are_separate_round_trip_contracts() -> None:
    shape = TransformerShape(
        case_id="example",
        batch_size=2,
        seq_len=16,
        d_model=32,
        num_heads=4,
        ffn_dim=64,
        num_layers=2,
        causal=True,
    )
    variant = RunVariant(dtype="float16", padding_ratio=0.25, input_scale=0.5)

    assert TransformerShape.from_dict(shape.as_dict()) == shape
    assert RunVariant.from_dict(variant.as_dict()) == variant


def test_measurement_protocol_uses_new_official_tolerances() -> None:
    protocol = MeasurementProtocol.for_preset("formal")

    assert protocol.rtol == 0.02
    assert protocol.atol == 0.002


def test_solution_hash_ignores_non_executable_metadata(tmp_path: Path) -> None:
    solution_root = tmp_path / "solution"
    solution_root.mkdir()
    (solution_root / "transformer.py").write_text("VALUE = 1\n", encoding="utf-8")
    metadata_path = solution_root / "metadata.json"
    metadata_path.write_text('{"note":"first"}\n', encoding="utf-8")
    original_implementation = solution_implementation_hash(solution_root)

    metadata_path.write_text('{"note":"second"}\n', encoding="utf-8")

    assert solution_implementation_hash(solution_root) == original_implementation

    (solution_root / "transformer.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert solution_implementation_hash(solution_root) != original_implementation


def test_solution_hash_includes_shared_runtime_contracts(tmp_path: Path) -> None:
    solution_root = tmp_path / "solution"
    solution_root.mkdir()
    (solution_root / "transformer.py").write_text("VALUE = 1\n", encoding="utf-8")
    policy_path = tmp_path / "policy_registry.py"
    route_path = tmp_path / "route_contracts.py"
    policy_path.write_text("POLICIES = {'auto'}\n", encoding="utf-8")
    route_path.write_text("ROUTE_SCHEMA = 1\n", encoding="utf-8")
    original_implementation = solution_implementation_hash(solution_root)

    policy_path.write_text("POLICIES = {'auto', 'graph'}\n", encoding="utf-8")
    assert solution_implementation_hash(solution_root) != original_implementation

    policy_path.write_text("POLICIES = {'auto'}\n", encoding="utf-8")
    route_path.write_text("ROUTE_SCHEMA = 2\n", encoding="utf-8")
    assert solution_implementation_hash(solution_root) != original_implementation


def test_manual_compile_controls_remain_part_of_the_measurement_protocol() -> None:
    protocol = MeasurementProtocol.for_preset(
        "smoke",
        compile_baseline=True,
        compile_solution=True,
        compile_mode="reduce-overhead",
    )

    assert protocol.compile_baseline is True
    assert protocol.compile_solution is True
    assert protocol.compile_mode == "reduce-overhead"


def test_immutable_results_and_mutable_references_have_distinct_writers(
    tmp_path: Path,
) -> None:
    immutable = tmp_path / "run.json"
    atomic_write_json(immutable, {"version": 1})
    with pytest.raises(ContractError, match="refusing to overwrite"):
        atomic_write_json(immutable, {"version": 2})
    assert load_json(immutable) == {"version": 1}

    reference = tmp_path / "reference.json"
    atomic_replace_json(reference, {"version": 1})
    atomic_replace_json(reference, {"version": 2})
    assert load_json(reference) == {"version": 2}
