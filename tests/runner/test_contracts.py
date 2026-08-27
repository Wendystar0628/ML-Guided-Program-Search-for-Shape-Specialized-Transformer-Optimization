"""Project and protocol contract tests for the benchmark runner."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from project_identity import solution_implementation_hash
from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    atomic_replace_json,
    atomic_write_json,
    load_json,
    load_workload_set,
    validate_official_snapshot,
)
from tests.support.runner_fixtures import (
    EXPECTED_CASES,
    EXPECTED_GROUPS,
    EXPECTED_OFFICIAL_SHA256,
    PROJECT_ROOT,
    WORKLOAD_SET_ID,
    canonical_workload_hash,
)


def test_official_snapshot_and_core_workload_contract() -> None:
    metadata = validate_official_snapshot(PROJECT_ROOT)
    snapshot_path = PROJECT_ROOT / metadata["snapshot_path"]

    assert metadata["sha256"] == EXPECTED_OFFICIAL_SHA256
    assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == (
        EXPECTED_OFFICIAL_SHA256
    )

    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    actual_cases = tuple(
        (
            case.case_id,
            case.batch_size,
            case.seq_len,
            case.d_model,
            case.num_heads,
            case.ffn_dim,
            case.num_layers,
            case.dtype,
            case.causal,
            case.padding_ratio,
        )
        for case in workload.cases
    )
    assert actual_cases == EXPECTED_CASES
    assert all(case.input_scale == 1.0 for case in workload.cases)
    assert (
        tuple(
            (group.group_id, group.weight, group.case_ids) for group in workload.groups
        )
        == EXPECTED_GROUPS
    )
    assert workload.sha256 == canonical_workload_hash()


def test_solution_hash_excludes_external_route_tables(tmp_path: Path) -> None:
    solution_root = tmp_path / "solution"
    solution_root.mkdir()
    (solution_root / "transformer.py").write_text("VALUE = 1\n", encoding="utf-8")
    route_path = solution_root / "dispatch_routes.json"
    route_path.write_text(
        '{"schema_version":1,"default_policy":"auto","routes":[]}\n',
        encoding="utf-8",
    )
    original_implementation = solution_implementation_hash(solution_root)

    route_path.write_text(
        '{"schema_version":1,"default_policy":"reference","routes":[]}\n',
        encoding="utf-8",
    )

    assert solution_implementation_hash(solution_root) == original_implementation


def test_compile_and_cuda_graph_candidates_are_mutually_exclusive() -> None:
    protocol = MeasurementProtocol(
        preset="smoke",
        compile_solution=True,
        cuda_graph_solution=True,
    )

    with pytest.raises(ContractError, match="cannot combine"):
        protocol.validate()


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
