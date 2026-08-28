from __future__ import annotations

import copy

import pytest

from route_contracts import (
    MANIFEST_SCHEMA_VERSION,
    RouteTable,
    load_verified_bundle,
    validate_bundle_manifest,
)
from runner.contracts import RunVariant, load_workload_set
from runner.verified_hardware import (
    VerifiedHardwareError,
    validate_checked_bundle_scope,
)
from tests.support.runner_fixtures import PROJECT_ROOT, WORKLOAD_SET_ID

_COVERED_CASE_IDS = tuple(f"official_{index:02d}" for index in range(1, 14))


def test_checked_rtx4080_bundle_matches_current_sources() -> None:
    route_path = (
        PROJECT_ROOT / "verified_hardware" / "nvidia_geforce_rtx_4080" / "routes.json"
    )

    table, _digest, manifest = load_verified_bundle(
        route_path,
        project_root=PROJECT_ROOT,
    )

    assert len(table.routes) == len(_COVERED_CASE_IDS)
    assert manifest.covered_case_ids == _COVERED_CASE_IDS
    assert manifest.excluded_case_ids == ("official_14",)


def _manifest() -> dict[str, object]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "workload_set": {
            "set_id": WORKLOAD_SET_ID,
            "sha256": "1" * 64,
        },
        "official": {"snapshot_sha256": "2" * 64},
        "solution": {"implementation_sha256": "3" * 64},
        "route_table": {"sha256": "4" * 64},
        "formal": {
            "protocol": {
                "preset": "formal",
                "compile_solution": False,
                "matmul_precision": "high",
                "allow_tf32": True,
            },
            "variant": RunVariant().as_dict(),
            "covered_case_ids": list(_COVERED_CASE_IDS),
            "excluded_case_ids": ["official_14"],
        },
    }


def test_manifest_binds_current_formal_scope() -> None:
    manifest = validate_bundle_manifest(_manifest())

    assert manifest.workload_set_id == WORKLOAD_SET_ID
    assert manifest.official_snapshot_sha256 == "2" * 64
    assert manifest.solution_implementation_sha256 == "3" * 64
    assert manifest.formal_variant == RunVariant().as_dict()
    assert manifest.covered_case_ids == _COVERED_CASE_IDS
    assert manifest.excluded_case_ids == ("official_14",)


def test_old_manifest_schema_is_rejected() -> None:
    document = _manifest()
    document["schema_version"] = MANIFEST_SCHEMA_VERSION - 1

    with pytest.raises(ValueError, match="schema_version"):
        validate_bundle_manifest(document)


def test_manifest_rejects_duplicate_covered_case_identity() -> None:
    document = _manifest()
    formal = document["formal"]
    assert isinstance(formal, dict)
    covered = formal["covered_case_ids"]
    assert isinstance(covered, list)
    covered.append(copy.deepcopy(covered[0]))

    with pytest.raises(ValueError, match="duplicates"):
        validate_bundle_manifest(document)


def test_manifest_rejects_covered_and_excluded_overlap() -> None:
    document = _manifest()
    formal = document["formal"]
    assert isinstance(formal, dict)
    excluded = formal["excluded_case_ids"]
    assert isinstance(excluded, list)
    excluded.append("official_01")

    with pytest.raises(ValueError, match="overlap"):
        validate_bundle_manifest(document)


def test_checked_bundle_scope_requires_exact_local_13_case_partition() -> None:
    manifest = validate_bundle_manifest(_manifest())
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    table = RouteTable(
        default_policy="auto",
        routes=tuple(({}, "auto") for _case_id in _COVERED_CASE_IDS),
    )

    validate_checked_bundle_scope(manifest, workload, table, RunVariant())


def test_checked_bundle_scope_rejects_missing_route() -> None:
    manifest = validate_bundle_manifest(_manifest())
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    table = RouteTable(
        default_policy="auto",
        routes=tuple(({}, "auto") for _case_id in _COVERED_CASE_IDS[:-1]),
    )

    with pytest.raises(VerifiedHardwareError, match="one exact route"):
        validate_checked_bundle_scope(manifest, workload, table, RunVariant())


def test_checked_bundle_scope_rejects_a_different_formal_variant() -> None:
    manifest = validate_bundle_manifest(_manifest())
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    table = RouteTable(
        default_policy="auto",
        routes=tuple(({}, "auto") for _case_id in _COVERED_CASE_IDS),
    )

    with pytest.raises(VerifiedHardwareError, match="Formal variant"):
        validate_checked_bundle_scope(
            manifest,
            workload,
            table,
            RunVariant(dtype="float16"),
        )
