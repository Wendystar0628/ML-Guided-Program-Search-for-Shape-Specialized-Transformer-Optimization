from __future__ import annotations

import copy
import json

import pytest

from route_contracts import (
    MANIFEST_SCHEMA_VERSION,
    RouteTable,
    load_route_table_with_digest,
    validate_bundle_manifest,
)
from runner.contracts import RunVariant, load_workload_set
from runner.verified_hardware import (
    VerifiedHardwareError,
    validate_checked_bundle_scope,
    validate_run_routes,
)
from tests.support.runner_fixtures import PROJECT_ROOT, WORKLOAD_SET_ID, official_shape

_COVERED_CASE_IDS = tuple(f"official_{index:02d}" for index in range(1, 14))


def test_rtx4080_bundle_declares_verified_and_provisional_scopes() -> None:
    bundle = PROJECT_ROOT / "verified_hardware" / "nvidia_geforce_rtx_4080"
    route_path = bundle / "routes.json"
    manifest = validate_bundle_manifest(
        json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    )
    table, route_digest = load_route_table_with_digest(route_path)

    assert len(table.routes) == len(_COVERED_CASE_IDS)
    assert route_digest == manifest.route_table_sha256
    assert manifest.covered_case_ids == _COVERED_CASE_IDS
    assert manifest.provisional_case_ids == ("official_14",)
    assert manifest.excluded_case_ids == ()


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
            "provisional_case_ids": ["official_14"],
            "excluded_case_ids": [],
        },
    }


def test_manifest_binds_current_formal_scope() -> None:
    manifest = validate_bundle_manifest(_manifest())

    assert manifest.workload_set_id == WORKLOAD_SET_ID
    assert manifest.official_snapshot_sha256 == "2" * 64
    assert manifest.solution_implementation_sha256 == "3" * 64
    assert manifest.formal_variant == RunVariant().as_dict()
    assert manifest.covered_case_ids == _COVERED_CASE_IDS
    assert manifest.provisional_case_ids == ("official_14",)
    assert manifest.excluded_case_ids == ()


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


def test_manifest_rejects_covered_and_provisional_overlap() -> None:
    document = _manifest()
    formal = document["formal"]
    assert isinstance(formal, dict)
    provisional = formal["provisional_case_ids"]
    assert isinstance(provisional, list)
    provisional.append("official_01")

    with pytest.raises(ValueError, match="overlap"):
        validate_bundle_manifest(document)


def test_checked_bundle_scope_requires_exact_local_13_case_partition() -> None:
    manifest = validate_bundle_manifest(_manifest())
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    table = RouteTable(
        default_policy="eager-sdpa",
        routes=tuple(({}, "eager-sdpa") for _case_id in _COVERED_CASE_IDS),
    )

    validate_checked_bundle_scope(manifest, workload, table, RunVariant())


def test_checked_bundle_scope_rejects_excluding_a_streamed_workload() -> None:
    document = _manifest()
    formal = document["formal"]
    assert isinstance(formal, dict)
    formal["provisional_case_ids"] = []
    formal["excluded_case_ids"] = ["official_14"]
    manifest = validate_bundle_manifest(document)
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    table = RouteTable(
        default_policy="eager-sdpa",
        routes=tuple(({}, "eager-sdpa") for _case_id in _COVERED_CASE_IDS),
    )

    with pytest.raises(VerifiedHardwareError, match="provisional_case_ids"):
        validate_checked_bundle_scope(manifest, workload, table, RunVariant())


def test_checked_bundle_scope_rejects_missing_route() -> None:
    manifest = validate_bundle_manifest(_manifest())
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    table = RouteTable(
        default_policy="eager-sdpa",
        routes=tuple(({}, "eager-sdpa") for _case_id in _COVERED_CASE_IDS[:-1]),
    )

    with pytest.raises(VerifiedHardwareError, match="one exact route"):
        validate_checked_bundle_scope(manifest, workload, table, RunVariant())


def test_checked_bundle_scope_rejects_streamed_only_policy_in_exact_routes() -> None:
    manifest = validate_bundle_manifest(_manifest())
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    policies = ["eager-sdpa"] * len(_COVERED_CASE_IDS)
    policies[0] = "mixed-fp16-core-cudnn"
    table = RouteTable(
        default_policy="eager-sdpa",
        routes=tuple(({}, policy) for policy in policies),
    )

    with pytest.raises(VerifiedHardwareError, match="not eligible for resident"):
        validate_checked_bundle_scope(manifest, workload, table, RunVariant())


def test_checked_bundle_scope_rejects_a_different_formal_variant() -> None:
    manifest = validate_bundle_manifest(_manifest())
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    table = RouteTable(
        default_policy="eager-sdpa",
        routes=tuple(({}, "eager-sdpa") for _case_id in _COVERED_CASE_IDS),
    )

    with pytest.raises(VerifiedHardwareError, match="Formal variant"):
        validate_checked_bundle_scope(
            manifest,
            workload,
            table,
            RunVariant(dtype="float16"),
        )


def _provisional_run() -> dict[str, object]:
    return {
        "workload": {
            "shape": official_shape("official_14").as_dict(),
            "variant": RunVariant().as_dict(),
        },
        "correctness": {"validation_level": "provisional"},
        "workload_execution": {"mode": "batch_streamed"},
        "execution_path": {"route_origin": "runtime_measurement"},
    }


def test_provisional_run_is_outside_the_verified_sweep() -> None:
    manifest = validate_bundle_manifest(_manifest())
    run = _provisional_run()

    with pytest.raises(VerifiedHardwareError, match="verified route scope"):
        validate_run_routes(
            [run],
            table=RouteTable(default_policy="eager-sdpa", routes=()),
            identity={},
            route_path=PROJECT_ROOT / "routes.json",
            route_sha256="0" * 64,
            project_root=PROJECT_ROOT,
            manifest=manifest,
        )
