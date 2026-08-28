from __future__ import annotations

import copy

import pytest

from solution.dispatch import MANIFEST_SCHEMA_VERSION, validate_bundle_manifest


def _manifest() -> dict[str, object]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "workload_set": {
            "set_id": "official_transformer_v1",
            "sha256": "1" * 64,
        },
        "official": {"snapshot_sha256": "2" * 64},
        "solution": {"implementation_sha256": "3" * 64},
        "route_table": {"sha256": "4" * 64},
        "formal": {
            "protocol": {"preset": "formal"},
            "source_summaries": [
                {
                    "summary_id": "formal-official-02",
                    "case_id": "official_02",
                    "route_sha256": "5" * 64,
                }
            ],
        },
    }


def test_manifest_binds_official_workload_solution_routes_and_formal_source() -> None:
    manifest = validate_bundle_manifest(_manifest())

    assert manifest.workload_set_id == "official_transformer_v1"
    assert manifest.official_snapshot_sha256 == "2" * 64
    assert manifest.solution_implementation_sha256 == "3" * 64
    assert manifest.source_summaries[0].case_id == "official_02"


def test_old_manifest_schema_is_rejected() -> None:
    document = _manifest()
    document["schema_version"] = MANIFEST_SCHEMA_VERSION - 1

    with pytest.raises(ValueError, match="schema_version"):
        validate_bundle_manifest(document)


def test_manifest_rejects_duplicate_formal_summary_identity() -> None:
    document = _manifest()
    duplicate = copy.deepcopy(document["formal"]["source_summaries"][0])
    document["formal"]["source_summaries"].append(duplicate)

    with pytest.raises(ValueError, match="duplicated"):
        validate_bundle_manifest(document)
