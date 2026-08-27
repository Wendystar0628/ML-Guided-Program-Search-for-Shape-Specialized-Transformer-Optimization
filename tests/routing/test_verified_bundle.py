"""Verified hardware bundle identity and lifecycle tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from runner import route_promotion
from runner.contracts import ContractError
from runner.locking import exclusive_file_lock, hardware_bundle_lock_path
from runner.route_promotion import (
    auto_promote_calibration,
    find_matching_verified_route,
    verified_profile_from_probe_result,
)
from solution.dispatch import ROUTE_FIELDS, load_verified_bundle
from tests.support.routing_fixtures import (
    FIXTURE_WORKLOAD_SET_ID,
    PROJECT_ROOT,
    bind_workload_summaries,
    promotable_summary,
    promotion_project,
    routing_probe_result,
    write_discovery_bundle,
)


def test_checked_in_verified_bundles_are_current() -> None:
    route_paths = sorted((PROJECT_ROOT / "verified_hardware").glob("*/routes.json"))
    assert route_paths
    for route_path in route_paths:
        table, _digest, _manifest = load_verified_bundle(
            route_path,
            project_root=PROJECT_ROOT,
        )
        assert table.routes


@pytest.mark.parametrize(
    ("identity_path", "replacement", "same_bundle"),
    [
        (("hardware_profile", "gpu", "name"), "Different GPU", False),
        (("hardware_profile", "software", "torch"), "different-torch", True),
        (("hardware_profile", "software", "cuda_runtime"), "different-cuda", True),
    ],
    ids=("device", "torch", "cuda-runtime"),
)
def test_bundle_discovery_requires_route_visible_runtime_identity(
    tmp_path: Path,
    identity_path: tuple[str, ...],
    replacement: str,
    same_bundle: bool,
) -> None:
    profile = verified_profile_from_probe_result(routing_probe_result())
    package = tmp_path / "verified_hardware" / "fixture_gpu"
    route_path = write_discovery_bundle(tmp_path, package, profile)

    assert find_matching_verified_route(tmp_path, profile) == route_path

    drifted = copy.deepcopy(profile)
    target = drifted
    for field in identity_path[:-1]:
        child = target[field]
        assert isinstance(child, dict)
        target = child
    target[identity_path[-1]] = replacement

    assert find_matching_verified_route(tmp_path, drifted) == (
        route_path if same_bundle else None
    )


def test_bundle_discovery_ignores_non_route_machine_label(tmp_path: Path) -> None:
    profile = verified_profile_from_probe_result(routing_probe_result())
    package = tmp_path / "verified_hardware" / "fixture_gpu"
    route_path = write_discovery_bundle(tmp_path, package, profile)
    drifted = copy.deepcopy(profile)
    drifted["hardware_profile"]["platform"]["machine"] = "Different Machine"

    assert find_matching_verified_route(tmp_path, drifted) == route_path


def test_incomplete_bundle_is_ignored_without_being_repaired(tmp_path: Path) -> None:
    profile = verified_profile_from_probe_result(routing_probe_result())
    package = tmp_path / "verified_hardware" / "fixture_gpu"
    package.mkdir(parents=True)
    (package / "profile.json").write_text(json.dumps(profile), encoding="utf-8")

    assert find_matching_verified_route(tmp_path, profile) is None
    assert {path.name for path in package.iterdir()} == {"profile.json"}


def test_manifest_binds_exact_routes_workload_and_solution(tmp_path: Path) -> None:
    profile = verified_profile_from_probe_result(routing_probe_result())
    package = tmp_path / "verified_hardware" / "fixture_gpu"
    route_path = write_discovery_bundle(tmp_path, package, profile)

    table, route_digest, manifest = load_verified_bundle(
        route_path,
        project_root=tmp_path,
    )

    assert table.routes
    assert all(set(match) == ROUTE_FIELDS for match, _policy in table.routes)
    assert manifest.route_table_sha256 == route_digest
    assert manifest.workload_set_id == FIXTURE_WORKLOAD_SET_ID
    assert manifest.source_summaries[0].case_id == "attention_fixture"
    assert len(manifest.source_summaries[0].route_sha256) == 64
    assert len(manifest.official_snapshot_sha256) == 64


def test_formal_calibration_creates_a_complete_verified_bundle(
    tmp_path: Path,
) -> None:
    project_root = promotion_project(tmp_path)
    probe_result = routing_probe_result()
    summary = promotable_summary(project_root)

    document, winners, route_path, created = auto_promote_calibration(
        project_root,
        [summary],
        probe_result=probe_result,
        full_workload_case_ids=["attention_fixture"],
    )

    bundle = project_root / "verified_hardware" / "fixture_gpu"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert created is True
    assert route_path == bundle / "routes.json"
    assert json.loads(route_path.read_text(encoding="utf-8")) == document
    assert winners[0]["candidate_id"] == "long-tail-online"
    assert set(document["routes"][0]["match"]) == ROUTE_FIELDS
    assert manifest["formal"]["source_summaries"] == [
        {
            "summary_id": "fixture-tuning-attention_fixture",
            "case_id": "attention_fixture",
            "route_sha256": manifest["formal"]["source_summaries"][0]["route_sha256"],
        }
    ]
    assert set(manifest["formal"]["source_summaries"][0]) == {
        "summary_id",
        "case_id",
        "route_sha256",
    }
    assert (bundle / "profile.json").is_file()
    assert (bundle / "run_verified.py").is_file()
    assert (bundle / "results" / ".gitignore").is_file()
    assert (
        find_matching_verified_route(
            project_root,
            verified_profile_from_probe_result(probe_result),
        )
        == route_path
    )


def test_new_bundle_requires_the_complete_formal_workload(tmp_path: Path) -> None:
    project_root = promotion_project(tmp_path)
    selected = promotable_summary(project_root)
    missing = promotable_summary(
        project_root,
        case_id="missing_fixture",
        causal=True,
    )
    bind_workload_summaries(project_root, [selected, missing])

    with pytest.raises(ContractError, match="complete Formal workload calibration"):
        auto_promote_calibration(
            project_root,
            [selected],
            probe_result=routing_probe_result(),
            full_workload_case_ids=["attention_fixture", "missing_fixture"],
        )

    assert not (project_root / "verified_hardware").exists()


def test_later_formal_result_updates_the_same_exact_bundle(tmp_path: Path) -> None:
    project_root = promotion_project(tmp_path)
    probe_result = routing_probe_result()
    first = promotable_summary(project_root)
    first_causal = promotable_summary(
        project_root,
        case_id="attention_fixture_causal",
        causal=True,
    )
    bind_workload_summaries(project_root, [first, first_causal])
    _, _, first_route_path, first_created = auto_promote_calibration(
        project_root,
        [first, first_causal],
        probe_result=probe_result,
        full_workload_case_ids=[
            "attention_fixture",
            "attention_fixture_causal",
        ],
    )
    second = promotable_summary(
        project_root,
        case_id="attention_fixture_causal",
        causal=True,
    )
    second["tuning_id"] = "fixture-tuning-second"
    bind_workload_summaries(project_root, [first, second])

    document, _, second_route_path, second_created = auto_promote_calibration(
        project_root,
        [second],
        probe_result=probe_result,
        full_workload_case_ids=[
            "attention_fixture",
            "attention_fixture_causal",
        ],
    )

    assert first_created is True
    assert second_created is False
    assert second_route_path == first_route_path
    assert len(document["routes"]) == 2
    assert {route["match"]["causal"] for route in document["routes"]} == {
        False,
        True,
    }


def test_new_runtime_reuses_the_same_gpu_directory(tmp_path: Path) -> None:
    project_root = promotion_project(tmp_path)
    original_probe = routing_probe_result()
    first = promotable_summary(project_root)
    _, _, original_route, created = auto_promote_calibration(
        project_root,
        [first],
        probe_result=original_probe,
        full_workload_case_ids=["attention_fixture"],
    )
    assert created is True

    new_probe = copy.deepcopy(original_probe)
    software = new_probe["probe"]["hardware_profile"]["software"]
    software.update(
        {
            "torch": "fixture-torch-next",
            "cuda_runtime": "fixture-cuda-next",
            "triton": "fixture-triton-next",
            "driver": "fixture-driver-next",
        }
    )
    second = promotable_summary(project_root)
    second["tuning_id"] = "fixture-tuning-next-runtime"
    second["device_profile"].update(
        {
            "torch": "fixture-torch-next",
            "cuda_runtime": "fixture-cuda-next",
            "triton": "fixture-triton-next",
            "driver": "fixture-driver-next",
        }
    )

    document, _, new_route, created = auto_promote_calibration(
        project_root,
        [second],
        probe_result=new_probe,
        full_workload_case_ids=["attention_fixture"],
    )

    assert created is False
    assert new_route == original_route
    assert len(document["routes"]) == 2
    assert {route["match"]["torch"] for route in document["routes"]} == {
        str(torch.__version__),
        "fixture-torch-next",
    }
    assert len(list((project_root / "verified_hardware").glob("*/routes.json"))) == 1
    persisted_profile = json.loads(
        original_route.with_name("profile.json").read_text(encoding="utf-8")
    )
    assert (
        persisted_profile["hardware_profile"]["software"]["torch"]
        == "fixture-torch-next"
    )


def test_bundle_publication_restores_both_files_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_root = promotion_project(tmp_path)
    probe = routing_probe_result()
    first = promotable_summary(project_root)
    _, _, route_path, _ = auto_promote_calibration(
        project_root,
        [first],
        probe_result=probe,
        full_workload_case_ids=["attention_fixture"],
    )
    manifest_path = route_path.with_name("manifest.json")
    before = (route_path.read_bytes(), manifest_path.read_bytes())
    second = promotable_summary(project_root)
    second["tuning_id"] = "fixture-tuning-retry"
    original_replace = route_promotion._atomic_replace_bytes
    failed = False

    def fail_manifest_once(path: Path, payload: bytes) -> None:
        nonlocal failed
        if path.name == "manifest.json" and not failed:
            failed = True
            raise OSError("injected manifest publication failure")
        original_replace(path, payload)

    monkeypatch.setattr(route_promotion, "_atomic_replace_bytes", fail_manifest_once)

    with pytest.raises(OSError, match="injected manifest"):
        auto_promote_calibration(
            project_root,
            [second],
            probe_result=probe,
            full_workload_case_ids=["attention_fixture"],
        )

    assert (route_path.read_bytes(), manifest_path.read_bytes()) == before
    load_verified_bundle(route_path, project_root=project_root)


def test_hardware_bundle_publication_is_single_writer(tmp_path: Path) -> None:
    project_root = promotion_project(tmp_path)
    summary = promotable_summary(project_root)
    lock_path = hardware_bundle_lock_path(project_root, "fixture_gpu")

    with (
        exclusive_file_lock(lock_path, purpose="test owner"),
        pytest.raises(ContractError, match="another process"),
    ):
        auto_promote_calibration(
            project_root,
            [summary],
            probe_result=routing_probe_result(),
            full_workload_case_ids=["attention_fixture"],
        )


def test_old_manifest_schema_is_rejected(tmp_path: Path) -> None:
    profile = verified_profile_from_probe_result(routing_probe_result())
    route_path = write_discovery_bundle(
        tmp_path,
        tmp_path / "verified_hardware" / "fixture_gpu",
        profile,
    )
    manifest_path = route_path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version must be 2"):
        load_verified_bundle(route_path, project_root=tmp_path)


def test_bundle_rejects_a_different_official_snapshot(tmp_path: Path) -> None:
    profile = verified_profile_from_probe_result(routing_probe_result())
    route_path = write_discovery_bundle(
        tmp_path,
        tmp_path / "verified_hardware" / "fixture_gpu",
        profile,
    )
    manifest_path = route_path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["official"]["snapshot_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="official snapshot hash is stale"):
        load_verified_bundle(route_path, project_root=tmp_path)
