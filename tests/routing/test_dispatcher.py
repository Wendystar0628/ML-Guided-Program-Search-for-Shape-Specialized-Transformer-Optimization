"""Offline dispatcher loading, provenance, and runtime-drift tests."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import pytest
import torch
import triton

from solution.dispatch import OfflineDispatcher
from tests.support.routing_fixtures import (
    exact_route_document,
    transformer_config,
    write_catalog_bundle,
)


def _resolve(
    dispatcher: OfflineDispatcher,
    *,
    device_name: str = "Fixture GPU",
    cuda_runtime: str | None = None,
):
    return dispatcher.resolve_result(
        transformer_config(),
        device="cuda",
        dtype=torch.float16,
        device_name=device_name,
        compute_capability="8.9",
        platform_system=platform.system(),
        torch_version=str(torch.__version__),
        cuda_runtime=cuda_runtime or str(torch.version.cuda),
        triton_version=str(triton.__version__),
        driver="fixture-driver",
    )


def test_explicit_exact_route_table_is_loaded_directly(tmp_path: Path) -> None:
    route_path = tmp_path / "routes.json"
    route_path.write_text(
        json.dumps(exact_route_document("reference")),
        encoding="utf-8",
    )

    dispatcher = OfflineDispatcher(route_path)

    assert dispatcher.path == route_path.resolve()
    assert _resolve(dispatcher).policy == "reference"


def test_missing_explicit_route_table_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unable to load route table"):
        OfflineDispatcher(tmp_path / "missing.json")


def test_dispatcher_reads_an_explicit_table_only_once(tmp_path: Path) -> None:
    route_path = tmp_path / "routes.json"
    route_path.write_text(
        json.dumps(exact_route_document("preprocess")),
        encoding="utf-8",
    )
    dispatcher = OfflineDispatcher(route_path)
    route_path.write_text("not valid json", encoding="utf-8")

    assert _resolve(dispatcher).policy == "preprocess"


def test_catalog_rejects_duplicate_exact_routes(tmp_path: Path) -> None:
    document = exact_route_document("auto")
    write_catalog_bundle(tmp_path / "gpu_a", document)
    write_catalog_bundle(tmp_path / "gpu_b", document)

    with pytest.raises(ValueError, match="duplicates another exact route"):
        OfflineDispatcher(catalog_root=tmp_path)


def test_catalog_resolution_reports_the_matching_bundle(tmp_path: Path) -> None:
    first = write_catalog_bundle(
        tmp_path / "gpu_a",
        exact_route_document("preprocess", device_name="Fixture GPU A"),
    )
    second = write_catalog_bundle(
        tmp_path / "gpu_b",
        exact_route_document("long-tail-online", device_name="Fixture GPU B"),
    )

    resolution = _resolve(
        OfflineDispatcher(catalog_root=tmp_path),
        device_name="Fixture GPU B",
    )

    assert resolution.policy == "long-tail-online"
    assert resolution.origin == "calibrated"
    assert resolution.source == str(second.resolve())
    assert resolution.table_sha256 == hashlib.sha256(second.read_bytes()).hexdigest()
    assert resolution.source != str(first.resolve())


def test_catalog_route_falls_back_after_runtime_drift(tmp_path: Path) -> None:
    write_catalog_bundle(
        tmp_path / "fixture_gpu",
        exact_route_document("long-tail-online"),
    )
    dispatcher = OfflineDispatcher(catalog_root=tmp_path)

    calibrated = _resolve(dispatcher)
    drifted = _resolve(dispatcher, cuda_runtime="different-runtime")

    assert (calibrated.policy, calibrated.origin) == (
        "long-tail-online",
        "calibrated",
    )
    assert (drifted.policy, drifted.origin) == ("auto", "fallback")
