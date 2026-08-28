from __future__ import annotations

import json
from pathlib import Path

import torch

from solution.dispatch import OfflineDispatcher
from tests.support.routing_fixtures import (
    exact_route_document,
    route_runtime_identity,
    transformer_config,
)


def _resolve(dispatcher: OfflineDispatcher, *, device_name: str = "Fixture GPU"):
    identity = route_runtime_identity(device_name=device_name)
    return dispatcher.resolve_result(
        transformer_config(),
        device=torch.device("cuda:0"),
        dtype=torch.float32,
        shape=(1, 128, 128),
        device_name=identity["device_name"],
        compute_capability=identity["compute_capability"],
        platform_system=identity["platform_system"],
        torch_version=identity["torch"],
        cuda_runtime=identity["cuda_runtime"],
        driver=identity["driver"],
        matmul_precision=identity["matmul_precision"],
        allow_tf32=identity["allow_tf32"],
    )


def test_explicit_exact_route_table_is_loaded_and_attributed(tmp_path: Path) -> None:
    route_path = tmp_path / "routes.json"
    route_path.write_text(json.dumps(exact_route_document()), encoding="utf-8")

    resolution = _resolve(OfflineDispatcher(route_path))

    assert resolution.policy == "graph"
    assert resolution.origin == "calibrated"
    assert resolution.source is not None
    assert resolution.table_sha256 is not None


def test_runtime_identity_drift_falls_back_closed(tmp_path: Path) -> None:
    route_path = tmp_path / "routes.json"
    route_path.write_text(json.dumps(exact_route_document()), encoding="utf-8")

    resolution = _resolve(OfflineDispatcher(route_path), device_name="Other GPU")

    assert resolution.policy == "auto"
    assert resolution.origin == "fallback"


def test_runtime_policy_drift_falls_back_closed(tmp_path: Path) -> None:
    route_path = tmp_path / "routes.json"
    route_path.write_text(json.dumps(exact_route_document()), encoding="utf-8")
    identity = route_runtime_identity()

    resolution = OfflineDispatcher(route_path).resolve_result(
        transformer_config(),
        device=torch.device("cuda:0"),
        dtype=torch.float32,
        shape=(1, 128, 128),
        device_name=identity["device_name"],
        compute_capability=identity["compute_capability"],
        platform_system=identity["platform_system"],
        torch_version=identity["torch"],
        cuda_runtime=identity["cuda_runtime"],
        driver=identity["driver"],
        matmul_precision="highest",
        allow_tf32=identity["allow_tf32"],
    )

    assert resolution.policy == "auto"
    assert resolution.origin == "fallback"


def test_missing_catalog_has_a_deterministic_auto_fallback(tmp_path: Path) -> None:
    dispatcher = OfflineDispatcher(catalog_root=tmp_path / "missing")

    resolution = _resolve(dispatcher)

    assert resolution.policy == "auto"
    assert resolution.origin == "fallback"
    assert dispatcher.source is None
