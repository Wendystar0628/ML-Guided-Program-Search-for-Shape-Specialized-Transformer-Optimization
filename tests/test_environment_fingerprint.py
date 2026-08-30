from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from deployment import environment
from deployment.environment import (
    EnvironmentFingerprint,
    official_definitions_digest,
    solution_implementation_digest,
)
from deployment.registry import (
    DEPLOYMENT_SCHEMA_VERSION,
    ShapeFingerprint,
    iter_deployed_configs,
    publish_deployed_config,
    resolve_deployed_config,
)
from solution.config import portable_config


def _environment(**changes: object) -> EnvironmentFingerprint:
    fingerprint = EnvironmentFingerprint(
        device_name="Test GPU",
        compute_capability="9.9",
        driver_version="600.1",
        torch_version="2.12.0",
        cuda_runtime_version="13.2",
        cudnn_version="92000",
        triton_version="3.7.0",
        matmul_precision="highest",
        allow_tf32=False,
        allow_fp16_reduced_precision_reduction=False,
        cudnn_allow_tf32=False,
        official_definitions_digest="official-a",
        solution_implementation_digest="solution-a",
    )
    return replace(fingerprint, **changes)


def _shape() -> ShapeFingerprint:
    return ShapeFingerprint(
        batch_size=1,
        qkv_dim=128,
        heads=4,
        seq_len=128,
        layers=4,
        causal=True,
        ffn_dim=128,
        dtype="float32",
        padding_ratio=0.0,
        input_scale=1.0,
    )


def _write_digest_fixture(root: Path) -> None:
    (root / "official").mkdir()
    (root / "solution").mkdir()
    (root / "official" / "test_shapes.json").write_text("{}", encoding="utf-8")
    (root / "official" / "torch_transformer_benchmark.py").write_text(
        "OFFICIAL = True\n", encoding="utf-8"
    )
    (root / "solution" / "kernel.py").write_text(
        "def kernel(): return 1\n", encoding="utf-8"
    )


def test_identity_is_stable_and_binds_every_environment_field() -> None:
    fingerprint = _environment()

    assert EnvironmentFingerprint.from_dict(fingerprint.to_dict()) == fingerprint
    assert EnvironmentFingerprint.from_dict(fingerprint.to_dict()).identity == (
        fingerprint.identity
    )
    assert len(fingerprint.identity) == 64
    assert len(fingerprint.measurement_identity) == 64

    for field, changed in {
        "device_name": "Other GPU",
        "compute_capability": "10.0",
        "driver_version": "601.0",
        "torch_version": "2.13.0",
        "cuda_runtime_version": "13.3",
        "cudnn_version": "93000",
        "triton_version": "3.8.0",
        "matmul_precision": "high",
        "allow_tf32": True,
        "allow_fp16_reduced_precision_reduction": True,
        "cudnn_allow_tf32": True,
        "official_definitions_digest": "official-b",
        "solution_implementation_digest": "solution-b",
    }.items():
        updated = replace(fingerprint, **{field: changed})
        assert updated.identity != fingerprint.identity
        if field not in {
            "official_definitions_digest",
            "solution_implementation_digest",
        }:
            assert updated.measurement_identity != fingerprint.measurement_identity

    for source_field in (
        "official_definitions_digest",
        "solution_implementation_digest",
    ):
        assert (
            replace(fingerprint, **{source_field: "changed"}).measurement_identity
            == fingerprint.measurement_identity
        )


def test_source_digests_track_only_their_owned_inputs(tmp_path: Path) -> None:
    _write_digest_fixture(tmp_path)
    official_before = official_definitions_digest(tmp_path)
    solution_before = solution_implementation_digest(tmp_path)

    (tmp_path / "official" / "test_shapes.json").write_text(
        '{"changed":true}', encoding="utf-8"
    )
    assert official_definitions_digest(tmp_path) != official_before
    assert solution_implementation_digest(tmp_path) == solution_before

    (tmp_path / "solution" / "kernel.py").write_text(
        "def kernel(): return 2\n", encoding="utf-8"
    )
    assert solution_implementation_digest(tmp_path) != solution_before


def test_detect_collects_complete_runtime_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_digest_fixture(tmp_path)
    fake_torch = SimpleNamespace(
        __version__="2.12.1+cu132",
        version=SimpleNamespace(cuda="13.2"),
        device=lambda value: SimpleNamespace(type="cuda", index=0),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_properties=lambda index: SimpleNamespace(uuid="abc"),
            get_device_capability=lambda index: (8, 9),
            get_device_name=lambda index: "NVIDIA Test GPU",
        ),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(version=lambda: 92000, allow_tf32=False),
            cuda=SimpleNamespace(
                matmul=SimpleNamespace(
                    allow_tf32=True,
                    allow_fp16_reduced_precision_reduction=False,
                )
            ),
        ),
        get_float32_matmul_precision=lambda: "high",
    )
    monkeypatch.setattr(environment, "torch", fake_torch)
    monkeypatch.setattr(environment, "_driver_version", lambda index, uuid: "610.88")
    monkeypatch.setattr(environment, "installed_triton_version", lambda: "3.7.1")

    fingerprint = EnvironmentFingerprint.detect("cuda:0", project_root=tmp_path)

    assert fingerprint.to_dict() == {
        "device_name": "NVIDIA Test GPU",
        "compute_capability": "8.9",
        "driver_version": "610.88",
        "torch_version": "2.12.1+cu132",
        "cuda_runtime_version": "13.2",
        "cudnn_version": "92000",
        "triton_version": "3.7.1",
        "matmul_precision": "high",
        "allow_tf32": True,
        "allow_fp16_reduced_precision_reduction": False,
        "cudnn_allow_tf32": False,
        "official_definitions_digest": official_definitions_digest(tmp_path),
        "solution_implementation_digest": solution_implementation_digest(tmp_path),
    }


def test_process_math_mode_disables_reduced_precision_fp16_reductions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(
                matmul=SimpleNamespace(
                    allow_fp16_reduced_precision_reduction=True,
                )
            )
        )
    )
    monkeypatch.setattr(environment, "torch", fake_torch)

    environment.configure_process_math_mode()

    assert not (
        fake_torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    )


def test_triton_version_supports_the_windows_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_version(distribution: str) -> str:
        if distribution == "triton":
            raise environment.importlib.metadata.PackageNotFoundError
        assert distribution == "triton-windows"
        return "3.7.1.post27"

    monkeypatch.setattr(environment.importlib.metadata, "version", fake_version)

    assert environment.installed_triton_version() == "3.7.1.post27"


def test_driver_detection_fails_instead_of_using_an_empty_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_driver(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError

    monkeypatch.setattr(environment.subprocess, "run", missing_driver)

    with pytest.raises(RuntimeError, match="NVIDIA driver version"):
        environment._driver_version(0, "abc")


def test_registry_exact_matches_full_environment_and_iterates_entries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "deployed.json"
    shape = _shape()
    config = portable_config()
    first = _environment()
    second = _environment(driver_version="601.0")

    publish_deployed_config(hardware=first, shape=shape, config=config, path=path)
    publish_deployed_config(hardware=second, shape=shape, config=config, path=path)

    assert resolve_deployed_config(hardware=first, shape=shape, path=path) == config
    assert (
        resolve_deployed_config(
            hardware=replace(first, torch_version="2.13.0"), shape=shape, path=path
        )
        is None
    )
    assert iter_deployed_configs(hardware=first, path=path) == ((shape, config),)
    assert iter_deployed_configs(hardware=second, path=path) == ((shape, config),)


def test_old_deployment_schema_is_invalidated_on_read_and_publish(
    tmp_path: Path,
) -> None:
    path = tmp_path / "deployed.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundles": [
                    {
                        "hardware": {
                            "device_name": "Test GPU",
                            "compute_capability": "9.9",
                        },
                        "entries": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    fingerprint = _environment()
    shape = _shape()
    config = portable_config()

    assert resolve_deployed_config(hardware=fingerprint, shape=shape, path=path) is None
    assert iter_deployed_configs(hardware=fingerprint, path=path) == ()

    publish_deployed_config(
        hardware=fingerprint,
        shape=shape,
        config=config,
        path=path,
    )
    migrated = json.loads(path.read_text(encoding="utf-8"))
    assert migrated["schema_version"] == DEPLOYMENT_SCHEMA_VERSION
    assert migrated["bundles"][0]["hardware"] == fingerprint.to_dict()
