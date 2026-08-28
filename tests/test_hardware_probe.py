"""Focused tests for static hardware facts and bounded performance anchors."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from runner import probe


def test_compact_cpu_environment_is_unchanged() -> None:
    environment = probe.collect_environment(torch.device("cpu"))

    assert set(environment) == {"device", "platform", "torch", "cuda_runtime"}
    assert environment["device"] == "cpu"


def test_cpu_probe_returns_limited_hardware_and_anchor_profiles() -> None:
    profile = probe._hardware_profile(torch.device("cpu"))
    anchors = probe._performance_anchors(torch.device("cpu"))

    assert profile["available"] is True
    assert profile["device_type"] == "cpu"
    assert profile["gpu"] == {"available": False, "reason": "cuda_required"}
    assert profile["platform"]["system"]
    assert profile["software"]["python"]
    assert anchors
    assert all(
        value == {"available": False, "reason": "cuda_required"}
        for value in anchors.values()
    )


def test_gpu_hardware_profile_reports_properties_and_theoretical_bandwidth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    properties = SimpleNamespace(
        name="Fixture GPU",
        major=8,
        minor=9,
        total_memory=16 * 1024**3,
        multi_processor_count=80,
        L2_cache_size=64 * 1024**2,
        shared_memory_per_multiprocessor=128 * 1024,
        shared_memory_per_block=48 * 1024,
        regs_per_multiprocessor=65536,
        warp_size=32,
        max_threads_per_multi_processor=1536,
        memory_bus_width=256,
        memory_clock_rate=10_000_000,
        clock_rate=2_500_000,
        pci_domain_id=0,
        pci_bus_id=1,
        pci_device_id=0,
    )
    monkeypatch.setattr(
        probe.torch.cuda,
        "get_device_properties",
        lambda _index: properties,
    )
    monkeypatch.setattr(
        probe,
        "_nvidia_smi_metadata",
        lambda _index: (
            {
                "driver_version": "600.00",
                "driver_model.current": "WDDM",
                "pci.bus_id": "00000000:01:00.0",
            },
            None,
        ),
    )
    monkeypatch.setattr(probe, "_triton_runtime", lambda: (True, "fixture-triton"))
    monkeypatch.setattr(probe.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(
        probe.torch.cuda,
        "mem_get_info",
        lambda _index: (12 * 1024**3, 16 * 1024**3),
    )
    monkeypatch.setattr(
        probe.torch.backends.cuda,
        "mem_efficient_sdp_enabled",
        lambda: True,
    )

    profile = probe._hardware_profile(torch.device("cuda:0"))
    gpu = profile["gpu"]

    assert profile["platform"]["driver_model"] == "WDDM"
    assert profile["software"]["driver"] == "600.00"
    assert profile["software"]["triton"] == "fixture-triton"
    assert profile["software"]["efficient_sdpa_enabled"] is True
    assert gpu["sm_count"] == 80
    assert gpu["architecture_family"] == "ada"
    assert gpu["bf16_supported"] is True
    assert gpu["l2_cache_bytes"] == 64 * 1024**2
    assert gpu["shared_memory_per_sm_bytes"] == 128 * 1024
    assert gpu["registers_per_sm"] == 65536
    assert gpu["memory_bus_width_bits"] == 256
    assert gpu["pci_bus_id"] == "00000000:01:00.0"
    assert gpu["theoretical_memory_bandwidth_gbps"] == pytest.approx(640.0)
    assert gpu["free_memory_bytes"] == 12 * 1024**3


@pytest.mark.parametrize(
    ("capability", "family"),
    [
        ((8, 0), "ampere"),
        ((8, 9), "ada"),
        ((9, 0), "hopper"),
        ((10, 0), "blackwell"),
        ((12, 0), "blackwell"),
    ],
)
def test_architecture_family_mapping(capability: tuple[int, int], family: str) -> None:
    assert probe._architecture_family(*capability) == family


def test_each_cuda_anchor_failure_is_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_device: torch.device) -> dict[str, object]:
        raise RuntimeError("fixture launch failure")

    def available(*_args: object) -> dict[str, bool]:
        return {"available": True}

    monkeypatch.setattr(probe, "_eager_launch_anchor", fail)
    monkeypatch.setattr(probe, "_cuda_graph_anchor", available)
    monkeypatch.setattr(probe, "_device_copy_anchor", available)
    monkeypatch.setattr(probe, "_gemm_anchor", available)
    monkeypatch.setattr(probe, "_softmax_anchor", available)

    anchors = probe._performance_anchors(torch.device("cuda:0"))

    assert anchors["eager_launch"]["available"] is False
    assert "fixture launch failure" in anchors["eager_launch"]["reason"]
    assert all(
        value["available"] is True
        for name, value in anchors.items()
        if name != "eager_launch"
    )


def test_gemm_anchor_persists_only_the_best_saturation_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measured_dimensions: list[int] = []

    def measure(
        _device: torch.device,
        _dtype: torch.dtype,
        dimension: int,
    ) -> dict[str, float | int]:
        measured_dimensions.append(dimension)
        return {
            "dimension": dimension,
            "latency_ms": 2.0 if dimension == 2048 else 3.0,
            "tflops": 40.0 if dimension == 2048 else 80.0,
        }

    monkeypatch.setattr(probe, "_measure_square_gemm", measure)

    anchor = probe._gemm_anchor(torch.device("cuda:0"), torch.float32)

    assert measured_dimensions == [2048, 4096]
    assert anchor["method"] == "saturated_square_torch_mm"
    assert anchor["dimension"] == 4096
    assert anchor["latency_ms"] == 3.0
    assert anchor["tflops"] == 80.0
    assert "measurements" not in anchor


def test_execute_probe_cpu_includes_new_sections_without_gpu_work() -> None:
    result = probe.execute_probe({"device": "cpu"})

    assert result["outcome"] == "success"
    assert result["failure"] is None
    assert result["probe"]["mode"] == "diagnostic"
    assert result["probe"]["device_operation_passed"] is True
    assert result["probe"]["runtime_policy"] == {
        "matmul_precision": "high",
        "allow_tf32": True,
    }
    assert result["probe"]["hardware_profile"]["device_type"] == "cpu"
    assert result["probe"]["performance_anchors"]["eager_launch"] == {
        "available": False,
        "reason": "cuda_required",
    }
    assert result["probe"]["sdpa"] == {
        "available": False,
        "reason": "cuda_required",
    }


def test_routing_probe_skips_unused_sdpa_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        probe,
        "_sdpa_capabilities",
        lambda _device: pytest.fail("routing probe must not execute SDPA diagnostics"),
    )

    result = probe.execute_probe({"device": "cpu", "probe_mode": "routing"})

    assert result["outcome"] == "success"
    assert result["probe"]["mode"] == "routing"
    assert "sdpa" not in result["probe"]
    assert result["probe"]["performance_anchors"]


def test_execute_probe_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unsupported probe mode"):
        probe.execute_probe({"device": "cpu", "probe_mode": "unknown"})


def test_safe_item_converts_unexpected_result_to_unavailable() -> None:
    result = probe._safe_item(lambda: None)  # type: ignore[arg-type,return-value]

    assert result == {
        "available": False,
        "reason": "probe_item_returned_invalid_result",
    }
