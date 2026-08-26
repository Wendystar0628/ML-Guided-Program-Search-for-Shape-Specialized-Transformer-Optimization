"""Tests for explainable cross-hardware candidate planning."""

from __future__ import annotations

import pytest

from runner.contracts import WorkloadCase
from runner.hardware_router import analyze_workload, build_routing_plan


def _case(
    *,
    case_id: str = "balanced",
    batch_size: int = 8,
    seq_len: int = 128,
    d_model: int = 512,
    num_heads: int = 8,
    ffn_dim: int = 2048,
    num_layers: int = 6,
    dtype: str = "float16",
    causal: bool = False,
    padding_ratio: float = 0.0,
) -> WorkloadCase:
    return WorkloadCase(
        case_id=case_id,
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=d_model,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        dtype=dtype,
        causal=causal,
        padding_ratio=padding_ratio,
    )


def _ada_profile() -> dict[str, object]:
    return {
        "device_type": "cuda",
        "device_name": "NVIDIA GeForce RTX 4080",
        "compute_capability": "8.9",
        "platform_system": "Windows",
        "torch": "2.12.1",
        "cuda_runtime": "13.2",
        "triton_available": True,
        "triton": "3.7.1",
        "bf16_supported": True,
        "cuda_graph_available": True,
        "driver": "580.0",
        "total_memory_bytes": 16 * 1024**3,
        "sm_count": 76,
        "l2_cache_bytes": 64 * 1024**2,
        "shared_memory_per_sm_bytes": 100 * 1024,
        "registers_per_sm": 65_536,
        "memory_bus_width_bits": 256,
        "memory_clock_khz": 11_200_000,
        "theoretical_memory_bandwidth_gbps": 716.8,
        "performance_anchors": {
            "launch_latency_us": 8.0,
            "graph_replay_per_node_us": 1.0,
            "memory_bandwidth_gbps": 620.0,
            "gemm_tflops": {
                "float16": 82.0,
                "bfloat16": 80.0,
                "float32": 44.0,
            },
            "softmax_giga_elements_per_s": 210.0,
        },
    }


def test_analyze_workload_reports_consistent_compute_and_traffic() -> None:
    case = _case(num_layers=2)

    analysis = analyze_workload(case)
    payload = analysis.as_dict()

    assert analysis.tokens == case.batch_size * case.seq_len
    assert analysis.head_dim == case.d_model // case.num_heads
    assert analysis.total_flops == (
        analysis.projection_ffn_flops + analysis.attention_flops
    )
    assert analysis.estimated_bytes > analysis.attention_matrix_bytes
    assert analysis.arithmetic_intensity_flops_per_byte > 0
    assert analysis.dense_gemm_fraction + analysis.attention_fraction == pytest.approx(
        1.0, abs=2e-6
    )
    assert payload["case_id"] == case.case_id


def test_launch_plan_prioritizes_graph_and_retains_auto() -> None:
    case = _case(
        case_id="launch",
        batch_size=1,
        seq_len=64,
        d_model=256,
        ffn_dim=1024,
        num_layers=4,
    )
    candidates = [
        "eager-reference",
        "eager-torch",
        "eager-auto",
        "compile-reduce-overhead",
        "launch-cudagraph",
    ]

    plan = build_routing_plan(case, _ada_profile(), candidates, limit=3)

    assert plan["source"] == "hardware_cost_model"
    assert plan["bottleneck_class"] == "launch_underfill"
    assert plan["candidate_order"][0] == "launch-cudagraph"
    assert "compile-reduce-overhead" in plan["candidate_order"]
    assert "eager-auto" in plan["candidate_order"]
    assert set(plan) == {
        "source",
        "bottleneck_class",
        "workload_analysis",
        "routing_signals",
        "candidate_order",
        "selection_reasons",
        "capability_rejections",
    }
    assert "confidence" not in plan
    assert plan["routing_signals"]["graph_replay_is_cheaper"] is True


def test_l2_inflated_copy_anchor_is_capped_by_dram_peak() -> None:
    profile = _ada_profile()
    profile["performance_anchors"]["memory_bandwidth_gbps"] = 1_200.0  # type: ignore[index]

    plan = build_routing_plan(_case(), profile, ["eager-auto"])

    assert plan["routing_signals"]["effective_bandwidth_gbps"] == 716.8


def test_long_attention_uses_l2_pressure_to_rank_online_candidate() -> None:
    case = _case(
        case_id="long",
        batch_size=1,
        seq_len=2048,
        num_layers=4,
    )
    profile = _ada_profile()
    profile["l2_cache_bytes"] = 24 * 1024**2

    plan = build_routing_plan(
        case,
        profile,
        [
            "eager-auto",
            "eager-triton",
            "attention-preprocess",
            "long-tail-online",
        ],
    )

    assert plan["bottleneck_class"] == "attention_memory"
    assert plan["candidate_order"][0] == "long-tail-online"
    assert "eager-auto" in plan["candidate_order"]


def test_tight_launch_limit_keeps_deployable_graph_and_auto() -> None:
    case = _case(
        case_id="launch",
        batch_size=1,
        seq_len=64,
        d_model=256,
        ffn_dim=1024,
        num_layers=4,
    )

    plan = build_routing_plan(
        case,
        _ada_profile(),
        ["eager-auto", "eager-cudagraph", "launch-cudagraph"],
        limit=2,
    )

    assert plan["candidate_order"] == ["launch-cudagraph", "eager-auto"]


def test_wide_bf16_uses_ridge_signal_for_compute_candidate() -> None:
    case = _case(
        case_id="wide",
        batch_size=16,
        seq_len=256,
        d_model=1024,
        ffn_dim=4096,
        num_layers=6,
        dtype="bfloat16",
    )

    plan = build_routing_plan(
        case,
        _ada_profile(),
        [
            "eager-auto",
            "compile-default",
            "compile-max-autotune",
            "wide-triton-inplace",
        ],
    )

    assert plan["bottleneck_class"] == "tensor_compute"
    assert plan["candidate_order"][0] == "wide-triton-inplace"
    assert "compile-max-autotune" in plan["candidate_order"]
    assert "eager-auto" in plan["candidate_order"]


def test_s512_attention_materialization_is_classified_as_softmax_reduction() -> None:
    profile = _ada_profile()
    anchors = profile["performance_anchors"]  # type: ignore[assignment]
    anchors["gemm_tflops"]["float16"] = 70.4  # type: ignore[index]
    anchors["softmax_giga_elements_per_s"] = 202.3  # type: ignore[index]

    plan = build_routing_plan(
        _case(batch_size=8, seq_len=512, num_layers=4),
        profile,
        ["eager-auto", "s512-native-softmax", "eager-triton"],
    )

    assert plan["bottleneck_class"] == "softmax_reduction"
    assert 0.15 <= plan["routing_signals"]["softmax_to_compute_lower_bound"] < 0.2
    assert plan["candidate_order"][0] == "s512-native-softmax"


def test_combined_triton_candidate_is_rejected_outside_its_kernel_shape() -> None:
    plan = build_routing_plan(
        _case(dtype="float32", seq_len=128),
        _ada_profile(),
        ["eager-auto", "eager-triton", "compile-default"],
    )

    assert "eager-triton" not in plan["candidate_order"]
    assert "eager-triton" in plan["capability_rejections"]


def test_cpu_profile_rejects_cuda_candidates_but_keeps_auto() -> None:
    profile = {
        "device_type": "cpu",
        "device_name": "CPU",
        "compute_capability": None,
        "cuda_runtime": None,
        "performance_anchors": {},
    }

    plan = build_routing_plan(
        _case(),
        profile,
        ["eager-auto", "compile-default", "eager-triton", "launch-cudagraph"],
    )

    assert plan["candidate_order"] == ["eager-auto"]
    assert set(plan["capability_rejections"]) == {
        "compile-default",
        "eager-triton",
        "launch-cudagraph",
    }


def test_missing_optional_runtimes_are_capability_rejections() -> None:
    profile = _ada_profile()
    profile["triton_available"] = False
    profile["cuda_graph_available"] = False

    plan = build_routing_plan(
        _case(),
        profile,
        ["eager-auto", "eager-triton", "launch-cudagraph"],
    )

    assert plan["candidate_order"] == ["eager-auto"]
    assert "Triton" in plan["capability_rejections"]["eager-triton"]
    assert "CUDA Graph" in plan["capability_rejections"]["launch-cudagraph"]


def test_torch_packed_candidate_remains_eligible_without_triton() -> None:
    profile = _ada_profile()
    profile["triton_available"] = False

    plan = build_routing_plan(
        _case(seq_len=512, num_layers=4, padding_ratio=0.75),
        profile,
        ["eager-auto", "padding-fused", "padding-packed"],
    )

    assert plan["candidate_order"] == ["padding-packed", "eager-auto"]
    assert "padding-packed" not in plan["capability_rejections"]
    assert "Triton" in plan["capability_rejections"]["padding-fused"]


def test_pre_ampere_bf16_rejects_specialized_triton_route() -> None:
    profile = _ada_profile()
    profile["compute_capability"] = "7.5"
    case = _case(
        batch_size=16,
        seq_len=256,
        d_model=1024,
        ffn_dim=4096,
        dtype="bfloat16",
    )

    plan = build_routing_plan(
        case,
        profile,
        ["eager-auto", "wide-triton-inplace", "compile-max-autotune"],
    )

    assert "wide-triton-inplace" not in plan["candidate_order"]
    assert "wide-triton-inplace" in plan["capability_rejections"]
    assert "eager-auto" in plan["candidate_order"]


def test_pre_ampere_online_attention_is_rejected() -> None:
    profile = _ada_profile()
    profile["compute_capability"] = "7.5"

    plan = build_routing_plan(
        _case(batch_size=1, seq_len=2048),
        profile,
        ["eager-auto", "attention-preprocess", "long-tail-online"],
    )

    assert "attention-preprocess" in plan["candidate_order"]
    assert "long-tail-online" not in plan["candidate_order"]
    assert "long-tail-online" in plan["capability_rejections"]


def test_limit_keeps_auto_and_ties_preserve_caller_order() -> None:
    candidates = ["unknown-b", "unknown-a", "eager-auto"]

    plan = build_routing_plan(_case(), _ada_profile(), candidates, limit=2)

    assert plan["candidate_order"] == ["eager-auto", "unknown-b"]


@pytest.mark.parametrize("limit", [0, -1, True])
def test_invalid_limit_is_rejected(limit: object) -> None:
    with pytest.raises(ValueError, match="limit"):
        build_routing_plan(_case(), _ada_profile(), ["eager-auto"], limit=limit)  # type: ignore[arg-type]
