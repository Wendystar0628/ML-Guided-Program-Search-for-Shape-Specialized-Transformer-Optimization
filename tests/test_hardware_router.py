"""Tests for explainable cold-start planning on official shapes."""

from __future__ import annotations

import pytest

from runner.contracts import RunVariant
from runner.hardware_router import analyze_workload, build_routing_plan
from tests.support.runner_fixtures import official_shape

ALL_CANDIDATES = (
    "eager-auto",
    "eager-safe",
    "causal-sdpa",
    "graph",
    "batch-tiled",
    "inplace-block",
)


def _ada_profile() -> dict[str, object]:
    return {
        "device_type": "cuda",
        "device_name": "NVIDIA GeForce RTX 4080",
        "compute_capability": "8.9",
        "architecture_family": "ada",
        "cuda_runtime": "13.2",
        "bf16_supported": True,
        "cuda_graph_available": True,
        "total_memory_bytes": 16 * 1024**3,
        "sm_count": 76,
        "l2_cache_bytes": 64 * 1024**2,
        "theoretical_memory_bandwidth_gbps": 716.8,
        "performance_anchors": {
            "launch_latency_us": 8.0,
            "graph_replay_per_node_us": 1.0,
            "memory_bandwidth_gbps": 620.0,
            "gemm_tflops": {"float32": 44.0},
        },
    }


def test_analysis_uses_shape_and_variant_as_separate_inputs() -> None:
    shape = official_shape("official_01")

    analysis = analyze_workload(shape, RunVariant())

    assert analysis.tokens == shape.batch_size * shape.seq_len
    assert analysis.head_dim == shape.d_model // shape.num_heads
    assert analysis.total_flops == (
        analysis.projection_ffn_flops + analysis.attention_flops
    )
    assert analysis.dense_attention_peak_bytes > 0
    assert analysis.estimated_peak_bytes > analysis.dense_attention_peak_bytes
    assert analysis.dense_gemm_fraction + analysis.attention_fraction == pytest.approx(
        1.0, abs=2e-6
    )


def test_small_batch_shape_prioritizes_graph_and_retains_auto_control() -> None:
    plan = build_routing_plan(
        official_shape("official_02"),
        RunVariant(),
        _ada_profile(),
        ALL_CANDIDATES,
        limit=3,
    )

    assert plan["candidate_order"][0] == "graph"
    assert "eager-auto" in plan["candidate_order"]
    assert plan["routing_signals"]["launch_dominant"] is True


def test_extreme_batch_shape_prioritizes_the_capacity_strategy() -> None:
    plan = build_routing_plan(
        official_shape("official_06"),
        RunVariant(),
        _ada_profile(),
        ALL_CANDIDATES,
        limit=3,
    )

    assert plan["candidate_order"][0] == "batch-tiled"
    assert "graph" in plan["capability_rejections"]


def test_wide_shape_gives_the_inplace_block_a_compute_aware_prior() -> None:
    plan = build_routing_plan(
        official_shape("official_08"),
        RunVariant(),
        _ada_profile(),
        ALL_CANDIDATES,
        limit=3,
    )

    assert plan["candidate_order"][0] == "inplace-block"
    assert plan["workload_analysis"]["d_model"] == 1024


def test_long_sequence_rejects_graph_but_keeps_causal_sdpa() -> None:
    plan = build_routing_plan(
        official_shape("official_13"),
        RunVariant(),
        _ada_profile(),
        ALL_CANDIDATES,
        limit=3,
    )

    assert "graph" in plan["capability_rejections"]
    assert "causal-sdpa" in plan["candidate_order"]


def test_cpu_profile_rejects_cuda_candidates_before_measurement() -> None:
    profile = _ada_profile()
    profile["device_type"] = "cpu"

    plan = build_routing_plan(
        official_shape("official_02"),
        RunVariant(),
        profile,
        ALL_CANDIDATES,
    )

    assert plan["candidate_order"] == []
    assert set(plan["capability_rejections"]) == set(ALL_CANDIDATES)


@pytest.mark.parametrize("limit", [0, -1, True])
def test_invalid_candidate_limit_is_rejected(limit: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_routing_plan(
            official_shape("official_02"),
            RunVariant(),
            _ada_profile(),
            ALL_CANDIDATES,
            limit=limit,  # type: ignore[arg-type]
        )
