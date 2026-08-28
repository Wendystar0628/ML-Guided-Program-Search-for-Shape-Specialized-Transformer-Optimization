"""Tests for the feasibility gate and incremental cold-start prior."""

from __future__ import annotations

import pytest

from runner.contracts import RunVariant
from runner.hardware_router import analyze_workload, build_routing_plan
from tests.support.runner_fixtures import official_shape

DEPLOYABLE_CANDIDATES = (
    "eager-auto",
    "graph",
    "graph-fused-norm",
    "mixed-fp16-efficient",
    "graph-mixed-fp16-efficient",
)
ALL_CANDIDATES = ("eager-auto", "eager-safe", *DEPLOYABLE_CANDIDATES[1:])


def _ada_profile() -> dict[str, object]:
    return {
        "device_type": "cuda",
        "device_name": "NVIDIA GeForce RTX 4080",
        "compute_capability": "8.9",
        "architecture_family": "ada",
        "cuda_runtime": "13.2",
        "cuda_graph_available": True,
        "triton_available": True,
        "efficient_sdpa_enabled": True,
        "total_memory_bytes": 16 * 1024**3,
        "free_memory_bytes": 16 * 1024**3,
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
    assert analysis.input_output_bytes > 0
    assert analysis.residual_norm_fusible_bytes > 0
    assert analysis.dense_gemm_fraction + analysis.attention_fraction == pytest.approx(
        1.0,
        abs=2e-6,
    )


@pytest.mark.parametrize("case_id", ["official_02", "official_03"])
def test_small_width128_shapes_prioritize_graph_fused_norm(case_id: str) -> None:
    plan = build_routing_plan(
        official_shape(case_id),
        RunVariant(),
        _ada_profile(),
        DEPLOYABLE_CANDIDATES,
        limit=3,
    )

    assert plan["candidate_order"] == [
        "graph-fused-norm",
        "graph",
        "eager-auto",
    ]
    assert plan["feasibility"]["baseline_executable"] is True
    assert plan["routing_signals"]["launch_dominant"] is True
    assert plan["routing_signals"]["effective_operator_bandwidth_gbps"] == 716.8
    assert plan["routing_signals"]["device_copy_bandwidth_gbps"] == 620.0
    assert plan["routing_signals"]["eager_node_timing_estimate_seconds"] > 0
    assert "relative cost gain" in plan["selection_reasons"]["graph"][1]
    graph_fused_reasons = " ".join(plan["selection_reasons"]["graph-fused-norm"])
    assert "residual-plus-LayerNorm fusion" in graph_fused_reasons
    assert "combined relative opportunity" in graph_fused_reasons


def test_operator_bandwidth_converts_copy_payload_when_theoretical_is_unknown() -> None:
    profile = _ada_profile()
    del profile["theoretical_memory_bandwidth_gbps"]

    plan = build_routing_plan(
        official_shape("official_02"),
        RunVariant(),
        profile,
        DEPLOYABLE_CANDIDATES,
        limit=3,
    )

    assert plan["routing_signals"]["device_copy_bandwidth_gbps"] == 620.0
    assert plan["routing_signals"]["effective_operator_bandwidth_gbps"] == 1240.0


@pytest.mark.parametrize(
    "case_id",
    [
        "official_01",
        "official_05",
        "official_07",
        "official_09",
        "official_10",
        "official_11",
    ],
)
def test_measured_s128_family_prioritizes_graph_mixed_attention(
    case_id: str,
) -> None:
    plan = build_routing_plan(
        official_shape(case_id),
        RunVariant(),
        _ada_profile(),
        DEPLOYABLE_CANDIDATES,
        limit=3,
    )

    assert plan["candidate_order"] == [
        "graph-mixed-fp16-efficient",
        "graph",
        "eager-auto",
    ]


def test_extreme_batch_does_not_receive_a_negative_graph_prior() -> None:
    plan = build_routing_plan(
        official_shape("official_06"),
        RunVariant(),
        _ada_profile(),
        DEPLOYABLE_CANDIDATES,
        limit=3,
    )

    assert plan["feasibility"]["baseline_executable"] is True
    assert plan["candidate_order"] == ["eager-auto"]
    assert "graph" in plan["capability_rejections"]


def test_negative_prior_still_retains_the_current_incumbent() -> None:
    plan = build_routing_plan(
        official_shape("official_06"),
        RunVariant(),
        _ada_profile(),
        DEPLOYABLE_CANDIDATES,
        limit=3,
        required_candidate_ids=("graph",),
    )

    assert "eager-auto" in plan["candidate_order"]
    assert "graph" in plan["candidate_order"]
    assert "current calibrated incumbent" in plan["selection_reasons"]["graph"][-1]


def test_unknown_graph_capability_fails_closed() -> None:
    profile = _ada_profile()
    del profile["cuda_graph_available"]

    plan = build_routing_plan(
        official_shape("official_02"),
        RunVariant(),
        profile,
        DEPLOYABLE_CANDIDATES,
    )

    assert "graph" not in plan["candidate_order"]
    assert "positively established" in plan["capability_rejections"]["graph"]


def test_diagnostic_safe_candidate_is_never_ranked_for_deployment() -> None:
    plan = build_routing_plan(
        official_shape("official_02"),
        RunVariant(),
        _ada_profile(),
        ALL_CANDIDATES,
    )

    assert "eager-safe" not in plan["candidate_order"]
    assert "diagnostic-only" in plan["capability_rejections"]["eager-safe"]


def test_case_14_is_rejected_before_candidate_ranking() -> None:
    plan = build_routing_plan(
        official_shape("official_14"),
        RunVariant(),
        _ada_profile(),
        DEPLOYABLE_CANDIDATES,
    )

    assert plan["candidate_order"] == []
    assert plan["decision_scope"] == "unsupported"
    assert plan["requires_full_workload_measurement"] is False
    assert plan["feasibility"]["baseline_executable"] is False
    assert plan["feasibility"]["estimated_peak_to_device_memory"] > 1000
    assert set(plan["capability_rejections"]) == set(DEPLOYABLE_CANDIDATES)


def test_cpu_profile_rejects_cuda_candidates_before_measurement() -> None:
    profile = _ada_profile()
    profile["device_type"] = "cpu"

    plan = build_routing_plan(
        official_shape("official_02"),
        RunVariant(),
        profile,
        DEPLOYABLE_CANDIDATES,
    )

    assert plan["candidate_order"] == []
    assert set(plan["capability_rejections"]) == set(DEPLOYABLE_CANDIDATES)


def test_router_never_selects_more_than_two_challengers() -> None:
    plan = build_routing_plan(
        official_shape("official_07"),
        RunVariant(),
        _ada_profile(),
        DEPLOYABLE_CANDIDATES,
        limit=20,
    )

    assert len(plan["candidate_order"]) == 3
    assert "eager-auto" in plan["candidate_order"]


@pytest.mark.parametrize("limit", [0, -1, True])
def test_invalid_candidate_limit_is_rejected(limit: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_routing_plan(
            official_shape("official_02"),
            RunVariant(),
            _ada_profile(),
            DEPLOYABLE_CANDIDATES,
            limit=limit,  # type: ignore[arg-type]
        )
