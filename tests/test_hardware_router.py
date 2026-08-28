"""Tests for the feasibility gate and incremental cold-start prior."""

from __future__ import annotations

import pytest

from runner.contracts import RunVariant
from runner.hardware_router import analyze_workload, build_routing_plan
from tests.support.runner_fixtures import official_shape

DEPLOYABLE_CANDIDATES = ("eager-auto", "graph", "inplace-block")
ALL_CANDIDATES = ("eager-auto", "eager-safe", "graph", "inplace-block")


def _ada_profile() -> dict[str, object]:
    return {
        "device_type": "cuda",
        "device_name": "NVIDIA GeForce RTX 4080",
        "compute_capability": "8.9",
        "architecture_family": "ada",
        "cuda_runtime": "13.2",
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
    assert analysis.input_output_bytes > 0
    assert analysis.exact_gelu_temporary_bytes > 0
    assert analysis.dense_gemm_fraction + analysis.attention_fraction == pytest.approx(
        1.0,
        abs=2e-6,
    )


def test_small_shape_ranks_incremental_challengers_and_retains_auto() -> None:
    plan = build_routing_plan(
        official_shape("official_02"),
        RunVariant(),
        _ada_profile(),
        DEPLOYABLE_CANDIDATES,
        limit=3,
    )

    assert plan["candidate_order"][0] == "graph"
    assert set(plan["candidate_order"]) == set(DEPLOYABLE_CANDIDATES)
    assert plan["feasibility"]["baseline_executable"] is True
    assert plan["routing_signals"]["launch_dominant"] is True
    assert "relative lower-bound gain" in plan["selection_reasons"]["graph"][1]


def test_extreme_batch_does_not_receive_a_negative_graph_prior() -> None:
    plan = build_routing_plan(
        official_shape("official_06"),
        RunVariant(),
        _ada_profile(),
        DEPLOYABLE_CANDIDATES,
        limit=3,
    )

    assert plan["feasibility"]["baseline_executable"] is True
    assert plan["candidate_order"] == ["inplace-block", "eager-auto"]
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


def test_wide_shape_uses_removable_traffic_not_compute_bound_bonus() -> None:
    plan = build_routing_plan(
        official_shape("official_08"),
        RunVariant(),
        _ada_profile(),
        DEPLOYABLE_CANDIDATES,
        limit=3,
    )

    assert "inplace-block" in plan["candidate_order"]
    assert plan["workload_analysis"]["d_model"] == 1024
    reasons = " ".join(plan["selection_reasons"]["inplace-block"])
    assert "removable exact-GELU temporary traffic" in reasons
    assert "compute" not in reasons


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
        official_shape("official_02"),
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
