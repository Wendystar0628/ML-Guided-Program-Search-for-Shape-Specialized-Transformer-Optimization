"""Focused tests for official-shape candidate tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from project_identity import solution_implementation_hash
from runner import tuning
from runner.contracts import ContractError, RunVariant
from runner.tuning_contracts import select_deployable_winner
from tests.support.routing_fixtures import formal_summary
from tests.support.runner_fixtures import official_shape, tiny_protocol


def _candidate_ids(case_id: str) -> list[str]:
    return [
        candidate.candidate_id
        for candidate in tuning.candidates_for_shape(
            official_shape(case_id),
            RunVariant(),
        )
    ]


def _execution_path(candidate_id: str) -> dict[str, Any]:
    paths = {
        "eager-sdpa": {
            "requested_policy": "eager-sdpa",
            "selected_policy": "eager-sdpa",
            "attention_backend": "causal_sdpa",
            "runtime_wrapper": "eager",
            "residual_norm_backend": "torch",
            "observed_execution": {
                "complete": True,
                "attention_backends": ["causal_sdpa"],
                "residual_norm_backends": ["torch"],
            },
        },
        "eager-safe": {
            "requested_policy": "safe",
            "selected_policy": "safe",
            "attention_backend": "safe_streaming",
            "runtime_wrapper": "eager",
            "residual_norm_backend": "torch",
            "observed_execution": {
                "complete": True,
                "attention_backends": ["safe_streaming"],
                "residual_norm_backends": ["torch"],
            },
        },
        "graph": {
            "requested_policy": "graph",
            "selected_policy": "graph",
            "attention_backend": "causal_sdpa",
            "runtime_wrapper": "cuda_graph",
            "residual_norm_backend": "torch",
            "observed_execution": {
                "complete": True,
                "attention_backends": ["causal_sdpa"],
                "residual_norm_backends": ["torch"],
                "runtime_wrappers": ["cuda_graph"],
            },
        },
        "graph-fused-norm": {
            "requested_policy": "graph-fused-norm",
            "selected_policy": "graph-fused-norm",
            "attention_backend": "causal_sdpa",
            "runtime_wrapper": "cuda_graph",
            "residual_norm_backend": "compiled_residual_layer_norm",
            "observed_execution": {
                "complete": True,
                "attention_backends": ["causal_sdpa"],
                "residual_norm_backends": ["compiled_residual_layer_norm"],
                "runtime_wrappers": ["cuda_graph"],
            },
        },
        "mixed-fp16-efficient": {
            "requested_policy": "mixed-fp16-efficient",
            "selected_policy": "mixed-fp16-efficient",
            "attention_backend": "mixed_fp16_efficient",
            "runtime_wrapper": "eager",
            "residual_norm_backend": "torch",
            "observed_execution": {
                "complete": True,
                "attention_backends": ["mixed_fp16_efficient"],
                "residual_norm_backends": ["torch"],
            },
        },
        "mixed-fp16-core-efficient": {
            "requested_policy": "mixed-fp16-core-efficient",
            "selected_policy": "mixed-fp16-core-efficient",
            "attention_backend": "mixed_fp16_efficient",
            "attention_compute_dtype": "float16",
            "linear_backend": "autocast_fp16",
            "linear_compute_dtype": "float16",
            "runtime_wrapper": "eager",
            "residual_norm_backend": "torch",
            "observed_execution": {
                "complete": True,
                "attention_backends": ["mixed_fp16_efficient"],
                "attention_compute_dtypes": ["float16"],
                "linear_backends": ["autocast_fp16"],
                "linear_compute_dtypes": ["float16"],
                "residual_norm_backends": ["torch"],
            },
        },
        "mixed-fp16-core-efficient-triton-norm": {
            "requested_policy": "mixed-fp16-core-efficient-triton-norm",
            "selected_policy": "mixed-fp16-core-efficient-triton-norm",
            "attention_backend": "mixed_fp16_efficient",
            "attention_compute_dtype": "float16",
            "linear_backend": "autocast_fp16",
            "linear_compute_dtype": "float16",
            "runtime_wrapper": "eager",
            "residual_norm_backend": "triton_residual_layer_norm",
            "observed_execution": {
                "complete": True,
                "attention_backends": ["mixed_fp16_efficient"],
                "attention_compute_dtypes": ["float16"],
                "linear_backends": ["autocast_fp16"],
                "linear_compute_dtypes": ["float16"],
                "residual_norm_backends": ["triton_residual_layer_norm"],
            },
        },
        "mixed-fp16-core-cudnn": {
            "requested_policy": "mixed-fp16-core-cudnn",
            "selected_policy": "mixed-fp16-core-cudnn",
            "attention_backend": "mixed_fp16_cudnn",
            "attention_compute_dtype": "float16",
            "linear_backend": "autocast_fp16",
            "linear_compute_dtype": "float16",
            "runtime_wrapper": "eager",
            "residual_norm_backend": "torch",
            "observed_execution": {
                "complete": True,
                "attention_backends": ["mixed_fp16_cudnn"],
                "attention_compute_dtypes": ["float16"],
                "linear_backends": ["autocast_fp16"],
                "linear_compute_dtypes": ["float16"],
                "residual_norm_backends": ["torch"],
            },
        },
        "graph-mixed-fp16-efficient": {
            "requested_policy": "graph-mixed-fp16-efficient",
            "selected_policy": "graph-mixed-fp16-efficient",
            "attention_backend": "mixed_fp16_efficient",
            "runtime_wrapper": "cuda_graph",
            "residual_norm_backend": "torch",
            "observed_execution": {
                "complete": True,
                "attention_backends": ["mixed_fp16_efficient"],
                "residual_norm_backends": ["torch"],
                "runtime_wrappers": ["cuda_graph"],
            },
        },
        "graph-mixed-fp16-efficient-compiled-norm": {
            "requested_policy": "graph-mixed-fp16-efficient-compiled-norm",
            "selected_policy": "graph-mixed-fp16-efficient-compiled-norm",
            "attention_backend": "mixed_fp16_efficient",
            "runtime_wrapper": "cuda_graph",
            "residual_norm_backend": "compiled_residual_layer_norm",
            "observed_execution": {
                "complete": True,
                "attention_backends": ["mixed_fp16_efficient"],
                "residual_norm_backends": ["compiled_residual_layer_norm"],
                "runtime_wrappers": ["cuda_graph"],
            },
        },
        "graph-mixed-fp16-core-efficient-compiled-norm": {
            "requested_policy": "graph-mixed-fp16-core-efficient-compiled-norm",
            "selected_policy": "graph-mixed-fp16-core-efficient-compiled-norm",
            "attention_backend": "mixed_fp16_efficient",
            "attention_compute_dtype": "float16",
            "linear_backend": "autocast_fp16",
            "linear_compute_dtype": "float16",
            "runtime_wrapper": "cuda_graph",
            "residual_norm_backend": "compiled_residual_layer_norm",
            "observed_execution": {
                "complete": True,
                "attention_backends": ["mixed_fp16_efficient"],
                "attention_compute_dtypes": ["float16"],
                "linear_backends": ["autocast_fp16"],
                "linear_compute_dtypes": ["float16"],
                "residual_norm_backends": ["compiled_residual_layer_norm"],
                "runtime_wrappers": ["cuda_graph"],
            },
        },
        "batch-tiled-mixed-fp16-core-efficient-compiled-norm": {
            "requested_policy": ("batch-tiled-mixed-fp16-core-efficient-compiled-norm"),
            "selected_policy": ("batch-tiled-mixed-fp16-core-efficient-compiled-norm"),
            "attention_backend": "mixed_fp16_efficient",
            "attention_compute_dtype": "float16",
            "linear_backend": "autocast_fp16",
            "linear_compute_dtype": "float16",
            "runtime_wrapper": "batch_tiled_cuda_graph",
            "batch_tile_size": 128,
            "residual_norm_backend": "compiled_residual_layer_norm",
            "observed_execution": {
                "complete": True,
                "attention_backends": ["mixed_fp16_efficient"],
                "attention_compute_dtypes": ["float16"],
                "linear_backends": ["autocast_fp16"],
                "linear_compute_dtypes": ["float16"],
                "residual_norm_backends": ["compiled_residual_layer_norm"],
                "runtime_wrappers": ["batch_tiled_cuda_graph"],
            },
        },
        "batch-tiled-mixed-fp16-core-efficient-triton-mixed-norm": {
            "requested_policy": (
                "batch-tiled-mixed-fp16-core-efficient-triton-mixed-norm"
            ),
            "selected_policy": (
                "batch-tiled-mixed-fp16-core-efficient-triton-mixed-norm"
            ),
            "attention_backend": "mixed_fp16_efficient",
            "attention_compute_dtype": "float16",
            "linear_backend": "autocast_fp16",
            "linear_compute_dtype": "float16",
            "runtime_wrapper": "batch_tiled_cuda_graph",
            "batch_tile_size": 128,
            "residual_norm_backend": "triton_mixed_residual_layer_norm",
            "observed_execution": {
                "complete": True,
                "attention_backends": ["mixed_fp16_efficient"],
                "attention_compute_dtypes": ["float16"],
                "linear_backends": ["autocast_fp16"],
                "linear_compute_dtypes": ["float16"],
                "residual_norm_backends": ["triton_mixed_residual_layer_norm"],
                "runtime_wrappers": ["batch_tiled_cuda_graph"],
            },
        },
        "compiled-mixed-fp16-core-efficient": {
            "requested_policy": "compiled-mixed-fp16-core-efficient",
            "selected_policy": "compiled-mixed-fp16-core-efficient",
            "attention_backend": "mixed_fp16_efficient",
            "attention_compute_dtype": "float16",
            "linear_backend": "autocast_fp16",
            "linear_compute_dtype": "float16",
            "runtime_wrapper": "compiled_forward",
            "compile_mode": "max-autotune",
            "residual_norm_backend": "torch",
            "observed_execution": {
                "complete": True,
                "attention_backends": ["mixed_fp16_efficient"],
                "attention_compute_dtypes": ["float16"],
                "linear_backends": ["autocast_fp16"],
                "linear_compute_dtypes": ["float16"],
                "residual_norm_backends": ["torch"],
                "runtime_wrappers": ["compiled_forward"],
            },
        },
        "compiled-mixed-fp16-core-shape13-triton-attention": {
            "requested_policy": ("compiled-mixed-fp16-core-shape13-triton-attention"),
            "selected_policy": ("compiled-mixed-fp16-core-shape13-triton-attention"),
            "attention_backend": "triton_shape13_causal_attention",
            "attention_compute_dtype": "float16",
            "linear_backend": "autocast_fp16",
            "linear_compute_dtype": "float16",
            "runtime_wrapper": "compiled_forward",
            "compile_mode": "max-autotune-no-cudagraphs",
            "residual_norm_backend": "torch",
            "observed_execution": {
                "complete": True,
                "attention_backends": ["triton_shape13_causal_attention"],
                "attention_compute_dtypes": ["float16"],
                "linear_backends": ["autocast_fp16"],
                "linear_compute_dtypes": ["float16"],
                "residual_norm_backends": ["torch"],
                "runtime_wrappers": ["compiled_forward"],
            },
        },
    }
    return paths[candidate_id]


def test_candidates_are_small_and_specific_to_official_shape_families() -> None:
    common = [
        "eager-sdpa",
        "eager-safe",
        "graph",
    ]
    extras = {
        "official_01": [
            "graph-mixed-fp16-efficient",
            "graph-mixed-fp16-efficient-compiled-norm",
            "graph-mixed-fp16-core-efficient-compiled-norm",
        ],
        "official_02": ["graph-fused-norm"],
        "official_03": ["graph-fused-norm"],
        "official_04": ["graph-fused-norm"],
        "official_05": [
            "graph-mixed-fp16-efficient",
            "graph-mixed-fp16-efficient-compiled-norm",
            "graph-mixed-fp16-core-efficient-compiled-norm",
            "graph-mixed-fp16-core-efficient-triton-mixed-norm-reuse-input",
        ],
        "official_06": [
            "mixed-fp16-core-efficient",
            "mixed-fp16-core-efficient-triton-norm",
            "batch-tiled-mixed-fp16-core-efficient-compiled-norm",
            "batch-tiled-mixed-fp16-core-efficient-triton-mixed-norm",
        ],
        "official_07": [
            "graph-mixed-fp16-efficient",
            "graph-mixed-fp16-efficient-compiled-norm",
            "graph-mixed-fp16-core-efficient-compiled-norm",
            "compiled-mixed-fp16-core-efficient",
        ],
        "official_08": [
            "mixed-fp16-core-efficient",
            "compiled-mixed-fp16-core-efficient",
            "compiled-shape08-fp16-shadow-weights",
        ],
        "official_09": [
            "graph-mixed-fp16-efficient",
            "graph-mixed-fp16-efficient-compiled-norm",
            "graph-mixed-fp16-core-efficient-compiled-norm",
        ],
        "official_10": [
            "graph-mixed-fp16-efficient",
            "graph-mixed-fp16-efficient-compiled-norm",
            "graph-mixed-fp16-core-efficient-compiled-norm",
        ],
        "official_11": [
            "graph-mixed-fp16-efficient",
            "graph-mixed-fp16-efficient-compiled-norm",
            "graph-mixed-fp16-core-efficient-compiled-norm",
            "compiled-mixed-fp16-core-efficient",
        ],
        "official_12": ["graph-fused-norm"],
        "official_13": [
            "mixed-fp16-efficient",
            "mixed-fp16-core-efficient",
            "compiled-mixed-fp16-core-efficient",
            "compiled-mixed-fp16-core-shape13-triton-attention",
        ],
    }

    for index in range(1, 14):
        case_id = f"official_{index:02d}"
        assert _candidate_ids(case_id) == [*common, *extras.get(case_id, [])]

    assert _candidate_ids("official_14") == [
        "mixed-fp16-efficient",
        "mixed-fp16-cudnn",
        "mixed-fp16-core-efficient",
        "mixed-fp16-core-cudnn",
    ]


def test_runtime_variant_is_not_hidden_inside_the_shape() -> None:
    shape = official_shape("official_02")

    assert [
        candidate.candidate_id
        for candidate in tuning.candidates_for_shape(
            shape,
            RunVariant(padding_ratio=0.5),
        )
    ] == ["eager-safe"]


def test_select_candidates_requires_explicit_valid_order() -> None:
    shape = official_shape("official_02")
    variant = RunVariant()

    selected = tuning.select_candidates(
        shape,
        variant,
        ["graph", "eager-sdpa"],
    )
    assert [item.candidate_id for item in selected] == [
        "graph",
        "eager-sdpa",
    ]
    with pytest.raises(ContractError, match="at least one"):
        tuning.select_candidates(shape, variant, [])
    with pytest.raises(ContractError, match="not available"):
        tuning.select_candidates(shape, variant, ["removed-candidate"])


def test_deployed_policy_maps_back_to_one_candidate() -> None:
    variant = RunVariant()

    assert (
        tuning.deployable_candidate_id_for_policy(
            official_shape("official_02"), variant, "eager-sdpa"
        )
        == "eager-sdpa"
    )
    assert (
        tuning.deployable_candidate_id_for_policy(
            official_shape("official_02"), variant, "safe"
        )
        is None
    )
    assert (
        tuning.deployable_candidate_id_for_policy(
            official_shape("official_02"), variant, "graph"
        )
        == "graph"
    )
    assert (
        tuning.deployable_candidate_id_for_policy(
            official_shape("official_02"), variant, "graph-fused-norm"
        )
        == "graph-fused-norm"
    )
    assert (
        tuning.deployable_candidate_id_for_policy(
            official_shape("official_12"), variant, "graph-fused-norm"
        )
        == "graph-fused-norm"
    )
    assert (
        tuning.deployable_candidate_id_for_policy(
            official_shape("official_13"), variant, "mixed-fp16-efficient"
        )
        == "mixed-fp16-efficient"
    )
    assert (
        tuning.deployable_candidate_id_for_policy(
            official_shape("official_07"),
            variant,
            "graph-mixed-fp16-efficient",
        )
        == "graph-mixed-fp16-efficient"
    )


def test_smoke_finalist_uses_target_latency_not_worker_speedup() -> None:
    shape = official_shape("official_02")
    summary = formal_summary(
        challenger_policy="graph",
        control_speedup=4.0,
        control_target_median_ms=2.0,
        challenger_speedup=1.1,
        challenger_target_median_ms=1.0,
    )
    summary["protocol"]["preset"] = "smoke"

    plans = tuning.build_formal_candidate_plans(
        [shape],
        RunVariant(),
        [summary],
        [None],
    )

    assert plans[0]["candidate_order"] == ["eager-sdpa", "graph"]


def test_deployable_winner_uses_p90_then_candidate_id_for_latency_ties() -> None:
    summary = formal_summary(
        control_target_median_ms=1.0,
        control_target_p90_ms=1.2,
        challenger_target_median_ms=1.0,
        challenger_target_p90_ms=1.1,
    )

    assert select_deployable_winner(summary)["candidate_id"] == "graph"

    summary["observations"][1]["target_p90_ms"] = 1.2
    assert select_deployable_winner(summary)["candidate_id"] == "eager-sdpa"


def test_streamed_only_candidate_cannot_be_selected_for_an_exact_route() -> None:
    summary = formal_summary(
        case_id="official_14",
        challenger_policy="mixed-fp16-core-cudnn",
        challenger_target_median_ms=0.5,
        control_target_median_ms=1.0,
    )
    summary["observations"] = [summary["observations"][1]]

    with pytest.raises(ContractError, match="no correct, applied dispatch candidate"):
        select_deployable_winner(summary)


def test_tuning_runs_serial_candidates_and_selects_the_measured_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solution_root = tmp_path / "solution"
    solution_root.mkdir()
    (solution_root / "transformer.py").write_text("VALUE = 1\n", encoding="utf-8")
    implementation_hash = solution_implementation_hash(solution_root)
    calls: list[str] = []

    def fake_run(
        _project_root: Path,
        *,
        candidate_id: str,
        solution_policy: str,
        result_dir: Path,
        **_kwargs: Any,
    ) -> tuple[dict[str, Any], Path]:
        calls.append(candidate_id)
        speedup = {"eager-sdpa": 2.0, "graph": 1.5}[candidate_id]
        target = {"eager-sdpa": 2.0, "graph": 1.0}[candidate_id]
        baseline = target * speedup
        result = {
            "run_id": f"run-{candidate_id}",
            "outcome": "success",
            "correctness": {
                "passed": True,
                "failed_elements": 0,
                "max_abs_error": 0.0,
            },
            "performance": {
                "baseline": {"median_ms": baseline, "p90_ms": baseline},
                "target": {"median_ms": target, "p90_ms": target},
                "speedup": speedup,
            },
            "source": {
                "solution_sha256": implementation_hash,
                "official_sha256": "fixture-official",
            },
            "execution_path": _execution_path(candidate_id),
        }
        return result, result_dir / f"{candidate_id}.json"

    monkeypatch.setattr(tuning, "run_managed_benchmark", fake_run)

    summary = tuning.run_tuning_case(
        tmp_path,
        workload_set_id="official_transformer_v1",
        workload_sha256="fixture-workload",
        shape=official_shape("official_02"),
        variant=RunVariant(),
        base_protocol=tiny_protocol(),
        device="cpu",
        requested_candidates=["eager-sdpa", "graph"],
    )

    assert calls == ["eager-sdpa", "graph"]
    assert summary["complete"] is True
    assert summary["winner"]["candidate_id"] == "graph"
    assert summary["deployable_winner"]["candidate_id"] == "graph"
    assert Path(summary["summary_path"]).exists()


def test_candidate_fallback_is_observed_but_cannot_win() -> None:
    candidate = next(
        item
        for item in tuning.candidates_for_shape(
            official_shape("official_02"), RunVariant()
        )
        if item.candidate_id == "graph"
    )
    result = {
        "run_id": "fallback",
        "outcome": "success",
        "correctness": {"passed": True, "failed_elements": 0},
        "performance": {
            "baseline": {"median_ms": 2.0},
            "target": {"median_ms": 1.0, "p90_ms": 1.0},
            "speedup": 2.0,
        },
        "source": {
            "solution_sha256": "fixture-solution",
            "official_sha256": "fixture-official",
        },
        "execution_path": {
            "requested_policy": "graph",
            "selected_policy": "safe",
        },
    }

    observation = tuning._observation(candidate, result, Path("fallback.json"))

    assert observation["policy_applied"] is False


def test_graph_candidate_requires_replay_and_underlying_backend_evidence() -> None:
    candidate = tuning.candidate_spec("graph")
    assert candidate is not None
    path = {
        "requested_policy": "graph",
        "selected_policy": "graph",
        "attention_backend": "causal_sdpa",
        "runtime_wrapper": "cuda_graph",
        "residual_norm_backend": "torch",
        "observed_execution": {
            "complete": True,
            "attention_backends": ["causal_sdpa"],
            "residual_norm_backends": ["torch"],
            "runtime_wrappers": ["cuda_graph"],
        },
    }

    assert candidate.evidence_matches(path)
    del path["observed_execution"]["runtime_wrappers"]
    assert not candidate.evidence_matches(path)
