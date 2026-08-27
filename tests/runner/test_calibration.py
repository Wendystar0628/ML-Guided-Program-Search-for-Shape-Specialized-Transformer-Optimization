"""Service-level tests for hardware-aware calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from runner.calibration import (
    CalibrationDependencies,
    CalibrationEvent,
    CalibrationRequest,
    CalibrationService,
    hardware_profile_from_probe,
)
from runner.contracts import ContractError, WorkloadCase
from runner.supervisor import CancellationToken
from tests.support.runner_fixtures import (
    EXPECTED_CASES,
    PROJECT_ROOT,
    WORKLOAD_SET_ID,
    routing_probe_result,
    staged_tuning_summary,
)


def _request(**overrides: Any) -> CalibrationRequest:
    values: dict[str, Any] = {
        "project_root": PROJECT_ROOT,
        "workload_set_id": WORKLOAD_SET_ID,
        "candidate_limit": 1,
    }
    values.update(overrides)
    return CalibrationRequest(**values)


def test_routing_probe_is_flattened_to_compact_model_inputs() -> None:
    profile = hardware_profile_from_probe(routing_probe_result())

    assert profile["device_type"] == "cuda"
    assert profile["device_name"] == "fixture-gpu"
    assert profile["compute_capability"] == "8.9"
    assert profile["architecture_family"] == "ada"
    assert profile["triton"] == "fixture-triton"
    assert profile["triton_available"] is True
    assert profile["bf16_supported"] is True
    assert profile["cuda_graph_available"] is True
    assert profile["platform_system"] == "Windows"
    assert profile["sm_count"] == 76
    assert profile["memory_clock_khz"] == 1_000
    assert profile["performance_anchors"] == {
        "launch_latency_us": 4.0,
        "graph_replay_per_node_us": 2.0,
        "memory_bandwidth_gbps": 600.0,
        "gemm_tflops": {
            "float16": 80.0,
            "bfloat16": 75.0,
            "float32": 40.0,
        },
        "softmax_giga_elements_per_s": 250.0,
    }
    assert "gpu" not in profile
    assert "sdpa" not in profile


def test_plan_only_covers_the_full_workload_without_tuning(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_probe(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], Path]:
        del args
        assert kwargs["probe_mode"] == "routing"
        calls.append("probe")
        return routing_probe_result(), tmp_path / "probe.json"

    def fake_plan(
        case: WorkloadCase,
        hardware_profile: dict[str, Any],
        candidate_ids: tuple[str, ...],
        *,
        limit: int,
    ) -> dict[str, Any]:
        del hardware_profile, candidate_ids
        assert limit == 1
        calls.append(f"plan:{case.case_id}")
        return {
            "source": "hardware_cost_model",
            "candidate_order": ["eager-auto"],
        }

    service = CalibrationService(
        CalibrationDependencies(
            run_probe=fake_probe,
            build_plan=fake_plan,
            run_tuning=lambda *args, **kwargs: pytest.fail(
                "plan-only must not run tuning"
            ),
            find_verified_route=lambda *args, **kwargs: None,
        )
    )

    result = service.run(_request(plan_only=True))

    assert result.outcome == "planned"
    assert result.exit_code == 0
    assert result.case_ids == tuple(case[0] for case in EXPECTED_CASES)
    assert calls == ["probe", *(f"plan:{case[0]}" for case in EXPECTED_CASES)]


def test_formal_plan_only_can_inspect_one_case_on_a_new_device(
    tmp_path: Path,
) -> None:
    service = CalibrationService(
        CalibrationDependencies(
            run_probe=lambda *args, **kwargs: (
                routing_probe_result(),
                tmp_path / "probe.json",
            ),
            find_verified_route=lambda *args, **kwargs: None,
            build_plan=lambda *args, **kwargs: {
                "source": "hardware_cost_model",
                "candidate_order": ["eager-auto"],
            },
            run_tuning=lambda *args, **kwargs: pytest.fail(
                "plan-only must not run tuning"
            ),
        )
    )

    result = service.run(
        _request(
            preset="formal",
            plan_only=True,
            case_ids=("mask_s512_padding_fp16",),
        )
    )

    assert result.outcome == "planned"
    assert result.case_ids == ("mask_s512_padding_fp16",)


@pytest.mark.parametrize(
    ("matmul_precision", "allow_tf32"),
    [("highest", True), ("medium", True), ("high", False)],
)
def test_formal_calibration_rejects_non_deployable_precision_before_probe(
    matmul_precision: str,
    allow_tf32: bool,
) -> None:
    service = CalibrationService(
        CalibrationDependencies(
            run_probe=lambda *args, **kwargs: pytest.fail(
                "invalid Formal settings must fail before probing"
            )
        )
    )

    with pytest.raises(ContractError, match="formal calibration deployment"):
        service.run(
            _request(
                preset="formal",
                matmul_precision=matmul_precision,
                allow_tf32=allow_tf32,
            )
        )


def test_calibration_rejects_duplicate_case_ids_before_probe() -> None:
    service = CalibrationService(
        CalibrationDependencies(
            run_probe=lambda *args, **kwargs: pytest.fail(
                "duplicate case IDs must fail before probing"
            )
        )
    )

    with pytest.raises(ContractError, match="must not contain duplicates"):
        service.run(
            _request(
                case_ids=("launch_s64_fp16", "launch_s64_fp16"),
            )
        )


def test_smoke_calibration_probes_once_then_runs_planned_workloads(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    profiles: list[dict[str, Any]] = []

    def fake_probe(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], Path]:
        del args
        assert kwargs["probe_mode"] == "routing"
        calls.append("probe")
        return routing_probe_result(), tmp_path / "probe.json"

    def fake_plan(
        case: WorkloadCase,
        hardware_profile: dict[str, Any],
        candidate_ids: tuple[str, ...],
        *,
        limit: int,
    ) -> dict[str, Any]:
        del candidate_ids
        assert limit == 1
        calls.append(f"plan:{case.case_id}")
        profiles.append(hardware_profile)
        return {
            "source": "hardware_cost_model",
            "candidate_order": ["eager-auto"],
        }

    def fake_tuning(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        case = kwargs["case"]
        calls.append(f"tune:{case.case_id}")
        assert kwargs["requested_candidates"] == ["eager-auto"]
        assert kwargs["device_profile"] is profiles[0]
        return staged_tuning_summary(case, ["eager-auto"], "smoke", tmp_path)

    service = CalibrationService(
        CalibrationDependencies(
            run_probe=fake_probe,
            build_plan=fake_plan,
            run_tuning=fake_tuning,
            find_verified_route=lambda *args, **kwargs: None,
            promote=lambda *args, **kwargs: pytest.fail(
                "Smoke calibration must not publish routes"
            ),
        )
    )

    result = service.run(
        _request(
            case_ids=("launch_s64_fp16", "wide_s256_bf16"),
        )
    )

    assert result.outcome == "smoke_complete"
    assert calls == [
        "probe",
        "plan:launch_s64_fp16",
        "plan:wide_s256_bf16",
        "tune:launch_s64_fp16",
        "tune:wide_s256_bf16",
    ]
    assert len(profiles) == 2
    assert profiles[0] is profiles[1]


def test_formal_calibration_runs_smoke_then_formal_and_promotes_once(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    route_path = tmp_path / "verified_hardware" / "fixture" / "routes.json"
    probe = routing_probe_result()

    def fake_probe(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], Path]:
        del args, kwargs
        calls.append("probe")
        return probe, tmp_path / "probe.json"

    def fake_plan(case: WorkloadCase, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        calls.append(f"plan:{case.case_id}")
        return {
            "source": "hardware_cost_model",
            "candidate_order": ["eager-auto"],
        }

    def fake_tuning(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        case = kwargs["case"]
        preset = kwargs["base_protocol"].preset
        calls.append(f"{preset}:{case.case_id}")
        return staged_tuning_summary(
            case,
            list(kwargs["requested_candidates"]),
            preset,
            tmp_path,
        )

    def fake_promote(
        project_root: Path,
        summaries: list[dict[str, Any]],
        *,
        probe_result: dict[str, Any],
        full_workload_case_ids: list[str],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], Path, bool]:
        del project_root
        calls.append("promote")
        assert probe_result is probe
        assert full_workload_case_ids == [case[0] for case in EXPECTED_CASES]
        assert [summary["case_id"] for summary in summaries] == [
            "launch_s64_fp16",
            "wide_s256_bf16",
        ]
        assert all(summary["protocol"]["preset"] == "formal" for summary in summaries)
        winners = [dict(summary["deployable_winner"]) for summary in summaries]
        return {"schema_version": 2}, winners, route_path, False

    service = CalibrationService(
        CalibrationDependencies(
            run_probe=fake_probe,
            build_plan=fake_plan,
            run_tuning=fake_tuning,
            find_verified_route=lambda *args, **kwargs: route_path,
            promote=fake_promote,
        )
    )

    result = service.run(
        _request(
            preset="formal",
            case_ids=("launch_s64_fp16", "wide_s256_bf16"),
        )
    )

    assert result.outcome == "formal_promoted"
    assert result.route_path == route_path
    assert calls == [
        "probe",
        "plan:launch_s64_fp16",
        "plan:wide_s256_bf16",
        "smoke:launch_s64_fp16",
        "smoke:wide_s256_bf16",
        "formal:launch_s64_fp16",
        "formal:wide_s256_bf16",
        "promote",
    ]


def test_formal_calibration_does_not_promote_without_a_deployable_winner(
    tmp_path: Path,
) -> None:
    route_path = tmp_path / "verified_hardware" / "fixture" / "routes.json"

    def fake_tuning(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        preset = kwargs["base_protocol"].preset
        summary = staged_tuning_summary(
            kwargs["case"],
            ["eager-auto"],
            preset,
            tmp_path,
        )
        if preset == "formal":
            summary["winner"] = None
            summary["deployable_winner"] = None
        return summary

    service = CalibrationService(
        CalibrationDependencies(
            run_probe=lambda *args, **kwargs: (
                routing_probe_result(),
                tmp_path / "probe.json",
            ),
            build_plan=lambda *args, **kwargs: {
                "source": "hardware_cost_model",
                "candidate_order": ["eager-auto"],
            },
            run_tuning=fake_tuning,
            find_verified_route=lambda *args, **kwargs: route_path,
            promote=lambda *args, **kwargs: pytest.fail(
                "incomplete Formal results must not publish routes"
            ),
        )
    )

    result = service.run(_request(preset="formal", case_ids=("launch_s64_fp16",)))

    assert result.outcome == "formal_failed"
    assert result.exit_code == 1
    assert result.stage == "formal"


def test_calibration_stops_before_planning_when_probe_fails(tmp_path: Path) -> None:
    events: list[CalibrationEvent] = []

    service = CalibrationService(
        CalibrationDependencies(
            run_probe=lambda *args, **kwargs: (
                {"outcome": "runtime_error"},
                tmp_path / "probe.json",
            ),
            build_plan=lambda *args, **kwargs: pytest.fail(
                "failed probe must not build a plan"
            ),
            run_tuning=lambda *args, **kwargs: pytest.fail(
                "failed probe must not run workloads"
            ),
        )
    )

    result = service.run(
        _request(case_ids=("launch_s64_fp16",)),
        on_event=events.append,
    )

    assert result.outcome == "probe_failed"
    assert result.exit_code == 1
    assert [event.kind for event in events] == ["probe_started", "probe_completed"]


def test_cooperative_cancellation_retains_one_compact_checkpoint(
    tmp_path: Path,
) -> None:
    session_id = f"cancel-{tmp_path.name}"
    token = CancellationToken()
    events: list[CalibrationEvent] = []

    def fake_tuning(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args
        assert kwargs["cancellation_token"] is token
        token.cancel()
        return {
            "tuning_id": "cancelled-summary",
            "observations": [{"outcome": "cancelled"}],
        }

    service = CalibrationService(
        CalibrationDependencies(
            run_probe=lambda *args, **kwargs: (
                routing_probe_result(),
                tmp_path / "probe.json",
            ),
            build_plan=lambda *args, **kwargs: {
                "source": "hardware_cost_model",
                "candidate_order": ["eager-auto"],
            },
            run_tuning=fake_tuning,
            find_verified_route=lambda *args, **kwargs: None,
            implementation_hash=lambda *args, **kwargs: "fixture-implementation",
        )
    )

    result = service.run(
        _request(
            session_id=session_id,
            case_ids=("launch_s64_fp16",),
        ),
        on_event=events.append,
        cancellation_token=token,
    )

    try:
        assert result.outcome == "cancelled"
        assert result.exit_code == 130
        assert result.session_id == session_id
        assert result.checkpoint_path is not None
        checkpoint = json.loads(result.checkpoint_path.read_text(encoding="utf-8"))
        assert checkpoint == {
            "schema_version": 1,
            "session_id": session_id,
            "status": "cancelled",
            "stage": "smoke",
            "active_case_id": "launch_s64_fp16",
            "workload": {
                "set_id": WORKLOAD_SET_ID,
                "sha256": checkpoint["workload"]["sha256"],
            },
            "solution_implementation_sha256": "fixture-implementation",
            "case_ids": ["launch_s64_fp16"],
            "completed_summary_ids": ["cancelled-summary"],
            "outcome": "cancelled",
            "updated_at": checkpoint["updated_at"],
        }
        assert len(checkpoint["workload"]["sha256"]) == 64
        assert all(event.session_id == session_id for event in events)
        json.dumps([event.as_dict() for event in events], allow_nan=False)
        json.dumps(result.as_dict(), allow_nan=False)
    finally:
        if result.checkpoint_path is not None:
            result.checkpoint_path.unlink(missing_ok=True)


def test_completed_calibration_removes_transient_checkpoint(tmp_path: Path) -> None:
    session_id = f"complete-{tmp_path.name}"
    service = CalibrationService(
        CalibrationDependencies(
            run_probe=lambda *args, **kwargs: (
                routing_probe_result(),
                tmp_path / "probe.json",
            ),
            build_plan=lambda *args, **kwargs: {
                "source": "hardware_cost_model",
                "candidate_order": ["eager-auto"],
            },
            run_tuning=lambda *args, **kwargs: staged_tuning_summary(
                kwargs["case"],
                ["eager-auto"],
                kwargs["base_protocol"].preset,
                tmp_path,
            ),
            find_verified_route=lambda *args, **kwargs: None,
            implementation_hash=lambda *args, **kwargs: "fixture-implementation",
        )
    )

    result = service.run(
        _request(
            session_id=session_id,
            case_ids=("launch_s64_fp16",),
        )
    )

    checkpoint_path = (
        PROJECT_ROOT / "results" / "calibration" / f"{session_id}.json"
    )
    assert result.outcome == "smoke_complete"
    assert result.session_id == session_id
    assert result.checkpoint_path is None
    assert not checkpoint_path.exists()
