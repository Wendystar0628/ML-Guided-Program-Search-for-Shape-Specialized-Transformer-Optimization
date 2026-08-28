from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from runner import calibration
from runner.calibration import (
    CalibrationDependencies,
    CalibrationRequest,
    CalibrationService,
)
from runner.contracts import ContractError, RunVariant, load_workload_set
from tests.support.routing_fixtures import routing_probe_result
from tests.support.runner_fixtures import PROJECT_ROOT, WORKLOAD_SET_ID


def _service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    planned: list[tuple[str, RunVariant]],
    *,
    plan_document: dict[str, Any] | None = None,
) -> CalibrationService:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    monkeypatch.setattr(calibration, "load_workload_set", lambda *_args: workload)

    def run_probe(*_args: Any, **_kwargs: Any):
        return routing_probe_result(), tmp_path / "probe.json"

    def build_plan(shape: Any, variant: RunVariant, *_args: Any, **_kwargs: Any):
        planned.append((shape.case_id, variant))
        return plan_document or {
            "source": "fixture",
            "candidate_order": ["eager-auto", "graph"],
            "feasibility": {"baseline_executable": True},
        }

    return CalibrationService(
        CalibrationDependencies(
            run_probe=run_probe,
            build_plan=build_plan,
            run_tuning=lambda *_args, **_kwargs: pytest.fail(
                "plan-only calibration must not tune"
            ),
            find_verified_route=lambda *_args, **_kwargs: None,
            promote=lambda *_args, **_kwargs: pytest.fail(
                "plan-only calibration must not promote"
            ),
            implementation_hash=lambda _path: "fixture-solution",
        )
    )


def test_plan_only_defaults_to_official_01_through_13(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned: list[tuple[str, RunVariant]] = []
    result = _service(tmp_path, monkeypatch, planned).run(
        CalibrationRequest(
            project_root=tmp_path,
            workload_set_id=WORKLOAD_SET_ID,
            plan_only=True,
            session_id="plan-default",
        )
    )

    assert result.outcome == "planned"
    assert result.case_ids == tuple(f"official_{index:02d}" for index in range(1, 14))
    assert [case_id for case_id, _variant in planned] == list(result.case_ids)
    assert not (tmp_path / "results" / "calibration" / "plan-default.json").exists()


def test_variant_is_forwarded_explicitly_to_each_routing_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned: list[tuple[str, RunVariant]] = []
    variant = RunVariant(dtype="float16")

    result = _service(tmp_path, monkeypatch, planned).run(
        CalibrationRequest(
            project_root=tmp_path,
            workload_set_id=WORKLOAD_SET_ID,
            case_ids=("official_02", "official_08"),
            variant=variant,
            plan_only=True,
            session_id="plan-selected",
        )
    )

    assert result.case_ids == ("official_02", "official_08")
    assert planned == [("official_02", variant), ("official_08", variant)]


def test_duplicate_shape_ids_are_rejected_before_probe(tmp_path: Path) -> None:
    request = CalibrationRequest(
        project_root=tmp_path,
        workload_set_id=WORKLOAD_SET_ID,
        case_ids=("official_02", "official_02"),
        plan_only=True,
    )

    with pytest.raises(ContractError, match="duplicates"):
        CalibrationService().run(request)


def test_explicit_official_14_is_rejected_before_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)
    monkeypatch.setattr(calibration, "load_workload_set", lambda *_args: workload)
    probe_called = False

    def run_probe(*_args: Any, **_kwargs: Any):
        nonlocal probe_called
        probe_called = True
        pytest.fail("resource guard must reject before the hardware probe")

    service = CalibrationService(
        CalibrationDependencies(
            run_probe=run_probe,
            implementation_hash=lambda _path: "fixture-solution",
        )
    )

    with pytest.raises(ContractError, match="official_14"):
        service.run(
            CalibrationRequest(
                project_root=tmp_path,
                workload_set_id=WORKLOAD_SET_ID,
                case_ids=("official_14",),
                plan_only=True,
                session_id="guarded-shape",
            )
        )

    assert not probe_called
    assert not (tmp_path / "results" / "calibration").exists()


def test_formal_deployment_keeps_the_official_runtime_precision_contract() -> None:
    request = CalibrationRequest(
        project_root=Path("."),
        workload_set_id=WORKLOAD_SET_ID,
        preset="formal",
        matmul_precision="medium",
    )

    with pytest.raises(ContractError, match="matmul-precision high"):
        CalibrationService._validate_request(request)


def test_probe_profile_uses_only_nested_hardware_and_runtime_policy() -> None:
    result = routing_probe_result()
    result["hardware_profile"] = {
        "device_type": "cpu",
        "gpu": {"name": "legacy-top-level"},
    }
    result["environment"] = {
        "device": "cpu",
        "gpu": "legacy-environment",
        "driver": "legacy-driver",
    }

    profile = calibration.hardware_profile_from_probe(result)

    assert profile["device_type"] == "cuda"
    assert profile["device_name"] == "Fixture GPU"
    assert profile["driver"] == "fixture-driver"
    assert profile["matmul_precision"] == "high"
    assert profile["allow_tf32"] is True
    assert "triton" not in profile


def test_incumbent_lookup_uses_the_shared_exact_route_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_path = tmp_path / "routes.json"
    route_path.write_text("{}", encoding="utf-8")
    captured: dict[str, Any] = {}

    monkeypatch.setattr(calibration, "load_route_table", lambda _path: object())

    def resolve(_table: object, key: dict[str, object]) -> SimpleNamespace:
        captured.update(key)
        return SimpleNamespace(origin="calibrated", policy="auto")

    monkeypatch.setattr(calibration, "resolve_route_result", resolve)
    monkeypatch.setattr(
        calibration,
        "deployable_candidate_id_for_policy",
        lambda *_args: "eager-auto",
    )
    shape = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID).shapes[1]
    profile = calibration.hardware_profile_from_probe(routing_probe_result())

    incumbent = CalibrationService()._incumbent_candidate_id(
        shape,
        RunVariant(),
        profile,
        route_path,
    )

    assert incumbent == "eager-auto"
    assert captured["driver"] == "fixture-driver"
    assert captured["matmul_precision"] == "high"
    assert captured["allow_tf32"] is True
    assert "triton" not in captured


def test_infeasible_plan_is_visible_without_inventing_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = {
        "source": "fixture",
        "candidate_order": [],
        "feasibility": {
            "baseline_executable": False,
            "rejection_reason": "estimated peak exceeds the device safety limit",
        },
    }
    planned: list[tuple[str, RunVariant]] = []
    service = _service(
        tmp_path,
        monkeypatch,
        planned,
        plan_document=plan,
    )

    preview = service.run(
        CalibrationRequest(
            project_root=tmp_path,
            workload_set_id=WORKLOAD_SET_ID,
            case_ids=("official_02",),
            plan_only=True,
            session_id="unsupported-preview",
        )
    )
    result = service.run(
        CalibrationRequest(
            project_root=tmp_path,
            workload_set_id=WORKLOAD_SET_ID,
            case_ids=("official_02",),
            session_id="unsupported-run",
        )
    )

    assert preview.outcome == "planned"
    assert preview.exit_code == 0
    assert preview.smoke_plans[0]["candidate_order"] == []
    assert preview.smoke_plans[0]["decision_scope"] == "unsupported"
    assert preview.smoke_plans[0]["requires_full_workload_measurement"] is False
    assert result.outcome == "unsupported"
    assert result.exit_code == 1
    assert result.stage == "planning"
    assert "estimated peak" in str(result.message)
