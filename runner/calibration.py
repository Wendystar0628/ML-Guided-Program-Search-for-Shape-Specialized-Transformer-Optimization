"""Reusable cold-start calibration workflow.

The command-line interface and future agents call the same service.  The
service owns orchestration and returns structured state; callers decide how to
present progress events.
"""

from __future__ import annotations

import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    WorkloadCase,
    load_workload_set,
    select_workload_case,
    solution_implementation_hash,
)
from runner.hardware_router import build_routing_plan
from runner.route_promotion import (
    auto_promote_calibration,
    find_matching_verified_route,
    verified_profile_from_probe_result,
)
from runner.routing_contracts import validate_selected_route_groups
from runner.supervisor import run_managed_probe
from runner.tuning import (
    align_shared_smoke_plans,
    build_formal_candidate_plans,
    candidates_for_case,
    deployable_candidate_id_for_policy,
    is_deployable_candidate,
    run_tuning_case,
    select_candidates,
)
from solution.dispatch import load_route_table, make_route_key, resolve_route_result

ProgressCallback = Callable[["CalibrationEvent"], None]


@dataclass(frozen=True)
class CalibrationRequest:
    """Inputs for one hardware-aware calibration run."""

    project_root: Path
    workload_set_id: str
    case_ids: tuple[str, ...] = ()
    device: str = "cuda:0"
    preset: str = "smoke"
    timeout_seconds: float | None = None
    candidate_limit: int = 3
    plan_only: bool = False
    matmul_precision: str = "high"
    allow_tf32: bool = True


@dataclass(frozen=True)
class CalibrationEvent:
    """One structured progress update emitted by :class:`CalibrationService`."""

    kind: str
    case_id: str | None = None
    stage: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationResult:
    """Structured terminal state for a calibration run."""

    outcome: str
    exit_code: int
    stage: str
    workload_set_id: str
    case_ids: tuple[str, ...]
    hardware_profile: Mapping[str, Any] | None = None
    probe_result: Mapping[str, Any] | None = None
    probe_result_path: Path | None = None
    smoke_plans: tuple[Mapping[str, Any], ...] = ()
    smoke_summaries: tuple[Mapping[str, Any], ...] = ()
    formal_plans: tuple[Mapping[str, Any], ...] = ()
    formal_summaries: tuple[Mapping[str, Any], ...] = ()
    deployed_winners: tuple[Mapping[str, Any], ...] = ()
    route_path: Path | None = None
    route_action: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class CalibrationDependencies:
    """Replaceable process and persistence boundaries for calibration tests."""

    run_probe: Callable[..., Any] = run_managed_probe
    build_plan: Callable[..., Any] = build_routing_plan
    run_tuning: Callable[..., Any] = run_tuning_case
    find_verified_route: Callable[..., Any] = find_matching_verified_route
    promote: Callable[..., Any] = auto_promote_calibration
    implementation_hash: Callable[..., Any] = solution_implementation_hash


@dataclass(frozen=True)
class _RoutingProbeContext:
    hardware_profile: dict[str, Any]
    raw_result: dict[str, Any]
    result_path: Path


def hardware_profile_from_probe(result: Mapping[str, Any]) -> dict[str, Any]:
    """Build the compact flat profile consumed by the routing prior."""

    environment = result.get("environment")
    probe = result.get("probe")
    if not isinstance(environment, Mapping) or not isinstance(probe, Mapping):
        raise ContractError("successful routing probe is missing device details")

    profile: dict[str, Any] = {}
    for candidate in (
        result.get("hardware_profile"),
        probe.get("hardware_profile"),
    ):
        if not isinstance(candidate, Mapping):
            continue
        device_type = candidate.get("device_type")
        if isinstance(device_type, str):
            profile["device_type"] = device_type
        gpu = candidate.get("gpu")
        if isinstance(gpu, Mapping):
            gpu_fields = {
                "name": "device_name",
                "compute_capability": "compute_capability",
                "architecture_family": "architecture_family",
                "bf16_supported": "bf16_supported",
                "cuda_graph_available": "cuda_graph_available",
                "total_memory_bytes": "total_memory_bytes",
                "sm_count": "sm_count",
                "l2_cache_bytes": "l2_cache_bytes",
                "shared_memory_per_sm_bytes": "shared_memory_per_sm_bytes",
                "registers_per_sm": "registers_per_sm",
                "memory_bus_width_bits": "memory_bus_width_bits",
                "memory_clock_rate_khz": "memory_clock_khz",
                "theoretical_memory_bandwidth_gbps": (
                    "theoretical_memory_bandwidth_gbps"
                ),
            }
            for source_name, profile_name in gpu_fields.items():
                value = gpu.get(source_name)
                if value is not None:
                    profile[profile_name] = value
        software = candidate.get("software")
        if isinstance(software, Mapping):
            for name in (
                "driver",
                "torch",
                "cuda_runtime",
                "triton",
                "triton_available",
            ):
                value = software.get(name)
                if value is not None:
                    profile[name] = value
        platform_profile = candidate.get("platform")
        if isinstance(platform_profile, Mapping):
            system = platform_profile.get("system")
            if system is not None:
                profile["platform_system"] = system

    resolved_device = environment.get("device")
    if isinstance(resolved_device, str) and resolved_device:
        profile.setdefault("device_type", resolved_device.split(":", maxsplit=1)[0])
    environment_fields = {
        "gpu": "device_name",
        "compute_capability": "compute_capability",
        "total_memory_bytes": "total_memory_bytes",
        "driver": "driver",
        "torch": "torch",
        "cuda_runtime": "cuda_runtime",
    }
    for source_name, profile_name in environment_fields.items():
        value = environment.get(source_name)
        if value is not None:
            profile.setdefault(profile_name, value)
    profile.setdefault("platform_system", platform.system())

    raw_anchors = probe.get("performance_anchors")
    if isinstance(raw_anchors, Mapping):
        anchors: dict[str, Any] = {}
        launch = raw_anchors.get("eager_launch")
        if isinstance(launch, Mapping):
            value = launch.get("effective_latency_us")
            if _is_number(value):
                anchors["launch_latency_us"] = value
        graph = raw_anchors.get("cuda_graph_replay")
        if isinstance(graph, Mapping):
            value = graph.get("effective_latency_per_node_us")
            if value is None:
                value = graph.get("replay_latency_us")
            if _is_number(value):
                anchors["graph_replay_per_node_us"] = value
        device_copy = raw_anchors.get("device_copy")
        if isinstance(device_copy, Mapping):
            value = device_copy.get("effective_bandwidth_gbps")
            if _is_number(value):
                anchors["memory_bandwidth_gbps"] = value
        gemm_tflops: dict[str, Any] = {}
        for source_name, dtype_name in (
            ("gemm_float16", "float16"),
            ("gemm_bfloat16", "bfloat16"),
            ("gemm_float32", "float32"),
        ):
            gemm = raw_anchors.get(source_name)
            if isinstance(gemm, Mapping):
                value = gemm.get("tflops")
                if _is_number(value):
                    gemm_tflops[dtype_name] = value
        if gemm_tflops:
            anchors["gemm_tflops"] = gemm_tflops
        softmax = raw_anchors.get("softmax_fp32")
        if isinstance(softmax, Mapping):
            value = softmax.get("throughput_gigaelements_per_second")
            if _is_number(value):
                anchors["softmax_giga_elements_per_s"] = value
        if anchors:
            profile["performance_anchors"] = anchors

    if not isinstance(profile.get("device_type"), str) or not profile["device_type"]:
        raise ContractError("routing hardware profile is missing device_type")
    if profile["device_type"].lower() == "cuda":
        for name in ("device_name", "compute_capability"):
            if not isinstance(profile.get(name), str) or not profile[name]:
                raise ContractError(f"routing hardware profile is missing {name}")
    return profile


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _exit_code(outcome: object) -> int:
    if outcome == "success":
        return 0
    if outcome == "invalid_output":
        return 2
    if outcome == "cancelled":
        return 130
    return 1


class CalibrationService:
    """Run one complete calibration without depending on CLI state or output."""

    def __init__(self, dependencies: CalibrationDependencies | None = None) -> None:
        self._dependencies = dependencies or CalibrationDependencies()

    def run(
        self,
        request: CalibrationRequest,
        *,
        on_event: ProgressCallback | None = None,
    ) -> CalibrationResult:
        self._validate_request(request)
        if len(request.case_ids) != len(set(request.case_ids)):
            raise ContractError("calibration case_ids must not contain duplicates")
        workload_set = load_workload_set(
            request.project_root,
            request.workload_set_id,
        )
        cases = (
            [
                select_workload_case(workload_set, case_id)
                for case_id in request.case_ids
            ]
            if request.case_ids
            else list(workload_set.cases)
        )
        case_ids = tuple(case.case_id for case in cases)
        full_case_ids = [case.case_id for case in workload_set.cases]

        probe_context, probe_code, probe_result, probe_result_path = self._probe(
            request,
            on_event,
        )
        if probe_context is None:
            return CalibrationResult(
                outcome=("cancelled" if probe_code == 130 else "probe_failed"),
                exit_code=probe_code,
                stage="probe",
                workload_set_id=request.workload_set_id,
                case_ids=case_ids,
                probe_result=probe_result,
                probe_result_path=probe_result_path,
            )
        hardware_profile = probe_context.hardware_profile

        existing_route: Path | None = None
        if str(hardware_profile["device_type"]).lower() == "cuda":
            verified_profile = verified_profile_from_probe_result(
                probe_context.raw_result
            )
            existing_route = self._dependencies.find_verified_route(
                request.project_root,
                verified_profile,
                workload_set_id=request.workload_set_id,
                workload_sha256=workload_set.sha256,
            )
        elif request.preset == "formal" and not request.plan_only:
            raise ContractError("formal calibration deployment requires a CUDA device")

        incumbents = [
            self._incumbent_candidate_id(case, hardware_profile, existing_route)
            for case in cases
        ]
        smoke_plans = [
            self._routing_plan_for_case(
                case,
                hardware_profile,
                request.candidate_limit,
                incumbent_candidate_id=incumbent,
            )
            for case, incumbent in zip(cases, incumbents, strict=True)
        ]
        smoke_plans = list(
            align_shared_smoke_plans(
                cases,
                smoke_plans,
                incumbents,
                candidate_limit=request.candidate_limit,
            )
        )
        for case, plan in zip(cases, smoke_plans, strict=True):
            self._emit(
                on_event,
                "routing_plan_ready",
                case_id=case.case_id,
                plan=plan,
            )

        result_base = {
            "workload_set_id": request.workload_set_id,
            "case_ids": case_ids,
            "hardware_profile": hardware_profile,
            "probe_result": probe_context.raw_result,
            "probe_result_path": probe_context.result_path,
            "smoke_plans": tuple(smoke_plans),
        }
        if request.plan_only:
            self._emit(on_event, "plan_only_completed")
            return CalibrationResult(
                outcome="planned",
                exit_code=0,
                stage="planning",
                **result_base,
            )

        if request.preset == "formal":
            try:
                validate_selected_route_groups(
                    list(case_ids),
                    workload_set.cases,
                )
            except ValueError as exc:
                raise ContractError(str(exc)) from exc
            if existing_route is None and set(case_ids) != set(full_case_ids):
                raise ContractError(
                    "a new verified hardware package requires one complete Formal "
                    "workload calibration"
                )

        smoke_protocol = MeasurementProtocol.for_preset(
            "smoke",
            matmul_precision=request.matmul_precision,
            allow_tf32=request.allow_tf32,
            timeout_seconds=request.timeout_seconds,
        )
        smoke_summaries = self._run_stage(
            request,
            cases,
            smoke_plans,
            workload_set.sha256,
            hardware_profile,
            smoke_protocol,
            "smoke",
            on_event,
        )
        smoke_result_base = {
            **result_base,
            "smoke_summaries": tuple(smoke_summaries),
        }
        if self._was_cancelled(smoke_summaries):
            return CalibrationResult(
                outcome="cancelled",
                exit_code=130,
                stage="smoke",
                **smoke_result_base,
            )
        self._emit(on_event, "stage_outputs", stage="smoke", summaries=smoke_summaries)

        if request.preset == "smoke":
            complete = all(
                summary.get("complete") is True and summary.get("winner") is not None
                for summary in smoke_summaries
            )
            if complete:
                self._emit(on_event, "smoke_screening_only", case_ids=case_ids)
            return CalibrationResult(
                outcome="smoke_complete" if complete else "screening_failed",
                exit_code=0 if complete else 1,
                stage="smoke",
                **smoke_result_base,
            )

        try:
            formal_plans = list(
                build_formal_candidate_plans(cases, smoke_summaries, incumbents)
            )
        except ContractError as exc:
            message = str(exc)
            self._emit(on_event, "formal_selection_skipped", message=message)
            return CalibrationResult(
                outcome="formal_selection_failed",
                exit_code=1,
                stage="formal_selection",
                message=message,
                **smoke_result_base,
            )
        expected_implementation = smoke_summaries[0]["source_implementation_sha256"]
        current_implementation = self._dependencies.implementation_hash(
            request.project_root / "solution"
        )
        if current_implementation != expected_implementation:
            message = "Solution implementation changed after Smoke screening"
            self._emit(on_event, "implementation_changed", message=message)
            return CalibrationResult(
                outcome="source_changed",
                exit_code=1,
                stage="formal_selection",
                formal_plans=tuple(formal_plans),
                message=message,
                **smoke_result_base,
            )

        self._emit(
            on_event,
            "formal_plans_ready",
            cases=cases,
            plans=formal_plans,
        )
        formal_protocol = MeasurementProtocol.for_preset(
            "formal",
            matmul_precision=request.matmul_precision,
            allow_tf32=request.allow_tf32,
            timeout_seconds=request.timeout_seconds,
        )
        formal_summaries = self._run_stage(
            request,
            cases,
            formal_plans,
            workload_set.sha256,
            hardware_profile,
            formal_protocol,
            "formal",
            on_event,
        )
        formal_result_base = {
            **smoke_result_base,
            "formal_plans": tuple(formal_plans),
            "formal_summaries": tuple(formal_summaries),
        }
        if self._was_cancelled(formal_summaries):
            return CalibrationResult(
                outcome="cancelled",
                exit_code=130,
                stage="formal",
                **formal_result_base,
            )
        self._emit(
            on_event,
            "stage_outputs",
            stage="formal",
            summaries=formal_summaries,
        )
        if any(
            summary.get("complete") is not True
            or summary.get("winner") is None
            or summary.get("deployable_winner") is None
            for summary in formal_summaries
        ):
            message = "Formal calibration has no complete deployable winner"
            self._emit(on_event, "promotion_skipped", message=message)
            return CalibrationResult(
                outcome="formal_failed",
                exit_code=1,
                stage="formal",
                message=message,
                **formal_result_base,
            )

        self._emit(on_event, "promotion_started")
        try:
            previous_route_bytes = (
                existing_route.read_bytes()
                if existing_route is not None and existing_route.is_file()
                else None
            )
            _, winners, route_path, created = self._dependencies.promote(
                request.project_root,
                formal_summaries,
                probe_result=probe_context.raw_result,
                full_workload_case_ids=full_case_ids,
            )
        except (ContractError, OSError) as exc:
            message = str(exc)
            self._emit(on_event, "promotion_failed", message=message)
            return CalibrationResult(
                outcome="promotion_failed",
                exit_code=1,
                stage="promotion",
                message=message,
                **formal_result_base,
            )
        route_changed = (
            previous_route_bytes is not None
            and route_path.is_file()
            and route_path.read_bytes() != previous_route_bytes
        )
        if created:
            route_action = "created verified package"
        elif route_changed:
            route_action = "updated verified package"
        else:
            route_action = "verified package already has the selected routes"
        self._emit(
            on_event,
            "promotion_completed",
            cases=cases,
            winners=winners,
            route_path=route_path,
            route_action=route_action,
        )
        return CalibrationResult(
            outcome="formal_promoted",
            exit_code=0,
            stage="promotion",
            deployed_winners=tuple(winners),
            route_path=route_path,
            route_action=route_action,
            **formal_result_base,
        )

    @staticmethod
    def _validate_request(request: CalibrationRequest) -> None:
        if request.preset not in {"smoke", "formal"}:
            raise ContractError(f"unsupported calibration preset: {request.preset}")
        if request.candidate_limit <= 0:
            raise ContractError("candidate-limit must be positive")
        if (
            request.preset == "formal"
            and not request.plan_only
            and (request.matmul_precision != "high" or request.allow_tf32 is not True)
        ):
            raise ContractError(
                "formal calibration deployment requires "
                "--matmul-precision high and --allow-tf32"
            )

    def _probe(
        self,
        request: CalibrationRequest,
        on_event: ProgressCallback | None,
    ) -> tuple[
        _RoutingProbeContext | None,
        int,
        dict[str, Any],
        Path,
    ]:
        self._emit(on_event, "probe_started")
        timeout = (
            request.timeout_seconds if request.timeout_seconds is not None else 30.0
        )
        result, result_path = self._dependencies.run_probe(
            request.project_root,
            device=request.device,
            timeout_seconds=timeout,
            matmul_precision=request.matmul_precision,
            allow_tf32=request.allow_tf32,
            probe_mode="routing",
        )
        self._emit(
            on_event,
            "probe_completed",
            result=result,
            result_path=result_path,
        )
        exit_code = _exit_code(result.get("outcome"))
        if exit_code != 0:
            return None, exit_code, result, result_path
        return (
            _RoutingProbeContext(
                hardware_profile=hardware_profile_from_probe(result),
                raw_result=result,
                result_path=result_path,
            ),
            0,
            result,
            result_path,
        )

    def _incumbent_candidate_id(
        self,
        case: WorkloadCase,
        hardware_profile: Mapping[str, Any],
        route_path: Path | None,
    ) -> str | None:
        if route_path is None or not route_path.is_file():
            return None
        table = load_route_table(route_path)
        key = make_route_key(
            case,
            shape=(case.batch_size, case.seq_len, case.d_model),
            dtype=case.dtype,
            device_type=str(hardware_profile["device_type"]),
            device_name=str(hardware_profile["device_name"]),
            compute_capability=str(hardware_profile["compute_capability"]),
            platform_system=str(hardware_profile["platform_system"]),
            torch_version=str(hardware_profile["torch"]),
            cuda_runtime=str(hardware_profile["cuda_runtime"]),
            triton_version=str(hardware_profile["triton"]),
        )
        resolution = resolve_route_result(table, key)
        if resolution.origin != "calibrated":
            return None
        candidate_id = deployable_candidate_id_for_policy(case, resolution.policy)
        if candidate_id is None:
            raise ContractError(
                f"current route {resolution.policy!r} for {case.case_id} has no "
                "deployable calibration candidate"
            )
        return candidate_id

    def _routing_plan_for_case(
        self,
        case: WorkloadCase,
        hardware_profile: Mapping[str, Any],
        candidate_limit: int,
        *,
        incumbent_candidate_id: str | None,
    ) -> dict[str, Any]:
        applicable = tuple(
            item.candidate_id
            for item in candidates_for_case(case)
            if is_deployable_candidate(item)
        )
        try:
            options: dict[str, Any] = {}
            if incumbent_candidate_id is not None:
                options["required_candidate_ids"] = (incumbent_candidate_id,)
            raw_plan = self._dependencies.build_plan(
                case,
                hardware_profile,
                applicable,
                limit=candidate_limit,
                **options,
            )
        except (TypeError, ValueError) as exc:
            raise ContractError(
                f"unable to build routing plan for {case.case_id}: {exc}"
            ) from exc
        if not isinstance(raw_plan, Mapping):
            raise ContractError(f"routing plan for {case.case_id} must be an object")
        raw_order = raw_plan.get("candidate_order")
        if (
            not isinstance(raw_order, Sequence)
            or isinstance(raw_order, (str, bytes))
            or not raw_order
            or any(not isinstance(value, str) for value in raw_order)
        ):
            raise ContractError(
                f"routing plan for {case.case_id} has no valid candidate order"
            )
        candidate_order = list(raw_order)
        if len(candidate_order) > candidate_limit:
            raise ContractError(
                f"routing plan for {case.case_id} exceeds candidate-limit"
            )
        if "eager-auto" in applicable and "eager-auto" not in candidate_order:
            raise ContractError(
                f"routing plan for {case.case_id} must retain eager-auto"
            )
        if (
            incumbent_candidate_id is not None
            and incumbent_candidate_id not in candidate_order
        ):
            raise ContractError(
                f"routing plan for {case.case_id} must retain the current incumbent"
            )
        select_candidates(case, candidate_order)
        plan = dict(raw_plan)
        plan["candidate_order"] = candidate_order
        plan["decision_scope"] = "candidate_order_only"
        plan["requires_full_workload_measurement"] = True
        return plan

    def _run_stage(
        self,
        request: CalibrationRequest,
        cases: list[WorkloadCase],
        plans: list[Mapping[str, Any]],
        workload_sha256: str,
        hardware_profile: Mapping[str, Any],
        protocol: MeasurementProtocol,
        stage: str,
        on_event: ProgressCallback | None,
    ) -> list[dict[str, Any]]:
        self._emit(on_event, "stage_started", stage=stage)
        summaries: list[dict[str, Any]] = []
        for case, plan in zip(cases, plans, strict=True):
            summary = self._dependencies.run_tuning(
                request.project_root,
                workload_set_id=request.workload_set_id,
                workload_sha256=workload_sha256,
                case=case,
                base_protocol=protocol,
                device=request.device,
                requested_candidates=plan["candidate_order"],
                routing_plan=plan,
                device_profile=hardware_profile,
            )
            summaries.append(summary)
            self._emit(
                on_event,
                "tuning_completed",
                case_id=case.case_id,
                stage=stage,
                summary=summary,
            )
            if self._summary_cancelled(summary):
                break
        return summaries

    @staticmethod
    def _summary_cancelled(summary: Mapping[str, Any]) -> bool:
        observations = summary.get("observations")
        return isinstance(observations, list) and any(
            isinstance(item, Mapping) and item.get("outcome") == "cancelled"
            for item in observations
        )

    @classmethod
    def _was_cancelled(cls, summaries: Sequence[Mapping[str, Any]]) -> bool:
        return any(cls._summary_cancelled(summary) for summary in summaries)

    @staticmethod
    def _emit(
        callback: ProgressCallback | None,
        kind: str,
        *,
        case_id: str | None = None,
        stage: str | None = None,
        **data: Any,
    ) -> None:
        if callback is not None:
            callback(
                CalibrationEvent(
                    kind=kind,
                    case_id=case_id,
                    stage=stage,
                    data=data,
                )
            )
