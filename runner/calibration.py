"""Reusable cold-start calibration workflow.

The command-line interface and future agents call the same service.  The
service owns orchestration and returns structured state; callers decide how to
present progress events.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from project_identity import solution_implementation_hash
from route_contracts import load_route_table, resolve_route_result
from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    RunVariant,
    TransformerShape,
    WorkloadSet,
    load_workload_set,
    new_run_id,
    select_transformer_shape,
    utc_now,
)
from runner.hardware_router import build_routing_plan
from runner.locking import device_measurement_lease
from runner.resource_guard import (
    ensure_local_benchmark_allowed,
    local_benchmark_shapes,
)
from runner.route_promotion import (
    auto_promote_calibration,
    find_matching_verified_route,
    verified_profile_from_probe_result,
)
from runner.routing_contracts import (
    exact_route_key,
    hardware_identity_from_flat_profile,
    validate_selected_route_groups,
)
from runner.supervisor import CancellationToken, run_managed_probe
from runner.tuning import (
    align_shared_smoke_plans,
    build_formal_candidate_plans,
    candidates_for_shape,
    deployable_candidate_id_for_policy,
    is_deployable_candidate,
    run_tuning_case,
    select_candidates,
)

ProgressCallback = Callable[["CalibrationEvent"], None]


@dataclass(frozen=True)
class CalibrationRequest:
    """Inputs for one hardware-aware calibration run."""

    project_root: Path
    workload_set_id: str
    variant: RunVariant = field(default_factory=RunVariant)
    case_ids: tuple[str, ...] = ()
    device: str = "cuda:0"
    preset: str = "smoke"
    timeout_seconds: float | None = None
    candidate_limit: int = 3
    plan_only: bool = False
    matmul_precision: str = "high"
    allow_tf32: bool = True
    session_id: str | None = None


@dataclass(frozen=True)
class CalibrationEvent:
    """One structured progress update emitted by :class:`CalibrationService`."""

    kind: str
    case_id: str | None = None
    stage: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict)
    session_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable progress event for a CLI or agent."""

        return {key: _json_value(value) for key, value in vars(self).items()}


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
    session_id: str | None = None
    checkpoint_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the terminal state using only JSON-compatible values."""

        return {key: _json_value(value) for key, value in vars(self).items()}


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


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return _json_value(as_dict())
    return str(value)


def _replace_json(path: Path, document: Mapping[str, Any]) -> None:
    """Atomically replace one mutable calibration checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        _json_value(document),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


class _CalibrationCheckpoint:
    """Small mutable snapshot used to inspect an interrupted calibration."""

    def __init__(
        self,
        project_root: Path,
        session_id: str,
        workload_set_id: str,
        solution_sha256: str | None,
    ) -> None:
        self.path = (
            project_root.resolve() / "results" / "calibration" / f"{session_id}.json"
        )
        if self.path.exists():
            raise ContractError(f"calibration session already exists: {session_id}")
        self._document: dict[str, Any] = {
            "schema_version": 1,
            "session_id": session_id,
            "status": "running",
            "stage": "starting",
            "active_case_id": None,
            "workload": {
                "set_id": workload_set_id,
                "sha256": None,
            },
            "solution_implementation_sha256": solution_sha256,
            "case_ids": [],
            "completed_summary_ids": [],
            "outcome": None,
        }
        self._publish()

    @property
    def stage(self) -> str:
        return str(self._document["stage"])

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self._document["case_ids"])

    def discard(self) -> None:
        self.path.unlink(missing_ok=True)

    def configure(self, *, workload_sha256: str, case_ids: Sequence[str]) -> None:
        self._document["workload"] = {
            "set_id": self._document["workload"]["set_id"],
            "sha256": workload_sha256,
        }
        self._document["case_ids"] = list(case_ids)
        self._publish()

    def enter(self, stage: str, case_id: str | None = None) -> None:
        self._document["stage"] = stage
        self._document["active_case_id"] = case_id
        self._publish()

    def record_summary(
        self,
        *,
        summary: Mapping[str, Any],
    ) -> None:
        summary_id = summary.get("tuning_id")
        if isinstance(summary_id, str) and summary_id:
            completed = self._document["completed_summary_ids"]
            if summary_id not in completed:
                completed.append(summary_id)
        observations = summary.get("observations")
        cancelled = isinstance(observations, list) and any(
            isinstance(item, Mapping) and item.get("outcome") == "cancelled"
            for item in observations
        )
        if not cancelled:
            self._document["active_case_id"] = None
        self._publish()

    def finish(self, result: CalibrationResult) -> bool:
        if result.outcome == "cancelled":
            self._document.update(
                {
                    "status": "cancelled",
                    "stage": result.stage,
                    "outcome": result.outcome,
                }
            )
            self._publish()
            return True
        self.path.unlink(missing_ok=True)
        return False

    def _publish(self) -> None:
        self._document["updated_at"] = utc_now()
        _replace_json(self.path, self._document)


def hardware_profile_from_probe(result: Mapping[str, Any]) -> dict[str, Any]:
    """Build the compact flat profile consumed by the routing prior."""

    probe = result.get("probe")
    if not isinstance(probe, Mapping):
        raise ContractError("successful routing probe is missing its probe payload")
    hardware_profile = probe.get("hardware_profile")
    if not isinstance(hardware_profile, Mapping):
        raise ContractError("routing probe is missing hardware_profile")
    runtime_policy = probe.get("runtime_policy")
    if not isinstance(runtime_policy, Mapping):
        raise ContractError("routing probe is missing runtime_policy")

    profile: dict[str, Any] = {}
    device_type = hardware_profile.get("device_type")
    if isinstance(device_type, str):
        profile["device_type"] = device_type
    gpu = hardware_profile.get("gpu")
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
            "theoretical_memory_bandwidth_gbps": ("theoretical_memory_bandwidth_gbps"),
        }
        for source_name, profile_name in gpu_fields.items():
            value = gpu.get(source_name)
            if value is not None:
                profile[profile_name] = value
    software = hardware_profile.get("software")
    if isinstance(software, Mapping):
        for name in ("driver", "torch", "cuda_runtime"):
            value = software.get(name)
            if value is not None:
                profile[name] = value
    platform_profile = hardware_profile.get("platform")
    if isinstance(platform_profile, Mapping):
        system = platform_profile.get("system")
        if system is not None:
            profile["platform_system"] = system

    matmul_precision = runtime_policy.get("matmul_precision")
    allow_tf32 = runtime_policy.get("allow_tf32")
    if matmul_precision not in {"highest", "high", "medium"}:
        raise ContractError("routing runtime_policy has invalid matmul_precision")
    if not isinstance(allow_tf32, bool):
        raise ContractError("routing runtime_policy has invalid allow_tf32")
    profile["matmul_precision"] = matmul_precision
    profile["allow_tf32"] = allow_tf32

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
        cancellation_token: CancellationToken | None = None,
    ) -> CalibrationResult:
        self._validate_request(request)
        if len(request.case_ids) != len(set(request.case_ids)):
            raise ContractError("calibration case_ids must not contain duplicates")
        workload_set = load_workload_set(
            request.project_root,
            request.workload_set_id,
        )
        shapes = (
            tuple(
                select_transformer_shape(workload_set, case_id)
                for case_id in request.case_ids
            )
            if request.case_ids
            else local_benchmark_shapes(workload_set.shapes)
        )
        for shape in shapes:
            ensure_local_benchmark_allowed(shape)
        session_id = request.session_id or new_run_id()
        token = cancellation_token or CancellationToken()
        solution_root = request.project_root / "solution"
        implementation_sha256 = (
            self._dependencies.implementation_hash(solution_root)
            if solution_root.is_dir()
            else None
        )
        checkpoint = _CalibrationCheckpoint(
            request.project_root,
            session_id,
            request.workload_set_id,
            implementation_sha256,
        )

        def emit_with_session(event: CalibrationEvent) -> None:
            if on_event is not None:
                on_event(replace(event, session_id=session_id))

        try:
            with device_measurement_lease(
                request.project_root,
                request.device,
                purpose="calibration",
            ):
                result = self._run(
                    request,
                    workload_set=workload_set,
                    shapes=shapes,
                    on_event=emit_with_session,
                    cancellation_token=token,
                    checkpoint=checkpoint,
                )
        except ContractError:
            checkpoint.discard()
            raise
        except KeyboardInterrupt:
            token.cancel()
            result = CalibrationResult(
                outcome="cancelled",
                exit_code=130,
                stage=checkpoint.stage,
                workload_set_id=request.workload_set_id,
                case_ids=checkpoint.case_ids or request.case_ids,
                message="calibration was interrupted by the user",
            )
        result = replace(result, session_id=session_id)
        checkpoint_retained = checkpoint.finish(result)
        return replace(
            result,
            checkpoint_path=checkpoint.path if checkpoint_retained else None,
        )

    def _run(
        self,
        request: CalibrationRequest,
        *,
        workload_set: WorkloadSet,
        shapes: tuple[TransformerShape, ...],
        on_event: ProgressCallback | None,
        cancellation_token: CancellationToken,
        checkpoint: _CalibrationCheckpoint,
    ) -> CalibrationResult:
        case_ids = tuple(shape.case_id for shape in shapes)
        local_shapes = local_benchmark_shapes(workload_set.shapes)
        full_case_ids = [shape.case_id for shape in local_shapes]
        checkpoint.configure(
            workload_sha256=workload_set.sha256,
            case_ids=case_ids,
        )
        if cancellation_token.is_cancelled:
            self._emit(on_event, "cancellation_observed", stage="starting")
            return CalibrationResult(
                outcome="cancelled",
                exit_code=130,
                stage="starting",
                workload_set_id=request.workload_set_id,
                case_ids=case_ids,
            )

        checkpoint.enter("probe")
        probe_context, probe_code, probe_result, probe_result_path = self._probe(
            request,
            on_event,
            cancellation_token,
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
        checkpoint.enter("planning")

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
            self._incumbent_candidate_id(
                shape,
                request.variant,
                hardware_profile,
                existing_route,
            )
            for shape in shapes
        ]
        smoke_plans = [
            self._routing_plan_for_shape(
                shape,
                request.variant,
                hardware_profile,
                request.candidate_limit,
                incumbent_candidate_id=incumbent,
            )
            for shape, incumbent in zip(shapes, incumbents, strict=True)
        ]
        unsupported_plans = [
            (shape, plan)
            for shape, plan in zip(shapes, smoke_plans, strict=True)
            if not plan["candidate_order"]
        ]
        if not unsupported_plans:
            smoke_plans = list(
                align_shared_smoke_plans(
                    shapes,
                    request.variant,
                    smoke_plans,
                    incumbents,
                    candidate_limit=request.candidate_limit,
                )
            )
        for shape, plan in zip(shapes, smoke_plans, strict=True):
            self._emit(
                on_event,
                "routing_plan_ready",
                case_id=shape.case_id,
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
        if cancellation_token.is_cancelled:
            self._emit(on_event, "cancellation_observed", stage="planning")
            return CalibrationResult(
                outcome="cancelled",
                exit_code=130,
                stage="planning",
                **result_base,
            )
        if request.plan_only:
            self._emit(on_event, "plan_only_completed")
            return CalibrationResult(
                outcome="planned",
                exit_code=0,
                stage="planning",
                **result_base,
            )
        if unsupported_plans:
            unsupported_details = []
            for shape, plan in unsupported_plans:
                feasibility = plan.get("feasibility")
                reason = (
                    feasibility.get("rejection_reason")
                    if isinstance(feasibility, Mapping)
                    else None
                )
                unsupported_details.append(
                    f"{shape.case_id}: {reason or 'no executable candidates'}"
                )
            message = "unsupported workload feasibility: " + "; ".join(
                unsupported_details
            )
            self._emit(
                on_event,
                "planning_unsupported",
                message=message,
                case_ids=[shape.case_id for shape, _plan in unsupported_plans],
            )
            return CalibrationResult(
                outcome="unsupported",
                exit_code=1,
                stage="planning",
                message=message,
                **result_base,
            )

        if request.preset == "formal":
            try:
                validate_selected_route_groups(
                    list(case_ids),
                    local_shapes,
                    request.variant,
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
            shapes,
            request.variant,
            smoke_plans,
            workload_set.sha256,
            hardware_profile,
            smoke_protocol,
            "smoke",
            on_event,
            cancellation_token,
            checkpoint,
        )
        smoke_result_base = {
            **result_base,
            "smoke_summaries": tuple(smoke_summaries),
        }
        if cancellation_token.is_cancelled or self._was_cancelled(smoke_summaries):
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

        checkpoint.enter("formal_selection")
        try:
            formal_plans = list(
                build_formal_candidate_plans(
                    shapes,
                    request.variant,
                    smoke_summaries,
                    incumbents,
                )
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
            shapes=shapes,
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
            shapes,
            request.variant,
            formal_plans,
            workload_set.sha256,
            hardware_profile,
            formal_protocol,
            "formal",
            on_event,
            cancellation_token,
            checkpoint,
        )
        formal_result_base = {
            **smoke_result_base,
            "formal_plans": tuple(formal_plans),
            "formal_summaries": tuple(formal_summaries),
        }
        if cancellation_token.is_cancelled or self._was_cancelled(formal_summaries):
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

        if cancellation_token.is_cancelled:
            self._emit(on_event, "cancellation_observed", stage="promotion")
            return CalibrationResult(
                outcome="cancelled",
                exit_code=130,
                stage="promotion",
                **formal_result_base,
            )
        checkpoint.enter("promotion")
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
                full_workload_shape_ids=full_case_ids,
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
        elif previous_route_bytes is None:
            route_action = "replaced stale verified package"
        elif route_changed:
            route_action = "updated verified package"
        else:
            route_action = "verified package already has the selected routes"
        self._emit(
            on_event,
            "promotion_completed",
            shapes=shapes,
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
        if not isinstance(request.variant, RunVariant):
            raise ContractError("calibration variant must be a RunVariant")
        request.variant.validate()
        if request.preset not in {"smoke", "formal"}:
            raise ContractError(f"unsupported calibration preset: {request.preset}")
        if request.candidate_limit <= 0:
            raise ContractError("candidate-limit must be positive")
        if request.session_id is not None:
            allowed = set(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            )
            if (
                not request.session_id
                or len(request.session_id) > 128
                or request.session_id in {".", ".."}
                or any(character not in allowed for character in request.session_id)
            ):
                raise ContractError(
                    "session_id must use 1-128 letters, digits, dots, dashes, "
                    "or underscores"
                )
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
        cancellation_token: CancellationToken,
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
            cancellation_token=cancellation_token,
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
        shape: TransformerShape,
        variant: RunVariant,
        hardware_profile: Mapping[str, Any],
        route_path: Path | None,
    ) -> str | None:
        if route_path is None or not route_path.is_file():
            return None
        table = load_route_table(route_path)
        try:
            key = exact_route_key(
                shape,
                variant,
                hardware_identity_from_flat_profile(hardware_profile),
            )
        except (TypeError, ValueError) as exc:
            raise ContractError(
                f"cannot resolve current route for {shape.case_id}: {exc}"
            ) from exc
        resolution = resolve_route_result(table, key)
        if resolution.origin != "calibrated":
            return None
        candidate_id = deployable_candidate_id_for_policy(
            shape,
            variant,
            resolution.policy,
        )
        if candidate_id is None:
            raise ContractError(
                f"current route {resolution.policy!r} for {shape.case_id} has no "
                "deployable calibration candidate"
            )
        return candidate_id

    def _routing_plan_for_shape(
        self,
        shape: TransformerShape,
        variant: RunVariant,
        hardware_profile: Mapping[str, Any],
        candidate_limit: int,
        *,
        incumbent_candidate_id: str | None,
    ) -> dict[str, Any]:
        applicable = tuple(
            item.candidate_id
            for item in candidates_for_shape(shape, variant)
            if is_deployable_candidate(item)
        )
        try:
            options: dict[str, Any] = {}
            if incumbent_candidate_id is not None:
                options["required_candidate_ids"] = (incumbent_candidate_id,)
            raw_plan = self._dependencies.build_plan(
                shape,
                variant,
                hardware_profile,
                applicable,
                limit=candidate_limit,
                **options,
            )
        except (TypeError, ValueError) as exc:
            raise ContractError(
                f"unable to build routing plan for {shape.case_id}: {exc}"
            ) from exc
        if not isinstance(raw_plan, Mapping):
            raise ContractError(f"routing plan for {shape.case_id} must be an object")
        raw_order = raw_plan.get("candidate_order")
        if (
            not isinstance(raw_order, Sequence)
            or isinstance(raw_order, (str, bytes))
            or any(not isinstance(value, str) for value in raw_order)
        ):
            raise ContractError(
                f"routing plan for {shape.case_id} has no valid candidate order"
            )
        candidate_order = list(raw_order)
        feasibility = raw_plan.get("feasibility")
        if not candidate_order:
            if (
                not isinstance(feasibility, Mapping)
                or feasibility.get("baseline_executable") is not False
            ):
                raise ContractError(
                    f"routing plan for {shape.case_id} has no candidates without "
                    "an unsupported feasibility decision"
                )
            plan = dict(raw_plan)
            plan["candidate_order"] = []
            plan["decision_scope"] = "unsupported"
            plan["requires_full_workload_measurement"] = False
            return plan
        if (
            isinstance(feasibility, Mapping)
            and feasibility.get("baseline_executable") is False
        ):
            raise ContractError(
                f"routing plan for {shape.case_id} ranks candidates for an "
                "unsupported workload"
            )
        if len(candidate_order) > candidate_limit:
            raise ContractError(
                f"routing plan for {shape.case_id} exceeds candidate-limit"
            )
        if "eager-auto" in applicable and "eager-auto" not in candidate_order:
            raise ContractError(
                f"routing plan for {shape.case_id} must retain eager-auto"
            )
        if (
            incumbent_candidate_id is not None
            and incumbent_candidate_id not in candidate_order
        ):
            raise ContractError(
                f"routing plan for {shape.case_id} must retain the current incumbent"
            )
        select_candidates(shape, variant, candidate_order)
        plan = dict(raw_plan)
        plan["candidate_order"] = candidate_order
        plan["decision_scope"] = "candidate_order_only"
        plan["requires_full_workload_measurement"] = True
        return plan

    def _run_stage(
        self,
        request: CalibrationRequest,
        shapes: list[TransformerShape],
        variant: RunVariant,
        plans: list[Mapping[str, Any]],
        workload_sha256: str,
        hardware_profile: Mapping[str, Any],
        protocol: MeasurementProtocol,
        stage: str,
        on_event: ProgressCallback | None,
        cancellation_token: CancellationToken,
        checkpoint: _CalibrationCheckpoint,
    ) -> list[dict[str, Any]]:
        checkpoint.enter(stage)
        self._emit(on_event, "stage_started", stage=stage)
        summaries: list[dict[str, Any]] = []
        for shape, plan in zip(shapes, plans, strict=True):
            if cancellation_token.is_cancelled:
                checkpoint.enter(stage, shape.case_id)
                self._emit(
                    on_event,
                    "cancellation_observed",
                    case_id=shape.case_id,
                    stage=stage,
                )
                break
            checkpoint.enter(stage, shape.case_id)
            summary = self._dependencies.run_tuning(
                request.project_root,
                workload_set_id=request.workload_set_id,
                workload_sha256=workload_sha256,
                shape=shape,
                variant=variant,
                base_protocol=protocol,
                device=request.device,
                requested_candidates=plan["candidate_order"],
                routing_plan=plan,
                device_profile=hardware_profile,
                cancellation_token=cancellation_token,
            )
            summaries.append(summary)
            checkpoint.record_summary(summary=summary)
            self._emit(
                on_event,
                "tuning_completed",
                case_id=shape.case_id,
                stage=stage,
                summary=summary,
            )
            if self._summary_cancelled(summary):
                break
        if not cancellation_token.is_cancelled and not self._was_cancelled(summaries):
            checkpoint.enter(stage)
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
