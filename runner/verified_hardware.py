"""Run one verified hardware bundle through the shared benchmark runner."""

from __future__ import annotations

import argparse
import math
import platform
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runner.candidates import candidate_spec_for_policy
from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    WorkloadCase,
    WorkloadSet,
    atomic_replace_json,
    load_json,
    load_workload_set,
)
from runner.locking import device_measurement_lease
from runner.probe import collect_environment
from runner.routing_contracts import (
    exact_route_key,
    hardware_identity_from_runtime,
    hardware_identity_from_verified_profile,
)
from runner.supervisor import CancellationToken
from runner.sweep import (
    BenchmarkSweepRequest,
    BenchmarkSweepResult,
    BenchmarkSweepService,
)
from solution.dispatch import (
    load_verified_bundle,
    resolve_route_result,
    validate_verified_route_table,
)

_STABLE_HARDWARE_FIELDS = (
    "device_type",
    "device_name",
    "compute_capability",
)


class VerifiedHardwareError(RuntimeError):
    """Raised when a run cannot be attributed to this verified bundle."""


@dataclass(frozen=True)
class BundlePaths:
    """Resolved paths owned by one verified hardware bundle."""

    project_root: Path
    bundle_root: Path
    profile: Path
    routes: Path
    sweeps: Path

    @property
    def manifest(self) -> Path:
        return self.bundle_root / "manifest.json"

    @property
    def reference_formal(self) -> Path:
        return self.bundle_root / "results" / "reference_formal.json"

    @classmethod
    def from_bundle(cls, bundle_root: Path) -> BundlePaths:
        bundle_root = bundle_root.resolve()
        project_root = bundle_root.parents[1]
        return cls(
            project_root=project_root,
            bundle_root=bundle_root,
            profile=bundle_root / "profile.json",
            routes=bundle_root / "routes.json",
            sweeps=bundle_root / "results" / "sweeps",
        )


@dataclass(frozen=True)
class LaunchConfig:
    """User-controlled settings forwarded to the shared runner."""

    device: str = "cuda:0"
    preset: str = "formal"
    timeout: float | None = None


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def expected_hardware_identity(profile: Mapping[str, Any]) -> dict[str, str]:
    """Extract the stable GPU identity owned by one persisted Bundle."""

    try:
        route_identity = hardware_identity_from_verified_profile(profile).as_route_fields()
    except (TypeError, ValueError) as exc:
        raise VerifiedHardwareError(str(exc)) from exc
    return {
        field: route_identity[field] for field in _STABLE_HARDWARE_FIELDS
    }


def collect_runtime_identity(device_name: str) -> dict[str, dict[str, str]]:
    """Collect route identity fields plus useful machine provenance."""

    try:
        import torch
    except Exception as exc:
        raise VerifiedHardwareError(f"PyTorch is unavailable: {exc}") from exc

    try:
        device = torch.device(device_name)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise VerifiedHardwareError(f"invalid device {device_name!r}: {exc}") from exc
    if device.type != "cuda" or not torch.cuda.is_available():
        raise VerifiedHardwareError("verified GPU bundles require CUDA")

    try:
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(index)
    except (AssertionError, RuntimeError, ValueError) as exc:
        raise VerifiedHardwareError(
            f"cannot inspect CUDA device {device_name!r}: {exc}"
        ) from exc
    try:
        import triton
    except Exception:  # noqa: BLE001 - absence is itself a comparable runtime fact.
        triton_version = "unavailable"
    else:
        triton_version = str(getattr(triton, "__version__", "unknown"))
    environment = collect_environment(device)
    driver = environment.get("driver")

    return {
        "gpu": {
            "name": str(properties.name),
            "compute_capability": f"{properties.major}.{properties.minor}",
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "software": {
            "torch": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "triton": triton_version,
            "driver": str(driver) if driver else "unavailable",
        },
    }


def validate_hardware_identity(
    expected: Mapping[str, str],
    actual_runtime: Mapping[str, Any],
) -> None:
    """Require the same GPU while leaving software matching to exact routes."""

    try:
        actual = hardware_identity_from_runtime(actual_runtime).as_route_fields()
    except (TypeError, ValueError) as exc:
        raise VerifiedHardwareError(str(exc)) from exc
    labels = {
        "device_type": "device_type",
        "device_name": "gpu.name",
        "compute_capability": "gpu.compute_capability",
    }
    mismatches = [
        f"{labels[field]}: expected {expected.get(field)!r}, got {actual[field]!r}"
        for field in _STABLE_HARDWARE_FIELDS
        if actual[field] != expected.get(field)
    ]
    if mismatches:
        raise VerifiedHardwareError(
            "GPU does not match the verified hardware profile: "
            + "; ".join(mismatches)
        )


def _portable_source(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _dispatch_source_matches(
    source: object,
    *,
    route_path: Path,
    project_root: Path,
) -> bool:
    if not isinstance(source, str) or not source:
        return False
    if source.replace("\\", "/") == _portable_source(route_path, project_root):
        return True
    candidate = Path(source)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve() == route_path.resolve()


def _route_key(
    case: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, object]:
    try:
        return exact_route_key(case, hardware_identity_from_runtime(identity))
    except (TypeError, ValueError) as exc:
        raise VerifiedHardwareError(str(exc)) from exc


def _expected_route(
    routes: Mapping[str, Any],
    case: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> tuple[str, str]:
    try:
        table = validate_verified_route_table(routes)
    except (TypeError, ValueError) as exc:
        raise VerifiedHardwareError(f"invalid verified route table: {exc}") from exc
    resolution = resolve_route_result(table, _route_key(case, identity))
    return resolution.policy, resolution.origin


def _case_id(run: Mapping[str, Any]) -> str | None:
    workload = run.get("workload")
    case = workload.get("case") if isinstance(workload, Mapping) else None
    value = case.get("case_id") if isinstance(case, Mapping) else None
    return value if isinstance(value, str) and value else None


def validate_workload_route_coverage(
    workload_set: WorkloadSet,
    *,
    routes: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    """Require an explicit verified decision for every package workload."""

    try:
        route_identity = hardware_identity_from_runtime(identity)
        validate_verified_route_table(
            routes,
            expected_identity={
                "device_type": route_identity.device_type,
                "device_name": route_identity.device_name,
                "compute_capability": route_identity.compute_capability,
            },
        )
    except (TypeError, ValueError) as exc:
        raise VerifiedHardwareError(f"invalid verified route table: {exc}") from exc

    missing: list[str] = []
    for case in workload_set.cases:
        case_document = case.as_dict()
        _, origin = _expected_route(routes, case_document, identity)
        if origin != "calibrated":
            missing.append(str(case_document.get("case_id", "<unknown>")))
    if missing:
        raise VerifiedHardwareError(
            "verified route table has no exact decision for: " + ", ".join(missing)
        )


def validate_run_routes(
    runs: Sequence[Mapping[str, Any]],
    *,
    routes: Mapping[str, Any],
    identity: Mapping[str, Any],
    route_path: Path,
    route_sha256: str,
    project_root: Path,
) -> None:
    """Verify that every result used this bundle and its expected exact route."""

    for run in runs:
        case_id = _case_id(run) or "<missing-case-id>"
        workload = run.get("workload")
        case = workload.get("case") if isinstance(workload, Mapping) else None
        execution_path = run.get("execution_path")
        if not isinstance(case, Mapping) or not isinstance(execution_path, Mapping):
            raise VerifiedHardwareError(f"{case_id}: result is missing route details")
        if not _dispatch_source_matches(
            execution_path.get("dispatch_source"),
            route_path=route_path,
            project_root=project_root,
        ):
            raise VerifiedHardwareError(
                f"{case_id}: result used an unexpected dispatch source"
            )
        if execution_path.get("dispatch_table_sha256") != route_sha256:
            raise VerifiedHardwareError(
                f"{case_id}: result used an unexpected route-table hash"
            )

        expected_policy, expected_origin = _expected_route(routes, case, identity)
        if expected_origin != "calibrated":
            raise VerifiedHardwareError(
                f"{case_id}: verified workload resolved through fallback"
            )
        if execution_path.get("dispatch_policy") != expected_policy:
            raise VerifiedHardwareError(
                f"{case_id}: expected dispatch policy {expected_policy!r}, "
                f"got {execution_path.get('dispatch_policy')!r}"
            )
        if execution_path.get("route_origin") != expected_origin:
            raise VerifiedHardwareError(
                f"{case_id}: expected route origin {expected_origin!r}, "
                f"got {execution_path.get('route_origin')!r}"
            )
        try:
            workload_case = WorkloadCase.from_dict(dict(case))
            candidate = candidate_spec_for_policy(
                workload_case,
                expected_policy,
                deployable_only=True,
            )
        except (ContractError, RuntimeError, TypeError, ValueError) as exc:
            raise VerifiedHardwareError(
                f"{case_id}: cannot resolve execution evidence for "
                f"{expected_policy!r}: {exc}"
            ) from exc
        if candidate is None:
            raise VerifiedHardwareError(
                f"{case_id}: dispatch policy {expected_policy!r} has no deployable "
                "candidate for this workload"
            )
        if not candidate.dispatch_evidence_matches(execution_path):
            raise VerifiedHardwareError(
                f"{case_id}: dispatch selected {expected_policy!r}, but the reported "
                "execution path does not prove that policy ran without fallback"
            )


def run_verified(
    config: LaunchConfig,
    *,
    paths: BundlePaths,
    identity_collector: Callable[[str], dict[str, dict[str, str]]] = (
        collect_runtime_identity
    ),
    sweep_service: BenchmarkSweepService | None = None,
    cancellation_token: CancellationToken | None = None,
) -> Path:
    """Validate and measure one bundle while exclusively owning its GPU."""

    with device_measurement_lease(
        paths.project_root,
        config.device,
        purpose="verified hardware sweep",
    ):
        return _run_verified(
            config,
            paths=paths,
            identity_collector=identity_collector,
            sweep_service=sweep_service,
            cancellation_token=cancellation_token,
        )


def _run_verified(
    config: LaunchConfig,
    *,
    paths: BundlePaths,
    identity_collector: Callable[[str], dict[str, dict[str, str]]] = (
        collect_runtime_identity
    ),
    sweep_service: BenchmarkSweepService | None = None,
    cancellation_token: CancellationToken | None = None,
) -> Path:
    """Validate, execute, attribute, and summarize one verified-device sweep."""

    profile = load_json(paths.profile)
    routes = load_json(paths.routes)
    try:
        _table, route_sha256, manifest = load_verified_bundle(
            paths.routes,
            project_root=paths.project_root,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise VerifiedHardwareError(
            f"verified bundle provenance is stale or invalid: {exc}"
        ) from exc
    expected_identity = expected_hardware_identity(profile)
    actual_identity = identity_collector(config.device)
    validate_hardware_identity(expected_identity, actual_identity)

    workload_set = load_workload_set(
        paths.project_root,
        manifest.workload_set_id,
    )
    validate_workload_route_coverage(
        workload_set,
        routes=routes,
        identity=actual_identity,
    )

    def validate_before_persist(
        measured_workload: WorkloadSet,
        runs: Sequence[Mapping[str, Any]],
        _run_paths: Sequence[Path],
        summary: dict[str, Any],
    ) -> None:
        if measured_workload.sha256 != workload_set.sha256:
            raise VerifiedHardwareError("verified workload changed during the sweep")
        if summary.get("sweep_outcome") != "complete":
            failures = summary.get("failed_cases")
            if isinstance(failures, Sequence):
                detail = ", ".join(
                    f"{item.get('case_id')}={item.get('outcome')}"
                    for item in failures
                    if isinstance(item, Mapping)
                )
            else:
                detail = "unknown failure"
            raise VerifiedHardwareError(f"verified sweep is incomplete: {detail}")
        validate_run_routes(
            runs,
            routes=routes,
            identity=actual_identity,
            route_path=paths.routes,
            route_sha256=route_sha256,
            project_root=paths.project_root,
        )
        applied_case_ids = {_case_id(run) for run in runs}
        case_results = summary.get("case_results")
        if not isinstance(case_results, list):
            raise VerifiedHardwareError("verified summary is missing case results")
        for case_result in case_results:
            if (
                not isinstance(case_result, dict)
                or case_result.get("case_id") not in applied_case_ids
            ):
                raise VerifiedHardwareError(
                    "verified summary does not match the validated runs"
                )
            case_result["policy_applied"] = True

    protocol = MeasurementProtocol.for_preset(
        config.preset,
        matmul_precision="high",
        allow_tf32=True,
        timeout_seconds=config.timeout,
    )
    service = sweep_service or BenchmarkSweepService()
    try:
        sweep: BenchmarkSweepResult = service.run(
            BenchmarkSweepRequest(
                project_root=paths.project_root,
                workload_set_id=manifest.workload_set_id,
                protocol=protocol,
                device=config.device,
                target="solution",
                solution_policy="dispatch",
                output_root=paths.sweeps,
            ),
            validate_before_persist=validate_before_persist,
            cancellation_token=cancellation_token,
        )
    except VerifiedHardwareError:
        raise
    except (ContractError, OSError, TypeError, ValueError) as exc:
        raise VerifiedHardwareError(f"verified sweep failed: {exc}") from exc
    if sweep.summary.get("sweep_outcome") == "cancelled":
        return sweep.summary_path
    if config.preset == "formal":
        atomic_replace_json(paths.reference_formal, sweep.summary)
    return sweep.summary_path


def build_parser(hardware_id: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run the exact verified routes for {hardware_id}"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preset", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--timeout", type=_positive_float)
    return parser


def main_for_bundle(
    bundle_root: Path,
    argv: Sequence[str] | None = None,
) -> int:
    paths = BundlePaths.from_bundle(bundle_root)
    hardware_id = paths.bundle_root.name
    args = build_parser(hardware_id).parse_args(argv)
    try:
        summary_path = run_verified(
            LaunchConfig(
                device=args.device,
                preset=args.preset,
                timeout=args.timeout,
            ),
            paths=paths,
        )
    except KeyboardInterrupt:
        print(f"verified {hardware_id} run cancelled", file=sys.stderr)
        return 130
    except (ContractError, OSError, VerifiedHardwareError) as exc:
        print(f"verified {hardware_id} run failed: {exc}", file=sys.stderr)
        return 1
    summary = load_json(summary_path)
    if summary.get("sweep_outcome") == "cancelled":
        print(f"verified {hardware_id} run cancelled: {summary_path}", file=sys.stderr)
        return 130
    print(f"verified summary: {summary_path}")
    return 0
