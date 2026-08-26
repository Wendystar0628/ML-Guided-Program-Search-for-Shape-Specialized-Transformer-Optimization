"""Run one verified hardware bundle through the shared benchmark runner."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runner.contracts import (
    ContractError,
    atomic_write_json,
    load_json,
    load_workload_set,
)
from runner.sweep import summarize_sweep
from solution.dispatch import ROUTE_FIELDS, validate_route_table

WORKLOAD_SET_ID = "transformer_core_v1"
ROUTE_TABLE_ENV = "TRANSFORMER_ROUTE_TABLE"
_IDENTITY_FIELDS = (
    ("gpu", "name"),
    ("gpu", "compute_capability"),
    ("platform", "system"),
    ("platform", "machine"),
    ("software", "torch"),
    ("software", "cuda_runtime"),
    ("software", "triton"),
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
    runs: Path
    summaries: Path

    @classmethod
    def from_bundle(cls, bundle_root: Path) -> BundlePaths:
        bundle_root = bundle_root.resolve()
        project_root = bundle_root.parents[1]
        return cls(
            project_root=project_root,
            bundle_root=bundle_root,
            profile=bundle_root / "profile.json",
            routes=bundle_root / "routes.json",
            runs=bundle_root / "results" / "runs",
            summaries=bundle_root / "results" / "summaries",
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


def _nested_string(
    document: Mapping[str, Any],
    path: tuple[str, str],
) -> str:
    section = document.get(path[0])
    value = section.get(path[1]) if isinstance(section, Mapping) else None
    if not isinstance(value, str) or not value:
        raise VerifiedHardwareError(
            f"verified profile is missing {path[0]}.{path[1]}"
        )
    return value


def expected_runtime_identity(profile: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Extract the strict route identity from a persisted probe profile."""

    if profile.get("schema_version") != 1:
        raise VerifiedHardwareError("verified profile must use schema_version 1")
    if profile.get("device_operation_passed") is not True:
        raise VerifiedHardwareError("verified profile did not pass its device operation")
    hardware_profile = profile.get("hardware_profile")
    if not isinstance(hardware_profile, Mapping):
        raise VerifiedHardwareError("verified profile is missing hardware_profile")

    identity: dict[str, dict[str, str]] = {
        "gpu": {},
        "platform": {},
        "software": {},
    }
    for section, field in _IDENTITY_FIELDS:
        identity[section][field] = _nested_string(
            hardware_profile,
            (section, field),
        )
    return identity


def collect_runtime_identity(device_name: str) -> dict[str, dict[str, str]]:
    """Collect only facts that participate in an exact verified route key."""

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
        },
    }


def validate_runtime_identity(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> None:
    """Fail closed when any route-defining hardware or software fact differs."""

    mismatches: list[str] = []
    for section, field in _IDENTITY_FIELDS:
        expected_value = _nested_string(expected, (section, field))
        actual_value = _nested_string(actual, (section, field))
        if actual_value != expected_value:
            mismatches.append(
                f"{section}.{field}: expected {expected_value!r}, got {actual_value!r}"
            )
    if mismatches:
        raise VerifiedHardwareError(
            "runtime does not match the verified hardware profile: "
            + "; ".join(mismatches)
        )


def route_table_sha256(path: Path) -> str:
    """Hash the exact bytes loaded by the shared dispatcher."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise VerifiedHardwareError(f"cannot read route table {path}: {exc}") from exc


def build_benchmark_command(config: LaunchConfig, paths: BundlePaths) -> list[str]:
    """Build the single shared-runner command used by this bundle."""

    command = [
        sys.executable,
        "-m",
        "runner",
        "benchmark",
        "--target",
        "solution",
        "--solution-policy",
        "dispatch",
        "--workload-set",
        WORKLOAD_SET_ID,
        "--device",
        config.device,
        "--preset",
        config.preset,
        "--matmul-precision",
        "high",
        "--allow-tf32",
        "--result-dir",
        str(paths.runs.resolve()),
    ]
    if config.timeout is not None:
        command.extend(("--timeout", f"{config.timeout:g}"))
    return command


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
    return {
        "device_type": "cuda",
        "device_name": _nested_string(identity, ("gpu", "name")),
        "compute_capability": _nested_string(
            identity,
            ("gpu", "compute_capability"),
        ),
        "platform_system": _nested_string(identity, ("platform", "system")),
        "torch": _nested_string(identity, ("software", "torch")),
        "cuda_runtime": _nested_string(identity, ("software", "cuda_runtime")),
        "triton": _nested_string(identity, ("software", "triton")),
        "dtype": case.get("dtype"),
        "B": case.get("batch_size"),
        "S": case.get("seq_len"),
        "D": case.get("d_model"),
        "heads": case.get("num_heads"),
        "ffn": case.get("ffn_dim"),
        "layers": case.get("num_layers"),
        "causal": case.get("causal"),
    }


def _expected_route(
    routes: Mapping[str, Any],
    case: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> tuple[str, str]:
    raw_routes = routes.get("routes")
    default_policy = routes.get("default_policy")
    if routes.get("schema_version") != 2 or not isinstance(raw_routes, list):
        raise VerifiedHardwareError("route table must use schema_version 2")
    if not isinstance(default_policy, str) or not default_policy:
        raise VerifiedHardwareError("route table is missing default_policy")

    key = _route_key(case, identity)
    matches: list[str] = []
    for index, entry in enumerate(raw_routes):
        if not isinstance(entry, Mapping):
            raise VerifiedHardwareError(f"routes[{index}] must be an object")
        match = entry.get("match")
        policy = entry.get("policy")
        if not isinstance(match, Mapping) or not isinstance(policy, str) or not policy:
            raise VerifiedHardwareError(f"routes[{index}] is malformed")
        if all(key.get(field) == value for field, value in match.items()):
            matches.append(policy)
    if len(matches) > 1:
        raise VerifiedHardwareError("route table contains an ambiguous workload match")
    if matches:
        return matches[0], "calibrated"
    return default_policy, "fallback"


def _case_id(run: Mapping[str, Any]) -> str | None:
    workload = run.get("workload")
    case = workload.get("case") if isinstance(workload, Mapping) else None
    value = case.get("case_id") if isinstance(case, Mapping) else None
    return value if isinstance(value, str) and value else None


def validate_workload_route_coverage(
    workload_set: Mapping[str, Any],
    *,
    routes: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    """Require an explicit verified decision for every package workload."""

    try:
        table = validate_route_table(routes)
    except (TypeError, ValueError) as exc:
        raise VerifiedHardwareError(f"invalid verified route table: {exc}") from exc
    expected_device = {
        "device_type": "cuda",
        "device_name": _nested_string(identity, ("gpu", "name")),
        "compute_capability": _nested_string(
            identity,
            ("gpu", "compute_capability"),
        ),
        "platform_system": _nested_string(identity, ("platform", "system")),
        "torch": _nested_string(identity, ("software", "torch")),
        "cuda_runtime": _nested_string(identity, ("software", "cuda_runtime")),
        "triton": _nested_string(identity, ("software", "triton")),
    }
    for index, (match, _policy) in enumerate(table.routes):
        if set(match) != ROUTE_FIELDS:
            raise VerifiedHardwareError(
                f"routes[{index}] is not an exact hardware/software/shape route"
            )
        if any(match.get(field) != value for field, value in expected_device.items()):
            raise VerifiedHardwareError(
                f"routes[{index}] does not belong to the verified hardware profile"
            )

    cases = workload_set.get("cases")
    if not isinstance(cases, Sequence):
        raise VerifiedHardwareError("workload set is missing cases")
    missing: list[str] = []
    for case in cases:
        as_dict = getattr(case, "as_dict", None)
        if not callable(as_dict):
            raise VerifiedHardwareError("workload set contains an invalid case")
        case_document = as_dict()
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


def _new_result_paths(runs_directory: Path, before: set[Path]) -> list[Path]:
    return sorted(
        path.resolve()
        for path in runs_directory.glob("*.json")
        if path.resolve() not in before
    )


def _compact_case_result(run: Mapping[str, Any], run_path: Path) -> dict[str, Any]:
    case_id = _case_id(run)
    performance = run.get("performance")
    execution_path = run.get("execution_path")
    if (
        not isinstance(case_id, str)
        or not isinstance(performance, Mapping)
        or not isinstance(execution_path, Mapping)
    ):
        raise VerifiedHardwareError("cannot compact an incomplete benchmark result")
    baseline = performance.get("baseline")
    target = performance.get("target")
    if not isinstance(baseline, Mapping) or not isinstance(target, Mapping):
        raise VerifiedHardwareError(f"{case_id}: result is missing latency summaries")
    return {
        "case_id": case_id,
        "run_file": run_path.name,
        "policy": execution_path.get("dispatch_policy"),
        "route_origin": execution_path.get("route_origin"),
        "baseline_median_ms": baseline.get("median_ms"),
        "target_median_ms": target.get("median_ms"),
        "speedup": performance.get("speedup"),
    }


def build_verified_summary(
    workload_set: dict[str, Any],
    runs: list[dict[str, Any]],
    run_paths: Sequence[Path],
    *,
    hardware_id: str,
    identity: Mapping[str, Any],
    preset: str,
    route_source: str,
    route_sha256: str,
) -> dict[str, Any]:
    """Build the compact, reviewable index for one complete verified sweep."""

    sweep = summarize_sweep(workload_set, runs, target="solution")
    if sweep["sweep_outcome"] != "complete":
        failures = ", ".join(
            f"{item['case_id']}={item['outcome']}" for item in sweep["failed_cases"]
        )
        raise VerifiedHardwareError(f"verified sweep is incomplete: {failures}")

    sweep_ids = {run.get("sweep_id") for run in runs}
    if len(sweep_ids) != 1 or not all(isinstance(value, str) for value in sweep_ids):
        raise VerifiedHardwareError("verified results do not share one sweep_id")
    sweep_id = next(iter(sweep_ids))
    paths_by_case = {
        _case_id(run): path for run, path in zip(runs, run_paths, strict=True)
    }
    ordered_runs = sorted(
        runs,
        key=lambda run: [case.case_id for case in workload_set["cases"]].index(
            _case_id(run)
        ),
    )
    case_results = [
        _compact_case_result(run, paths_by_case[_case_id(run)]) for run in ordered_runs
    ]
    return {
        "schema_version": 1,
        "hardware_id": hardware_id,
        "workload_set_id": WORKLOAD_SET_ID,
        "sweep_id": sweep_id,
        "created_at": min(str(run.get("created_at")) for run in runs),
        "preset": preset,
        "runtime_identity": dict(identity),
        "route_table": {
            "source": route_source,
            "sha256": route_sha256,
        },
        "case_results": case_results,
        "groups": sweep["groups"],
        "group_balanced_geomean_speedup": sweep[
            "group_balanced_geomean_speedup"
        ],
        "worst_case_speedup": sweep["worst_case_speedup"],
    }


def run_verified(
    config: LaunchConfig,
    *,
    paths: BundlePaths,
    identity_collector: Callable[[str], dict[str, dict[str, str]]] = (
        collect_runtime_identity
    ),
    command_runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> Path:
    """Validate, execute, attribute, and summarize one verified-device sweep."""

    profile = load_json(paths.profile)
    routes = load_json(paths.routes)
    expected_identity = expected_runtime_identity(profile)
    actual_identity = identity_collector(config.device)
    validate_runtime_identity(expected_identity, actual_identity)

    workload_set = load_workload_set(paths.project_root, WORKLOAD_SET_ID)
    validate_workload_route_coverage(
        workload_set,
        routes=routes,
        identity=actual_identity,
    )
    route_sha256 = route_table_sha256(paths.routes)
    route_source = _portable_source(paths.routes, paths.project_root)
    paths.runs.mkdir(parents=True, exist_ok=True)
    paths.summaries.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for path in paths.runs.glob("*.json")}

    environment = os.environ.copy()
    environment[ROUTE_TABLE_ENV] = str(paths.routes.resolve())
    completed = command_runner(
        build_benchmark_command(config, paths),
        cwd=paths.project_root,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise VerifiedHardwareError(
            f"shared benchmark runner exited with code {completed.returncode}"
        )

    run_paths = _new_result_paths(paths.runs, before)
    runs = [load_json(path) for path in run_paths]
    expected_count = len(workload_set["cases"])
    if len(runs) != expected_count:
        raise VerifiedHardwareError(
            f"expected {expected_count} new result files, found {len(runs)}"
        )
    validate_run_routes(
        runs,
        routes=routes,
        identity=actual_identity,
        route_path=paths.routes,
        route_sha256=route_sha256,
        project_root=paths.project_root,
    )
    summary = build_verified_summary(
        workload_set,
        runs,
        run_paths,
        hardware_id=paths.bundle_root.name,
        identity=actual_identity,
        preset=config.preset,
        route_source=route_source,
        route_sha256=route_sha256,
    )
    summary_path = paths.summaries / f"{summary['sweep_id']}.json"
    atomic_write_json(summary_path, summary)
    return summary_path


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
    print(f"verified summary: {summary_path}")
    return 0
