"""Command-line entry point for the performance development loop."""

from __future__ import annotations

import argparse
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    WorkloadCase,
    load_json,
    load_workload_set,
    new_run_id,
    select_workload_case,
)
from runner.hardware_router import build_routing_plan
from runner.route_promotion import (
    auto_promote_calibration,
    find_matching_verified_route,
    promote_tuning_summaries,
    validate_promotion_case_set,
    verified_profile_from_probe_result,
)
from runner.supervisor import (
    run_managed_benchmark,
    run_managed_probe,
    run_managed_profile,
)
from runner.sweep import summarize_sweep
from runner.tuning import (
    SOLUTION_POLICIES,
    candidates_for_case,
    deployable_candidate_id_for_policy,
    run_tuning_case,
    select_candidates,
    solution_policy,
)
from solution.dispatch import load_route_table, make_route_key, resolve_route_result

DEFAULT_WORKLOAD_SET = "transformer_core_v1"
DEFAULT_CANDIDATE_LIMIT = 4


@dataclass(frozen=True)
class RoutingProbeContext:
    """One probe result reused by every case in a calibration command."""

    hardware_profile: dict[str, Any]
    raw_result: dict[str, Any]
    result_path: Path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _add_protocol_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--compile-solution", action="store_true")
    parser.add_argument("--cuda-graph-solution", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure the current Solution against the official baseline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser(
        "probe", help="probe the selected device in a fresh process"
    )
    probe.add_argument("--device", default="cuda:0")
    probe.add_argument("--timeout", type=float, default=30.0)
    probe.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    probe.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    probe.add_argument(
        "--mode",
        choices=("routing", "diagnostic"),
        default="diagnostic",
        help="routing collects route-planning anchors; diagnostic also checks SDPA backends",
    )

    benchmark = subparsers.add_parser(
        "benchmark", help="run correctness and end-to-end performance measurement"
    )
    benchmark.add_argument(
        "--target", choices=("baseline", "solution"), default="solution"
    )
    benchmark.add_argument("--workload-set", default=DEFAULT_WORKLOAD_SET)
    benchmark.add_argument(
        "--case-id",
        help="run one case; omit this option to sweep the ordered workload set",
    )
    benchmark.add_argument("--device", default="cuda:0")
    benchmark.add_argument(
        "--result-dir",
        type=Path,
        help="write per-case JSON results to this directory",
    )
    benchmark.add_argument(
        "--solution-policy",
        choices=SOLUTION_POLICIES,
        default="dispatch",
        help="select the project-specific Solution path",
    )
    _add_protocol_arguments(benchmark)

    profile = subparsers.add_parser(
        "profile", help="collect a compact top-operation profile for one case"
    )
    profile.add_argument(
        "--target", choices=("baseline", "solution"), default="solution"
    )
    profile.add_argument("--workload-set", default=DEFAULT_WORKLOAD_SET)
    profile.add_argument("--case-id", required=True)
    profile.add_argument("--device", default="cuda:0")
    profile.add_argument(
        "--solution-policy",
        choices=SOLUTION_POLICIES,
        default="dispatch",
        help="select the project-specific Solution path",
    )
    _add_protocol_arguments(profile)

    tune = subparsers.add_parser(
        "tune",
        help="screen a bounded set of optimization candidates serially",
    )
    tune.add_argument("--workload-set", default=DEFAULT_WORKLOAD_SET)
    tune.add_argument(
        "--case-id",
        action="append",
        required=True,
        help="case to screen; repeat to run more than one case",
    )
    tune.add_argument(
        "--candidate",
        action="append",
        help=(
            "candidate to measure in the supplied order; omit to probe once and "
            "build a coarse candidate plan"
        ),
    )
    tune.add_argument(
        "--candidate-limit",
        type=_positive_int,
        default=DEFAULT_CANDIDATE_LIMIT,
        help="maximum hardware-ranked candidates, including eager-auto",
    )
    tune.add_argument("--device", default="cuda:0")
    tune.add_argument("--preset", choices=("smoke", "formal"), default="smoke")
    tune.add_argument("--timeout", type=float)
    tune.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    tune.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    calibrate = subparsers.add_parser(
        "calibrate",
        help=(
            "probe once, measure bounded candidates, and automatically publish "
            "complete Formal routes"
        ),
    )
    calibrate.add_argument("--workload-set", default=DEFAULT_WORKLOAD_SET)
    calibrate.add_argument(
        "--case-id",
        action="append",
        help="case to calibrate; repeat as needed, or omit for the full workload",
    )
    calibrate.add_argument("--device", default="cuda:0")
    calibrate.add_argument("--preset", choices=("smoke", "formal"), default="smoke")
    calibrate.add_argument("--timeout", type=float)
    calibrate.add_argument(
        "--candidate-limit",
        type=_positive_int,
        default=DEFAULT_CANDIDATE_LIMIT,
        help="maximum hardware-ranked candidates, including eager-auto",
    )
    calibrate.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "run the routing probe and show coarse candidate plans without "
            "executing Transformer candidate benchmarks"
        ),
    )
    calibrate.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    calibrate.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    promote = subparsers.add_parser(
        "promote",
        help="promote formal tuning winners into the offline dispatcher",
    )
    promote.add_argument(
        "--tuning-id",
        action="append",
        required=True,
        help="formal tuning summary to promote; repeat for a shared runtime key",
    )
    promote.add_argument(
        "--route-table",
        type=Path,
        required=True,
        help="verified device-package routes.json to update",
    )
    return parser


def _protocol_from_args(args: argparse.Namespace) -> MeasurementProtocol:
    return MeasurementProtocol.for_preset(
        args.preset,
        compile_baseline=args.compile_baseline,
        compile_solution=args.compile_solution,
        cuda_graph_solution=args.cuda_graph_solution,
        compile_mode=args.compile_mode,
        matmul_precision=args.matmul_precision,
        allow_tf32=args.allow_tf32,
        timeout_seconds=args.timeout,
    )


def _exit_code(outcome: str) -> int:
    if outcome == "success":
        return 0
    if outcome == "invalid_output":
        return 2
    if outcome == "cancelled":
        return 130
    return 1


def _print_run_summary(result: dict[str, Any], result_path: Path) -> None:
    print(f"outcome: {result['outcome']}")
    correctness = result.get("correctness")
    if correctness is not None:
        print(f"correctness: {'PASS' if correctness['passed'] else 'FAIL'}")
    performance = result.get("performance")
    if performance is not None and result["outcome"] == "success":
        baseline = performance["baseline"]["median_ms"]
        print(f"baseline median: {baseline:.6f} ms")
        target_record = performance.get("target")
        if isinstance(target_record, dict):
            print(f"target median: {target_record['median_ms']:.6f} ms")
        speedup = performance.get("speedup")
        if speedup is not None:
            print(f"observed speedup: {speedup:.4f}x")
    profile = result.get("profile")
    if profile is not None and result["outcome"] == "success":
        print(f"profile iterations: {profile['iterations']}")
        hotspots = profile["operator_hotspots"]
        print(f"operator hotspots: {len(hotspots)}")
        for operation in hotspots:
            metric = operation["self_time_us_per_forward"]
            print(f"  {operation['name']}: {metric:.3f} us/forward")
    probe = result.get("probe")
    if probe is not None:
        passed = probe["device_operation_passed"]
        print(f"device operation: {'PASS' if passed else 'FAIL'}")
    failure = result.get("failure")
    if failure is not None:
        print(f"failure: {failure['stage']}/{failure['type']}: {failure['message']}")
    print(f"result: {result_path}")


def _print_sweep_summary(summary: dict[str, Any]) -> None:
    print("\n=== Sweep summary ===")
    print(f"outcome: {summary['sweep_outcome']}")
    for case in summary["case_results"]:
        speedup = case["speedup"]
        suffix = f" | {speedup:.4f}x" if speedup is not None else ""
        print(f"{case['case_id']}: {case['outcome']}{suffix}")
    if summary["sweep_outcome"] == "complete" and summary["groups"]:
        print("groups:")
        for group in summary["groups"]:
            print(f"  {group['display_name']}: {group['geomean_speedup']:.4f}x")
        print(
            f"group-balanced geomean: {summary['group_balanced_geomean_speedup']:.4f}x"
        )
        print(f"worst case: {summary['worst_case_speedup']:.4f}x")
    elif summary["failed_cases"]:
        failures = ", ".join(
            f"{item['case_id']}={item['outcome']}" for item in summary["failed_cases"]
        )
        print(f"aggregation unavailable: {failures}")


def _run_benchmark(args: argparse.Namespace, project_root: Path) -> int:
    workload_set = load_workload_set(project_root, args.workload_set)
    protocol = _protocol_from_args(args)
    if args.case_id is not None:
        case = select_workload_case(workload_set, args.case_id)
        with solution_policy(args.solution_policy):
            result, result_path = run_managed_benchmark(
                project_root,
                workload_set_id=args.workload_set,
                case=case,
                protocol=protocol,
                device=args.device,
                target=args.target,
                workload_sha256=workload_set["sha256"],
                result_dir=args.result_dir,
            )
        _print_run_summary(result, result_path)
        return _exit_code(result["outcome"])

    runs: list[dict[str, Any]] = []
    sweep_id = new_run_id()
    with solution_policy(args.solution_policy):
        for index, case in enumerate(workload_set["cases"], start=1):
            print(f"\n[{index}/{len(workload_set['cases'])}] {case.case_id}")
            result, result_path = run_managed_benchmark(
                project_root,
                workload_set_id=args.workload_set,
                case=case,
                protocol=protocol,
                device=args.device,
                target=args.target,
                workload_sha256=workload_set["sha256"],
                sweep_id=sweep_id,
                result_dir=args.result_dir,
            )
            runs.append(result)
            _print_run_summary(result, result_path)
            if result["outcome"] == "cancelled":
                break
    summary = summarize_sweep(workload_set, runs, target=args.target)
    _print_sweep_summary(summary)
    if any(run["outcome"] == "cancelled" for run in runs):
        return 130
    return 0 if summary["sweep_outcome"] == "complete" else 1


def _run_profile(args: argparse.Namespace, project_root: Path) -> int:
    workload_set = load_workload_set(project_root, args.workload_set)
    case = select_workload_case(workload_set, args.case_id)
    with solution_policy(args.solution_policy):
        result, result_path = run_managed_profile(
            project_root,
            workload_set_id=args.workload_set,
            case=case,
            protocol=_protocol_from_args(args),
            device=args.device,
            target=args.target,
            workload_sha256=workload_set["sha256"],
        )
    _print_run_summary(result, result_path)
    return _exit_code(result["outcome"])


def _run_probe(args: argparse.Namespace, project_root: Path) -> int:
    result, result_path = run_managed_probe(
        project_root,
        device=args.device,
        timeout_seconds=args.timeout,
        matmul_precision=args.matmul_precision,
        allow_tf32=args.allow_tf32,
        probe_mode=args.mode,
    )
    _print_run_summary(result, result_path)
    return _exit_code(result["outcome"])


def _hardware_profile_from_probe(result: Mapping[str, Any]) -> dict[str, Any]:
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
            for field in (
                "driver",
                "torch",
                "cuda_runtime",
                "triton",
                "triton_available",
            ):
                value = software.get(field)
                if value is not None:
                    profile[field] = value
        platform_profile = candidate.get("platform")
        if isinstance(platform_profile, Mapping):
            for source_name, profile_name in (("system", "platform_system"),):
                value = platform_profile.get(source_name)
                if value is not None:
                    profile[profile_name] = value

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
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                anchors["launch_latency_us"] = value
        graph = raw_anchors.get("cuda_graph_replay")
        if isinstance(graph, Mapping):
            value = graph.get("effective_latency_per_node_us")
            if value is None:
                value = graph.get("replay_latency_us")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                anchors["graph_replay_per_node_us"] = value
        device_copy = raw_anchors.get("device_copy")
        if isinstance(device_copy, Mapping):
            value = device_copy.get("effective_bandwidth_gbps")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
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
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    gemm_tflops[dtype_name] = value
        if gemm_tflops:
            anchors["gemm_tflops"] = gemm_tflops
        softmax = raw_anchors.get("softmax_fp32")
        if isinstance(softmax, Mapping):
            value = softmax.get("throughput_gigaelements_per_second")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                anchors["softmax_giga_elements_per_s"] = value
        if anchors:
            profile["performance_anchors"] = anchors

    for field in ("device_type",):
        if not isinstance(profile.get(field), str) or not profile[field]:
            raise ContractError(f"routing hardware profile is missing {field}")
    if profile["device_type"].lower() == "cuda":
        for field in ("device_name", "compute_capability"):
            if not isinstance(profile.get(field), str) or not profile[field]:
                raise ContractError(f"routing hardware profile is missing {field}")
    return profile


def _probe_for_routing(
    args: argparse.Namespace,
    project_root: Path,
) -> tuple[RoutingProbeContext | None, int]:
    print("\n=== Routing probe (once per command) ===")
    timeout = args.timeout if args.timeout is not None else 30.0
    result, result_path = run_managed_probe(
        project_root,
        device=args.device,
        timeout_seconds=timeout,
        matmul_precision=getattr(args, "matmul_precision", "high"),
        allow_tf32=getattr(args, "allow_tf32", True),
        probe_mode="routing",
    )
    _print_run_summary(result, result_path)
    exit_code = _exit_code(result["outcome"])
    if exit_code != 0:
        return None, exit_code
    return (
        RoutingProbeContext(
            hardware_profile=_hardware_profile_from_probe(result),
            raw_result=result,
            result_path=result_path,
        ),
        0,
    )


def _incumbent_candidate_id(
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
    case: WorkloadCase,
    hardware_profile: Mapping[str, Any],
    candidate_limit: int,
    *,
    incumbent_candidate_id: str | None = None,
) -> dict[str, Any]:
    applicable = tuple(item.candidate_id for item in candidates_for_case(case))
    try:
        options: dict[str, Any] = {}
        if incumbent_candidate_id is not None:
            options["required_candidate_ids"] = (incumbent_candidate_id,)
        raw_plan = build_routing_plan(
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
        raise ContractError(f"routing plan for {case.case_id} exceeds candidate-limit")
    if "eager-auto" in applicable and "eager-auto" not in candidate_order:
        raise ContractError(f"routing plan for {case.case_id} must retain eager-auto")
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


def _print_routing_plan(case_id: str, plan: Mapping[str, Any]) -> None:
    print(f"\n=== Coarse candidate plan: {case_id} ===")
    print(f"source: {plan.get('source', 'unknown')}")
    print("scope: candidate ordering only; full-workload measurement decides the route")
    bottleneck = plan.get("bottleneck_class")
    if isinstance(bottleneck, str):
        print(f"bottleneck: {bottleneck}")
    signals = plan.get("routing_signals")
    if isinstance(signals, Mapping):
        signal_labels = (
            ("machine_ridge_flops_per_byte", "ridge"),
            ("workload_intensity_to_ridge", "intensity/ridge"),
            ("attention_matrix_to_l2", "attention/L2"),
            ("estimated_blocks_per_sm", "blocks/SM"),
            ("softmax_to_compute_lower_bound", "softmax/compute-LB"),
        )
        parts = [
            f"{label}={float(signals[field]):.3f}"
            for field, label in signal_labels
            if isinstance(signals.get(field), (int, float))
            and not isinstance(signals.get(field), bool)
        ]
        if parts:
            print(f"signals: {', '.join(parts)}")
    order = plan.get("candidate_order")
    if isinstance(order, list):
        print(f"candidates: {', '.join(str(value) for value in order)}")
    reasons = plan.get("selection_reasons")
    if isinstance(reasons, Mapping):
        for candidate_id in order if isinstance(order, list) else []:
            values = reasons.get(candidate_id)
            if isinstance(values, list) and values:
                print(f"  {candidate_id}: {'; '.join(str(value) for value in values)}")
    rejections = plan.get("capability_rejections")
    if isinstance(rejections, Mapping) and rejections:
        rejected = ", ".join(str(value) for value in rejections)
        print(f"capability rejections: {rejected}")


def _print_tuning_summary(summary: dict[str, Any]) -> None:
    print(f"\n=== Candidate screening: {summary['case_id']} ===")
    print(f"tuning id: {summary['tuning_id']}")
    summary_path = summary.get("summary_path")
    if isinstance(summary_path, str):
        print(f"tuning summary: {summary_path}")
    for item in summary["observations"]:
        speedup = item["speedup"]
        target = item["target_median_ms"]
        outcome = item["outcome"]
        if outcome == "success" and not item["policy_applied"]:
            outcome = "policy_not_applied"
        details = ""
        if isinstance(target, (int, float)) and isinstance(speedup, (int, float)):
            details = f" | {target:.6f} ms | {speedup:.4f}x"
            conservative = item.get("conservative_speedup")
            if isinstance(conservative, (int, float)):
                details += f" | conservative {conservative:.4f}x"
        print(f"{item['candidate_id']}: {outcome}{details}")
        execution_path = item["execution_path"]
        if isinstance(execution_path, dict):
            route = " | ".join(
                str(value)
                for value in (
                    execution_path.get("resolved_qkv_layout"),
                    execution_path.get("resolved_attention"),
                    execution_path.get("padding_route"),
                    execution_path.get("block_fusion"),
                    (
                        execution_path.get("resolved_ffn")
                        if execution_path.get("resolved_ffn") != "torch_exact_gelu"
                        else None
                    ),
                )
                if value not in (None, "none")
            )
            if route:
                print(f"  route: {route}")
        if item["correctness_passed"] is False:
            print(
                "  correctness: FAIL | "
                f"failed_elements={item['failed_elements']} | "
                f"max_abs_error={item['max_abs_error']}"
            )
        print(f"  result: {item['result_path']}")
    winner = summary["winner"]
    if winner is None:
        print("winner: none (no correct successful candidate)")
    else:
        conservative = winner.get("conservative_speedup")
        conservative_suffix = (
            f" | conservative {conservative:.4f}x"
            if isinstance(conservative, (int, float))
            else ""
        )
        print(
            f"winner: {winner['candidate_id']} | "
            f"{winner['target_median_ms']:.6f} ms | {winner['speedup']:.4f}x"
            f"{conservative_suffix}"
        )


def _run_tune(args: argparse.Namespace, project_root: Path) -> int:
    workload_set = load_workload_set(project_root, args.workload_set)
    protocol = MeasurementProtocol.for_preset(
        args.preset,
        matmul_precision=args.matmul_precision,
        allow_tf32=args.allow_tf32,
        timeout_seconds=args.timeout,
    )
    hardware_profile: dict[str, Any] | None = None
    if args.candidate is None:
        probe_context, probe_exit_code = _probe_for_routing(args, project_root)
        if probe_context is None:
            return probe_exit_code
        hardware_profile = probe_context.hardware_profile

    summaries: list[dict[str, Any]] = []
    for case_id in args.case_id:
        case = select_workload_case(workload_set, case_id)
        available = {item.candidate_id for item in candidates_for_case(case)}
        if args.candidate:
            unavailable = sorted(set(args.candidate) - available)
            if unavailable:
                raise ContractError(
                    f"candidates are not available for {case_id}: {unavailable}; "
                    f"available={sorted(available)}"
                )
            requested_candidates = list(args.candidate)
            routing_plan: dict[str, Any] = {
                "source": "explicit_candidates",
                "decision_scope": "candidate_order_only",
                "requires_full_workload_measurement": True,
                "candidate_order": requested_candidates,
            }
        else:
            assert hardware_profile is not None
            routing_plan = _routing_plan_for_case(
                case,
                hardware_profile,
                args.candidate_limit,
            )
            requested_candidates = routing_plan["candidate_order"]
            _print_routing_plan(case_id, routing_plan)
        print(f"\n=== Full-workload candidate measurement: {case_id} ===")
        summary = run_tuning_case(
            project_root,
            workload_set_id=args.workload_set,
            workload_sha256=workload_set["sha256"],
            case=case,
            base_protocol=protocol,
            device=args.device,
            requested_candidates=requested_candidates,
            routing_plan=routing_plan,
            device_profile=hardware_profile,
        )
        summaries.append(summary)
        _print_tuning_summary(summary)
        if any(item["outcome"] == "cancelled" for item in summary["observations"]):
            return 130
    return 0 if all(summary["winner"] is not None for summary in summaries) else 1


def _run_calibrate(args: argparse.Namespace, project_root: Path) -> int:
    if (
        args.preset == "formal"
        and not args.plan_only
        and (args.matmul_precision != "high" or args.allow_tf32 is not True)
    ):
        raise ContractError(
            "formal calibration deployment requires "
            "--matmul-precision high and --allow-tf32"
        )
    workload_set = load_workload_set(project_root, args.workload_set)
    cases = (
        [select_workload_case(workload_set, case_id) for case_id in args.case_id]
        if args.case_id
        else list(workload_set["cases"])
    )
    probe_context, probe_exit_code = _probe_for_routing(args, project_root)
    if probe_context is None:
        return probe_exit_code
    hardware_profile = probe_context.hardware_profile

    full_case_ids = [case.case_id for case in workload_set["cases"]]
    selected_case_ids = [case.case_id for case in cases]
    existing_route: Path | None = None
    if args.preset == "formal":
        verified_profile = verified_profile_from_probe_result(probe_context.raw_result)
        existing_route = find_matching_verified_route(project_root, verified_profile)

    plans = [
        _routing_plan_for_case(
            case,
            hardware_profile,
            args.candidate_limit,
            incumbent_candidate_id=_incumbent_candidate_id(
                case,
                hardware_profile,
                existing_route,
            ),
        )
        for case in cases
    ]
    for case, plan in zip(cases, plans, strict=True):
        _print_routing_plan(case.case_id, plan)
    if args.plan_only:
        print(
            "\nplan-only: the routing probe and coarse candidate plans completed; "
            "no full Transformer candidate benchmarks were run"
        )
        return 0

    if args.preset == "formal":
        validate_promotion_case_set(selected_case_ids)
        if existing_route is None and set(selected_case_ids) != set(full_case_ids):
            raise ContractError(
                "a new verified hardware package requires one complete Formal "
                "workload calibration"
            )

    protocol = MeasurementProtocol.for_preset(
        args.preset,
        matmul_precision=args.matmul_precision,
        allow_tf32=args.allow_tf32,
        timeout_seconds=args.timeout,
    )
    summaries: list[dict[str, Any]] = []
    for case, plan in zip(cases, plans, strict=True):
        print(f"\n=== Full-workload candidate measurement: {case.case_id} ===")
        summary = run_tuning_case(
            project_root,
            workload_set_id=args.workload_set,
            workload_sha256=workload_set["sha256"],
            case=case,
            base_protocol=protocol,
            device=args.device,
            requested_candidates=plan["candidate_order"],
            routing_plan=plan,
            device_profile=hardware_profile,
        )
        summaries.append(summary)
        _print_tuning_summary(summary)
        if any(item["outcome"] == "cancelled" for item in summary["observations"]):
            return 130

    print("\n=== Calibration outputs ===")
    for summary in summaries:
        print(
            f"{summary['case_id']}: tuning-id={summary['tuning_id']} | "
            f"summary={summary['summary_path']}"
        )
    if args.preset == "formal":
        if any(
            summary.get("complete") is not True
            or summary.get("winner") is None
            or summary.get("deployable_winner") is None
            for summary in summaries
        ):
            print(
                "automatic route update skipped: calibration has no complete "
                "deployable winner"
            )
            return 1
        print("\n=== Automatic route update ===")
        try:
            previous_route_bytes = (
                existing_route.read_bytes()
                if existing_route is not None and existing_route.is_file()
                else None
            )
            _, winners, route_path, created = auto_promote_calibration(
                project_root,
                summaries,
                probe_result=probe_context.raw_result,
                full_workload_case_ids=full_case_ids,
            )
        except (ContractError, OSError) as exc:
            print(f"automatic route update failed: {exc}")
            return 1
        for case, winner in zip(cases, winners, strict=True):
            print(
                f"{case.case_id}: deployed {winner['candidate_id']} -> "
                f"{winner['solution_policy']}"
            )
        route_changed = (
            previous_route_bytes is not None
            and route_path.is_file()
            and route_path.read_bytes() != previous_route_bytes
        )
        if created:
            action = "created verified package"
        elif route_changed:
            action = "updated verified package"
        else:
            action = "verified package already has the selected routes"
        print(f"{action}: {route_path.parent}")
        print(f"dispatch routes: {route_path}")
    else:
        if any(summary.get("winner") is None for summary in summaries):
            return 1
        print("smoke calibration is screening-only and cannot be promoted")
        case_arguments = " ".join(f"--case-id {case.case_id}" for case in cases)
        print(
            "formal calibration: python -m runner calibrate --preset formal "
            f"{case_arguments}"
        )
    return 0


def _run_promote(args: argparse.Namespace, project_root: Path) -> int:
    summaries: list[dict[str, Any]] = []
    for tuning_id in args.tuning_id:
        if Path(tuning_id).name != tuning_id:
            raise ContractError("tuning-id must be a file-safe identifier")
        summary_path = project_root / "results" / "tuning" / f"{tuning_id}.json"
        summaries.append(load_json(summary_path))
    route_path = args.route_table
    if not route_path.is_absolute():
        route_path = project_root / route_path
    _, winners, route_path = promote_tuning_summaries(
        project_root,
        summaries,
        route_path=route_path.resolve(),
    )
    for winner in winners:
        print(f"promoted: {winner['candidate_id']} -> {winner['solution_policy']}")
    print(f"dispatch routes: {route_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    try:
        if args.command == "probe":
            return _run_probe(args, project_root)
        if args.command == "benchmark":
            return _run_benchmark(args, project_root)
        if args.command == "profile":
            return _run_profile(args, project_root)
        if args.command == "tune":
            return _run_tune(args, project_root)
        if args.command == "calibrate":
            return _run_calibrate(args, project_root)
        if args.command == "promote":
            return _run_promote(args, project_root)
        raise ContractError(f"unsupported command: {args.command}")
    except ContractError as exc:
        print(f"configuration error: {exc}")
        return 2
    except KeyboardInterrupt:
        print("cancelled by user")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
