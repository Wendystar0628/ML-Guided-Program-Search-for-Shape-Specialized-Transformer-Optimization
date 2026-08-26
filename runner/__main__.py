"""Command-line entry point for the performance development loop."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    load_json,
    load_workload_set,
    new_run_id,
    select_workload_case,
)
from runner.route_promotion import promote_tuning_summary
from runner.supervisor import (
    run_managed_benchmark,
    run_managed_probe,
    run_managed_profile,
)
from runner.sweep import summarize_sweep
from runner.tuning import (
    SOLUTION_POLICIES,
    candidates_for_case,
    run_tuning_case,
    solution_policy,
)

DEFAULT_WORKLOAD_SET = "rtx4080_core_v1"


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
        help="candidate to run; omit to use the applicable fixed set",
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

    promote = subparsers.add_parser(
        "promote",
        help="promote one formal tuning winner into the offline dispatcher",
    )
    promote.add_argument("--tuning-id", required=True)
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
    )
    _print_run_summary(result, result_path)
    return _exit_code(result["outcome"])


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
        summary = run_tuning_case(
            project_root,
            workload_set_id=args.workload_set,
            workload_sha256=workload_set["sha256"],
            case=case,
            base_protocol=protocol,
            device=args.device,
            requested_candidates=args.candidate,
        )
        summaries.append(summary)
        _print_tuning_summary(summary)
        if any(item["outcome"] == "cancelled" for item in summary["observations"]):
            return 130
    return 0 if all(summary["winner"] is not None for summary in summaries) else 1


def _run_promote(args: argparse.Namespace, project_root: Path) -> int:
    tuning_id = args.tuning_id
    if Path(tuning_id).name != tuning_id:
        raise ContractError("tuning-id must be a file-safe identifier")
    summary_path = project_root / "results" / "tuning" / f"{tuning_id}.json"
    summary = load_json(summary_path)
    _, winner, route_path = promote_tuning_summary(project_root, summary)
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
