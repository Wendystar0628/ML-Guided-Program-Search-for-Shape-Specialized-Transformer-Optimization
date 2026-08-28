"""Command-line entry point for the performance development loop."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from policy_registry import POLICY_SELECTORS, policy_ids
from runner.calibration import (
    CalibrationEvent,
    CalibrationRequest,
    CalibrationService,
)
from runner.contracts import (
    OFFICIAL_WORKLOAD_SET_ID,
    ContractError,
    MeasurementProtocol,
    RunVariant,
    TransformerShape,
    load_workload_set,
    select_transformer_shape,
)
from runner.supervisor import (
    run_managed_benchmark,
    run_managed_probe,
    run_managed_profile,
)
from runner.sweep import (
    BenchmarkSweepRequest,
    BenchmarkSweepService,
)
from runner.tuning import (
    candidates_for_shape,
    run_tuning_case,
)

DEFAULT_WORKLOAD_SET = OFFICIAL_WORKLOAD_SET_ID
DEFAULT_CALIBRATION_SMOKE_LIMIT = 3


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


def _add_variant_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)


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
        help=(
            "write a single-case result here, or create an isolated "
            "<sweep_id> directory here for a full sweep"
        ),
    )
    benchmark.add_argument(
        "--solution-policy",
        choices=tuple(sorted(POLICY_SELECTORS | policy_ids())),
        default="dispatch",
        help="select the project-specific Solution path",
    )
    _add_protocol_arguments(benchmark)
    _add_variant_arguments(benchmark)

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
        choices=tuple(sorted(POLICY_SELECTORS | policy_ids())),
        default="dispatch",
        help="select the project-specific Solution path",
    )
    _add_protocol_arguments(profile)
    _add_variant_arguments(profile)

    tune = subparsers.add_parser(
        "tune",
        help="measure explicitly selected optimization candidates serially",
    )
    tune.add_argument("--workload-set", default=DEFAULT_WORKLOAD_SET)
    tune.add_argument(
        "--case-id",
        action="append",
        required=True,
        help="case to measure; repeat to run more than one case",
    )
    tune.add_argument(
        "--candidate",
        action="append",
        required=True,
        help=(
            "candidate to measure in the supplied order; repeat to compare "
            "explicit experiment candidates"
        ),
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
    _add_variant_arguments(tune)

    calibrate = subparsers.add_parser(
        "calibrate",
        help=(
            "probe once, Smoke-screen candidates, formally verify finalists, "
            "and automatically publish complete routes"
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
    calibrate.add_argument(
        "--timeout",
        type=float,
        help="per-candidate worker timeout override for both calibration stages",
    )
    calibrate.add_argument(
        "--candidate-limit",
        type=_positive_int,
        default=DEFAULT_CALIBRATION_SMOKE_LIMIT,
        help=(
            "maximum deployable candidates in the Smoke screening pool, "
            "including eager-auto and the current incumbent (default: 3)"
        ),
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
    _add_variant_arguments(calibrate)

    return parser


def _protocol_from_args(args: argparse.Namespace) -> MeasurementProtocol:
    return MeasurementProtocol.for_preset(
        args.preset,
        compile_baseline=args.compile_baseline,
        compile_solution=args.compile_solution,
        compile_mode=args.compile_mode,
        matmul_precision=args.matmul_precision,
        allow_tf32=args.allow_tf32,
        timeout_seconds=args.timeout,
    )


def _variant_from_args(args: argparse.Namespace) -> RunVariant:
    variant = RunVariant(
        dtype=args.dtype,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
    )
    variant.validate()
    return variant


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
    for case_result in summary["case_results"]:
        speedup = case_result["speedup"]
        suffix = f" | {speedup:.4f}x" if speedup is not None else ""
        policy = case_result.get("actual_policy")
        policy_suffix = f" | {policy}" if isinstance(policy, str) else ""
        print(
            f"{case_result['case_id']}: {case_result['outcome']}{suffix}{policy_suffix}"
        )
    geomean = summary.get("geomean_speedup")
    if isinstance(geomean, (int, float)):
        print(f"unweighted geomean: {geomean:.4f}x")
    if summary["failed_cases"]:
        failures = ", ".join(
            f"{item['case_id']}={item['outcome']}" for item in summary["failed_cases"]
        )
        print(f"aggregation unavailable: {failures}")


def _run_benchmark(args: argparse.Namespace, project_root: Path) -> int:
    protocol = _protocol_from_args(args)
    variant = _variant_from_args(args)
    if args.case_id is not None:
        workload_set = load_workload_set(project_root, args.workload_set)
        shape = select_transformer_shape(workload_set, args.case_id)
        result, result_path = run_managed_benchmark(
            project_root,
            workload_set_id=args.workload_set,
            shape=shape,
            variant=variant,
            protocol=protocol,
            device=args.device,
            target=args.target,
            workload_sha256=workload_set.sha256,
            result_dir=args.result_dir,
            solution_policy=args.solution_policy,
        )
        _print_run_summary(result, result_path)
        return _exit_code(result["outcome"])

    def on_case_started(index: int, total: int, shape: TransformerShape) -> None:
        print(f"\n[{index}/{total}] {shape.case_id}")

    def on_case_completed(result: dict[str, Any], result_path: Path) -> None:
        _print_run_summary(result, result_path)

    sweep = BenchmarkSweepService().run(
        BenchmarkSweepRequest(
            project_root=project_root,
            workload_set_id=args.workload_set,
            protocol=protocol,
            device=args.device,
            variant=variant,
            target=args.target,
            solution_policy=args.solution_policy,
            output_root=args.result_dir,
        ),
        on_case_started=on_case_started,
        on_case_completed=on_case_completed,
    )
    _print_sweep_summary(sweep.summary)
    print(f"sweep summary: {sweep.summary_path}")
    if any(run["outcome"] == "cancelled" for run in sweep.runs):
        return 130
    return 0 if sweep.summary["sweep_outcome"] == "complete" else 1


def _run_profile(args: argparse.Namespace, project_root: Path) -> int:
    workload_set = load_workload_set(project_root, args.workload_set)
    shape = select_transformer_shape(workload_set, args.case_id)
    result, result_path = run_managed_profile(
        project_root,
        workload_set_id=args.workload_set,
        shape=shape,
        variant=_variant_from_args(args),
        protocol=_protocol_from_args(args),
        device=args.device,
        target=args.target,
        workload_sha256=workload_set.sha256,
        solution_policy=args.solution_policy,
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
            ("dense_attention_to_l2", "attention/L2"),
            ("estimated_peak_to_device_memory", "peak/device-memory"),
            ("estimated_blocks_per_sm", "blocks/SM"),
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
    protocol = summary.get("protocol")
    preset = protocol.get("preset") if isinstance(protocol, Mapping) else None
    stage = "Formal finalist verification" if preset == "formal" else "Smoke screening"
    print(f"\n=== {stage}: {summary['case_id']} ===")
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
        print(f"{item['candidate_id']}: {outcome}{details}")
        execution_path = item["execution_path"]
        if isinstance(execution_path, dict):
            route = " | ".join(
                str(value)
                for value in (
                    execution_path.get("selected_policy"),
                    execution_path.get("attention_backend"),
                    execution_path.get("runtime_wrapper"),
                    execution_path.get("residual_norm_backend"),
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
        print(
            f"winner: {winner['candidate_id']} | "
            f"{winner['target_median_ms']:.6f} ms | {winner['speedup']:.4f}x"
        )


def _run_tune(args: argparse.Namespace, project_root: Path) -> int:
    workload_set = load_workload_set(project_root, args.workload_set)
    variant = _variant_from_args(args)
    protocol = MeasurementProtocol.for_preset(
        args.preset,
        matmul_precision=args.matmul_precision,
        allow_tf32=args.allow_tf32,
        timeout_seconds=args.timeout,
    )
    summaries: list[dict[str, Any]] = []
    for case_id in args.case_id:
        shape = select_transformer_shape(workload_set, case_id)
        available = {item.candidate_id for item in candidates_for_shape(shape, variant)}
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
        print(f"\n=== Full-workload candidate measurement: {case_id} ===")
        summary = run_tuning_case(
            project_root,
            workload_set_id=args.workload_set,
            workload_sha256=workload_set.sha256,
            shape=shape,
            variant=variant,
            base_protocol=protocol,
            device=args.device,
            requested_candidates=requested_candidates,
            routing_plan=routing_plan,
            device_profile=None,
        )
        summaries.append(summary)
        _print_tuning_summary(summary)
        if any(item["outcome"] == "cancelled" for item in summary["observations"]):
            return 130
    return 0 if all(summary["winner"] is not None for summary in summaries) else 1


def _print_calibration_event(event: CalibrationEvent) -> None:
    if event.kind == "probe_started":
        print("\n=== Routing probe (once per command) ===")
    elif event.kind == "probe_completed":
        _print_run_summary(event.data["result"], event.data["result_path"])
    elif event.kind == "routing_plan_ready":
        _print_routing_plan(str(event.case_id), event.data["plan"])
    elif event.kind == "plan_only_completed":
        print(
            "\nplan-only: the routing probe and coarse candidate plans completed; "
            "candidate-limit describes the future Smoke pool, and no full "
            "Transformer candidate benchmarks were run"
        )
    elif event.kind == "stage_started":
        title = (
            "Smoke screening stage"
            if event.stage == "smoke"
            else "Formal finalist verification stage"
        )
        print(f"\n=== {title} ===")
    elif event.kind == "tuning_completed":
        _print_tuning_summary(event.data["summary"])
    elif event.kind == "stage_outputs":
        title = (
            "Screening outputs"
            if event.stage == "smoke"
            else "Formal calibration outputs"
        )
        print(f"\n=== {title} ===")
        for summary in event.data["summaries"]:
            print(
                f"{summary['case_id']}: tuning-id={summary['tuning_id']} | "
                f"summary={summary['summary_path']}"
            )
    elif event.kind == "smoke_screening_only":
        print("smoke calibration is screening-only and cannot be promoted")
        case_arguments = " ".join(
            f"--case-id {case_id}" for case_id in event.data["case_ids"]
        )
        print(
            "formal calibration: python -m runner calibrate --preset formal "
            f"{case_arguments}"
        )
    elif event.kind == "formal_selection_skipped":
        print(f"Formal finalist selection skipped: {event.data['message']}")
    elif event.kind == "implementation_changed":
        print(f"Formal finalist verification skipped: {event.data['message']}")
    elif event.kind == "formal_plans_ready":
        print("\n=== Formal finalists selected from Smoke measurements ===")
        for shape, plan in zip(
            event.data["shapes"],
            event.data["plans"],
            strict=True,
        ):
            print(f"{shape.case_id}: {', '.join(plan['candidate_order'])}")
    elif event.kind == "promotion_skipped":
        print(f"automatic route update skipped: {event.data['message']}")
    elif event.kind == "promotion_started":
        print("\n=== Automatic route update ===")
    elif event.kind == "promotion_failed":
        print(f"automatic route update failed: {event.data['message']}")
    elif event.kind == "promotion_completed":
        for shape, winner in zip(
            event.data["shapes"],
            event.data["winners"],
            strict=True,
        ):
            print(
                f"{shape.case_id}: deployed {winner['candidate_id']} -> "
                f"{winner['solution_policy']}"
            )
        route_path = event.data["route_path"]
        print(f"{event.data['route_action']}: {route_path.parent}")
        print(f"dispatch routes: {route_path}")


def _run_calibrate(args: argparse.Namespace, project_root: Path) -> int:
    request = CalibrationRequest(
        project_root=project_root,
        workload_set_id=args.workload_set,
        case_ids=tuple(args.case_id or ()),
        device=args.device,
        preset=args.preset,
        timeout_seconds=args.timeout,
        candidate_limit=args.candidate_limit,
        plan_only=args.plan_only,
        matmul_precision=args.matmul_precision,
        allow_tf32=args.allow_tf32,
        variant=_variant_from_args(args),
    )
    result = CalibrationService().run(
        request,
        on_event=_print_calibration_event,
    )
    if result.exit_code == 130 and result.checkpoint_path is not None:
        print(f"saved calibration checkpoint: {result.checkpoint_path}")
    return result.exit_code


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
        raise ContractError(f"unsupported command: {args.command}")
    except ContractError as exc:
        print(f"configuration error: {exc}")
        return 2
    except KeyboardInterrupt:
        print("cancelled by user")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
