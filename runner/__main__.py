"""Command-line entry point for the performance development loop."""

from __future__ import annotations

import argparse
from pathlib import Path

from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    load_workload_set,
    select_workload_case,
)
from runner.supervisor import run_managed_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the current Solution against the official baseline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser(
        "benchmark", help="run correctness and end-to-end performance measurement"
    )
    benchmark.add_argument("--workload-set", default="provisional_reference_v1")
    benchmark.add_argument("--case-id", default="default_fp32_noncausal_full")
    benchmark.add_argument("--preset", choices=("smoke", "formal"), default="smoke")
    benchmark.add_argument("--device", default="cuda:0")
    benchmark.add_argument("--timeout", type=float)
    benchmark.add_argument("--compile-baseline", action="store_true")
    benchmark.add_argument("--compile-solution", action="store_true")
    benchmark.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    benchmark.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    benchmark.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _print_summary(result: dict, result_path: Path) -> None:
    print(f"status: {result['status']}")
    correctness = result.get("correctness")
    if correctness is not None:
        print(f"correctness: {'PASS' if correctness['passed'] else 'FAIL'}")
    performance = result.get("performance")
    if performance is not None and result["status"] == "success":
        baseline = performance["baseline"]["median_ms"]
        solution = performance["solution"]["median_ms"]
        print(f"baseline median: {baseline:.6f} ms")
        print(f"solution median: {solution:.6f} ms")
        print(f"observed speedup: {performance['speedup']:.4f}x")
    failure = result.get("failure")
    if failure is not None:
        print(f"failure: {failure['kind']}: {failure['message']}")
    print(f"result: {result_path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    try:
        workload_set = load_workload_set(project_root, args.workload_set)
        case = select_workload_case(workload_set, args.case_id)
        protocol = MeasurementProtocol.for_preset(
            args.preset,
            compile_baseline=args.compile_baseline,
            compile_solution=args.compile_solution,
            compile_mode=args.compile_mode,
            matmul_precision=args.matmul_precision,
            allow_tf32=args.allow_tf32,
            timeout_seconds=args.timeout,
        )
        result, result_path = run_managed_benchmark(
            project_root,
            workload_set_id=args.workload_set,
            case=case,
            protocol=protocol,
            device=args.device,
        )
        _print_summary(result, result_path)
        if result["status"] == "success":
            return 0
        if result["status"] == "correctness_failed":
            return 2
        if result["status"] == "interrupted":
            return 130
        return 1
    except ContractError as exc:
        print(f"configuration error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
