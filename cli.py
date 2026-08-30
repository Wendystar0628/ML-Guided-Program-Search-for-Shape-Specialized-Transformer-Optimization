"""One small CLI for direct measurement, profiling, probing, and search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from benchmarking.configuration import resolve_config
from benchmarking.device_queue import DeviceLease
from benchmarking.measure import profile_config
from benchmarking.probe import execute_probe
from benchmarking.protocols import (
    ContractError,
    RunVariant,
    load_resident_shapes,
    load_shape,
    load_streamed_shapes,
    write_json,
)
from benchmarking.suite import new_run_directory, run_benchmark_suite
from deployment.environment import ImplementationScope, configure_process_math_mode

if TYPE_CHECKING:
    from autotune.search_sweep import SearchSweepRequest


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _variant_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=_positive_float, default=1.0)


def _search_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--storage", type=Path)
    parser.add_argument("--budget-seconds", type=_positive_float, default=900.0)
    parser.add_argument("--max-trials", type=_positive_int)
    parser.add_argument("--seed", type=int, default=1234)
    _variant_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shape-aware Transformer program search"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    benchmark = commands.add_parser(
        "benchmark",
        help="measure official shapes in isolated GPU processes",
    )
    benchmark.add_argument(
        "--group",
        choices=("resident", "shape14"),
        default="resident",
    )
    benchmark.add_argument("--case-id", action="append")
    benchmark.add_argument("--config", type=Path)
    benchmark.add_argument(
        "--preset",
        choices=("smoke", "formal", "final"),
        default="smoke",
    )
    benchmark.add_argument("--device", default="cuda:0")
    benchmark.add_argument("--output", type=Path)
    _variant_arguments(benchmark)

    profile = commands.add_parser("profile", help="profile one resident shape")
    profile.add_argument("--case-id", required=True)
    profile.add_argument("--config", type=Path)
    profile.add_argument("--device", default="cuda:0")
    profile.add_argument("--iterations", type=_positive_int, default=5)
    profile.add_argument("--output", type=Path)
    _variant_arguments(profile)

    probe = commands.add_parser("probe", help="inspect the selected CUDA device")
    probe.add_argument("--device", default="cuda:0")
    probe.add_argument("--output", type=Path)

    search = commands.add_parser("search", help="run generated branch-local TPE")
    search.add_argument("--case-id", action="append", required=True)
    _search_arguments(search)

    optimize = commands.add_parser(
        "optimize",
        help="repeat full search sweeps until search progress stops",
    )
    optimize.add_argument("--group", choices=("resident", "shape14"), required=True)
    optimize.add_argument(
        "--no-progress-patience",
        type=_positive_int,
        default=8,
    )
    optimize.add_argument("--max-iterations", type=_positive_int, default=32)
    _search_arguments(optimize)
    return parser


def _variant(args: argparse.Namespace) -> RunVariant:
    return RunVariant(
        dtype=args.dtype,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
    )


def _print_result(value: dict[str, Any]) -> None:
    if value.get("error") is not None:
        print(f"{value['case_id']}: FAIL")
        print(f"  error: {value['error']}")
        return
    print(f"{value['case_id']}: {'PASS' if value['passed'] else 'FAIL'}")
    print(f"  optimized median: {value['median_ms']:.6f} ms")
    print(f"  optimized p90:    {value['p90_ms']:.6f} ms")
    if value["baseline_median_ms"] is not None:
        print(f"  baseline median:  {value['baseline_median_ms']:.6f} ms")
        print(f"  speedup:          {value['speedup']:.4f}x")


def _benchmark_case_ids(
    project_root: Path, args: argparse.Namespace
) -> tuple[str, ...]:
    if args.case_id:
        return tuple(args.case_id)
    if args.group == "resident":
        shapes = load_resident_shapes(project_root)
    elif args.group == "shape14":
        shapes = load_streamed_shapes(project_root)
    else:
        raise ContractError(f"unknown shape group: {args.group}")
    return tuple(shape.case_id for shape in shapes)


def _shape_scope(project_root: Path, case_id: str) -> ImplementationScope:
    shape = load_shape(project_root, case_id)
    return (
        ImplementationScope.SHAPE14 if shape.streamed else ImplementationScope.RESIDENT
    )


def _search_scope(
    project_root: Path,
    case_ids: tuple[str, ...],
) -> ImplementationScope:
    if not case_ids:
        raise ContractError("search requires at least one shape")
    scopes = {_shape_scope(project_root, case_id) for case_id in case_ids}
    if len(scopes) != 1:
        raise ContractError("one search cannot mix resident shapes with Shape 14")
    return next(iter(scopes))


def _search_storage_root(
    project_root: Path,
    scope: ImplementationScope,
    requested: Path | None,
) -> Path:
    from autotune.study_storage import scoped_search_root

    return scoped_search_root(project_root, scope.value, requested)


def _benchmark(args: argparse.Namespace, project_root: Path) -> int:
    output_directory = args.output or new_run_directory(
        project_root / "observations" / "benchmarks"
    )
    print(f"benchmark summary: {output_directory / 'summary.json'}")
    suite = run_benchmark_suite(
        project_root=project_root,
        case_ids=_benchmark_case_ids(project_root, args),
        config_path=args.config,
        variant=_variant(args),
        preset=args.preset,
        device=args.device,
        output_directory=output_directory,
    )
    for result in suite.summary["shapes"]:
        _print_result(result)
    geomean = suite.summary["resident_geomean_speedup"]
    if geomean is not None:
        print(f"resident geomean speedup: {geomean:.4f}x")
    return suite.exit_code


def _profile(args: argparse.Namespace, project_root: Path) -> int:
    shape = load_shape(project_root, args.case_id)
    variant = _variant(args)
    operations = profile_config(
        shape,
        resolve_config(
            args.config,
            shape,
            variant,
            args.device,
            project_root=project_root,
        ),
        variant,
        args.device,
        iterations=args.iterations,
    )
    for operation in operations:
        print(
            f"{operation['name']}: "
            f"{operation['self_time_us_per_forward']:.3f} us/forward"
        )
    if args.output is not None:
        write_json(args.output, {"case_id": shape.case_id, "operations": operations})
    return 0


def _probe(args: argparse.Namespace) -> int:
    result = execute_probe(args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output is not None:
        write_json(args.output, result)
    return 0


def _search(args: argparse.Namespace, project_root: Path) -> int:
    from autotune.run_log import SearchRunLog
    from autotune.search_sweep import SearchSweep, SearchSweepRequest

    case_ids = tuple(args.case_id)
    scope = _search_scope(project_root, case_ids)
    storage_root = _search_storage_root(project_root, scope, args.storage)
    request = SearchSweepRequest(
        project_root=project_root,
        case_ids=case_ids,
        scope=scope,
        device=args.device,
        storage_root=storage_root,
        budget_seconds=args.budget_seconds,
        max_trials=args.max_trials,
        seed=args.seed,
        variant=_variant(args),
    )
    target = (
        request.case_ids[0]
        if len(request.case_ids) == 1
        else f"{len(request.case_ids)}-shapes"
    )
    run_log = SearchRunLog(
        root=storage_root / "logs",
        mode="search",
        target=target,
        request=_run_log_request(request),
    )
    print(f"run log: {run_log.path}")
    try:
        result = SearchSweep(observer=run_log.record_shape).run(request)
    except KeyboardInterrupt:
        run_log.finish(status="interrupted", exit_code=130)
        raise
    except Exception as exc:
        run_log.fail(exc)
        raise
    run_log.finish(
        status=("interrupted" if result.exit_code == 130 else "finished"),
        exit_code=result.exit_code,
    )
    for item in result.shape_results:
        selected = item.selected_config
        print(f"{item.case_id}: {item.search_result.stop_reason}")
        print(f"  selected: {None if selected is None else selected.config_id}")
        print(f"  level-1 trials: {item.search_result.completed_level1}")
        print(f"  deployment updated: {item.deployment_updated}")
    return result.exit_code


def _run_log_request(
    request: SearchSweepRequest,
    *,
    group: str | None = None,
) -> dict[str, Any]:
    device_name = None
    compute_capability = None
    try:
        device = torch.device(request.device)
        if device.type == "cuda" and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(device)
            major, minor = torch.cuda.get_device_capability(device)
            compute_capability = f"{major}.{minor}"
    except (RuntimeError, ValueError):
        pass
    storage_root = _search_storage_root(
        request.project_root,
        request.scope,
        request.storage_root,
    )
    value: dict[str, Any] = {
        "case_ids": list(request.case_ids),
        "scope": request.scope.value,
        "device": request.device,
        "device_name": device_name,
        "compute_capability": compute_capability,
        "variant": request.variant.to_dict(),
        "seed": request.seed,
        "budget_seconds_per_shape": request.budget_seconds,
        "max_trials_per_shape": request.max_trials,
        "study_database": str(storage_root / "search.sqlite3"),
    }
    if group is not None:
        value["group"] = group
    return value


def _optimization_case_ids(project_root: Path, group: str) -> tuple[str, ...]:
    if group == "resident":
        shapes = load_resident_shapes(project_root)
    elif group == "shape14":
        shapes = load_streamed_shapes(project_root)
    else:
        raise ContractError(f"unknown shape group: {group}")
    if not shapes:
        raise ContractError(f"shape group is empty: {group}")
    return tuple(shape.case_id for shape in shapes)


def _optimize(args: argparse.Namespace, project_root: Path) -> int:
    from autotune.optimization_loop import (
        OptimizationIteration,
        OptimizationLoop,
        OptimizationLoopPolicy,
    )
    from autotune.run_log import SearchRunLog
    from autotune.search_sweep import SearchSweep, SearchSweepRequest

    scope = ImplementationScope(args.group)
    storage_root = _search_storage_root(project_root, scope, args.storage)
    request = SearchSweepRequest(
        project_root=project_root,
        case_ids=_optimization_case_ids(project_root, args.group),
        scope=scope,
        device=args.device,
        storage_root=storage_root,
        budget_seconds=args.budget_seconds,
        max_trials=args.max_trials,
        seed=args.seed,
        variant=_variant(args),
    )
    log_request = _run_log_request(request, group=args.group)
    log_request.update(
        {
            "max_iterations": args.max_iterations,
            "no_progress_patience": args.no_progress_patience,
        }
    )
    run_log = SearchRunLog(
        root=storage_root / "logs",
        mode="optimize",
        target=args.group,
        request=log_request,
    )
    print(f"run log: {run_log.path}")

    def print_iteration(iteration: OptimizationIteration) -> None:
        print(f"iteration {iteration.index}")
        for item in iteration.search_result.shape_results:
            selected = item.selected_config
            print(f"  {item.case_id}: {item.search_result.stop_reason}")
            print(f"    selected: {None if selected is None else selected.config_id}")
            print(f"    deployment updated: {item.deployment_updated}")
        print(f"  deployments: {iteration.deployment_updates}")
        print(
            f"  shapes with new Level-1 evidence: {iteration.shapes_with_level1_progress}"
        )
        print(f"  no-progress streak: {iteration.no_progress_streak}")
        run_log.record_iteration(iteration)

    try:
        result = OptimizationLoop(SearchSweep(observer=run_log.record_shape)).run(
            request,
            OptimizationLoopPolicy(
                no_progress_patience=args.no_progress_patience,
                max_iterations=args.max_iterations,
            ),
            observer=print_iteration,
        )
    except KeyboardInterrupt:
        run_log.finish(status="interrupted", exit_code=130)
        raise
    except Exception as exc:
        run_log.fail(exc)
        raise
    run_log.finish(
        status=("interrupted" if result.exit_code == 130 else "finished"),
        stop_reason=result.stop_reason,
        exit_code=result.exit_code,
        iterations=result.iterations_run,
        total_deployment_updates=result.total_deployment_updates,
    )
    print(f"stopped: {result.stop_reason}")
    print(f"iterations: {result.iterations_run}")
    print(f"deployment updates: {result.total_deployment_updates}")
    return result.exit_code


def main(argv: list[str] | None = None) -> int:
    configure_process_math_mode()
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parent
    try:
        if args.command == "probe":
            return _probe(args)
        with DeviceLease(
            device=args.device,
            root=project_root / "observations" / "locks",
            on_wait=print,
        ):
            if args.command == "benchmark":
                return _benchmark(args, project_root)
            if args.command == "profile":
                return _profile(args, project_root)
            if args.command == "search":
                return _search(args, project_root)
            if args.command == "optimize":
                return _optimize(args, project_root)
            raise ContractError(f"unsupported command: {args.command}")
    except (ContractError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("cancelled")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
