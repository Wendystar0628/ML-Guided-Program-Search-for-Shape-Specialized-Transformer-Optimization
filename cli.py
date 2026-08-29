"""One small CLI for direct measurement, profiling, probing, and search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from benchmarking.measure import measure_config, profile_config
from benchmarking.probe import execute_probe
from benchmarking.protocols import (
    ContractError,
    MeasurementProtocol,
    RunVariant,
    TransformerShape,
    load_json,
    load_shape,
    load_shapes,
    write_json,
)
from deployment.registry import (
    EnvironmentFingerprint,
    ShapeFingerprint,
    resolve_deployed_config,
)
from solution.config import ConfigSpec, portable_config, portable_streamed_config


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shape-aware Transformer program search"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="measure one or all official shapes")
    run.add_argument("--case-id", action="append")
    run.add_argument("--config", type=Path)
    run.add_argument("--preset", choices=("smoke", "formal"), default="smoke")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--output", type=Path)
    _variant_arguments(run)

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
    search.add_argument("--device", default="cuda:0")
    search.add_argument("--storage", type=Path)
    search.add_argument("--budget-seconds", type=_positive_float, default=900.0)
    search.add_argument("--max-trials", type=_positive_int)
    search.add_argument("--seed", type=int, default=1234)
    _variant_arguments(search)
    return parser


def _variant(args: argparse.Namespace) -> RunVariant:
    return RunVariant(
        dtype=args.dtype,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
    )


def _shape_key(shape: TransformerShape, variant: RunVariant) -> ShapeFingerprint:
    return ShapeFingerprint(
        batch_size=shape.batch_size,
        qkv_dim=shape.d_model,
        heads=shape.num_heads,
        seq_len=shape.seq_len,
        layers=shape.num_layers,
        causal=shape.causal,
        ffn_dim=shape.ffn_dim,
        dtype=variant.dtype,
        padding_ratio=variant.padding_ratio,
        input_scale=variant.input_scale,
    )


def _default_config(
    shape: TransformerShape,
    variant: RunVariant,
    device: str,
) -> ConfigSpec:
    hardware = EnvironmentFingerprint.detect(torch.device(device))
    deployed = resolve_deployed_config(
        hardware=hardware,
        shape=_shape_key(shape, variant),
    )
    if deployed is not None:
        return deployed
    return portable_streamed_config() if shape.streamed else portable_config()


def _config(
    path: Path | None,
    shape: TransformerShape,
    variant: RunVariant,
    device: str,
) -> ConfigSpec:
    if path is None:
        return _default_config(shape, variant, device)
    return ConfigSpec.from_dict(load_json(path))


def _print_result(value: dict[str, Any]) -> None:
    print(f"{value['case_id']}: {'PASS' if value['passed'] else 'FAIL'}")
    print(f"  optimized median: {value['median_ms']:.6f} ms")
    print(f"  optimized p90:    {value['p90_ms']:.6f} ms")
    if value["baseline_median_ms"] is not None:
        print(f"  baseline median:  {value['baseline_median_ms']:.6f} ms")
        print(f"  speedup:          {value['speedup']:.4f}x")


def _run(args: argparse.Namespace, project_root: Path) -> int:
    variant = _variant(args)
    shapes = (
        tuple(load_shape(project_root, case_id) for case_id in args.case_id)
        if args.case_id
        else load_shapes(project_root)
    )
    if args.config is not None and len(shapes) != 1:
        raise ContractError("--config requires exactly one --case-id")
    output_root = args.output or project_root / "benchmark_runs"
    exit_code = 0
    for shape in shapes:
        result = measure_config(
            shape,
            _config(args.config, shape, variant, args.device),
            variant,
            MeasurementProtocol.for_preset(args.preset),
            args.device,
            include_baseline=not shape.streamed,
        )
        document = result.to_dict()
        _print_result(document)
        write_json(output_root / f"{shape.case_id}.json", document)
        if not result.passed:
            exit_code = 1
    return exit_code


def _profile(args: argparse.Namespace, project_root: Path) -> int:
    shape = load_shape(project_root, args.case_id)
    variant = _variant(args)
    operations = profile_config(
        shape,
        _config(args.config, shape, variant, args.device),
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
    from autotune.service import SearchService, SearchServiceRequest

    result = SearchService().run(
        SearchServiceRequest(
            project_root=project_root,
            case_ids=tuple(args.case_id),
            device=args.device,
            storage_root=args.storage,
            budget_seconds=args.budget_seconds,
            max_trials=args.max_trials,
            seed=args.seed,
            variant=_variant(args),
        )
    )
    for item in result.shape_results:
        selected = item.selected_config
        print(f"{item.case_id}: {item.search_result.stop_reason}")
        print(f"  selected: {None if selected is None else selected.config_id}")
        print(f"  level-1 trials: {item.search_result.completed_level1}")
        print(f"  deployment updated: {item.deployment_updated}")
    return result.exit_code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parent
    try:
        if args.command == "run":
            return _run(args, project_root)
        if args.command == "profile":
            return _profile(args, project_root)
        if args.command == "probe":
            return _probe(args)
        if args.command == "search":
            return _search(args, project_root)
        raise ContractError(f"unsupported command: {args.command}")
    except (ContractError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("cancelled")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
