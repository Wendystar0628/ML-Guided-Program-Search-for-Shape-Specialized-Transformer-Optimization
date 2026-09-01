"""Run the report-facing resident mechanism-family ablation grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarking.ablation import (
    ABLATION_FAMILIES,
    DEFAULT_ABLATION_SHAPES,
    AblationFamily,
    new_ablation_run_directory,
    run_component_ablation_suite,
    write_component_ablation_csv,
)
from benchmarking.device_isolation import DeviceLease
from benchmarking.protocols import RunVariant
from deployment.environment import configure_process_math_mode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure legal leave-one-family-out Transformer programs"
    )
    parser.add_argument("--case-id", action="append")
    parser.add_argument(
        "--family",
        action="append",
        choices=tuple(family.value for family in ABLATION_FAMILIES),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--figure-csv",
        type=Path,
        help="optionally refresh the checked-in plot table from this run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_process_math_mode()
    output = args.output or new_ablation_run_directory(
        PROJECT_ROOT / "result" / "ablation"
    )
    case_ids = tuple(args.case_id or DEFAULT_ABLATION_SHAPES)
    families = tuple(
        AblationFamily(value) for value in (args.family or ABLATION_FAMILIES)
    )
    print(f"ablation summary: {output / 'ablation.json'}")
    with DeviceLease(
        device=args.device,
        root=PROJECT_ROOT / "observations" / "locks",
        on_wait=print,
    ):
        result = run_component_ablation_suite(
            project_root=PROJECT_ROOT,
            case_ids=case_ids,
            families=families,
            variant=RunVariant(),
            device=args.device,
            output_directory=output,
        )

    if args.figure_csv is not None:
        figure_csv = args.figure_csv
        if not figure_csv.is_absolute():
            figure_csv = PROJECT_ROOT / figure_csv
        write_component_ablation_csv(result.summary, figure_csv)
        print(f"figure data: {figure_csv}")

    for item in result.summary["comparisons"]:
        label = f"{item['case_id']} / {item['mechanism_label']}"
        if item["status"] == "measured":
            retained = 100.0 * item["retained_performance_fraction"]
            print(f"{label}: {retained:.1f}% retained")
        else:
            print(f"{label}: {item['status']}")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
