#!/usr/bin/env python3
"""Run the final competition benchmark and create compact presentation files."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "result"
WORKING_ROOT = OUTPUT_ROOT / ".working"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _baseline_status(result: dict[str, Any]) -> str:
    if result.get("baseline_median_ms") is not None:
        return "measured"
    if result.get("case_id") == "official_14":
        return "not_measured_dense_s2"
    return "unavailable"


def build_final_report(
    resident_benchmark: dict[str, Any],
    shape14_benchmark: dict[str, Any],
    hardware: dict[str, Any],
    workload: dict[str, Any],
) -> dict[str, Any]:
    raw_shapes = workload.get("ordered_shapes")
    if not isinstance(raw_shapes, list):
        raise TypeError("official workload has no ordered_shapes array")
    shapes_by_id = {
        str(shape["case_id"]): shape
        for shape in raw_shapes
        if isinstance(shape, dict) and "case_id" in shape
    }
    resident_shapes = resident_benchmark.get("shapes", [])
    shape14_shapes = shape14_benchmark.get("shapes", [])
    if not isinstance(resident_shapes, list):
        raise TypeError("resident benchmark summary has no shapes array")
    if not isinstance(shape14_shapes, list):
        raise TypeError("Shape 14 benchmark summary has no shapes array")

    results: list[dict[str, Any]] = []
    for measured in [*resident_shapes, *shape14_shapes]:
        if not isinstance(measured, dict):
            raise TypeError("benchmark shape result must be an object")
        case_id = str(measured.get("case_id"))
        shape = shapes_by_id.get(case_id)
        if shape is None:
            raise ValueError(f"benchmark contains unknown shape: {case_id}")
        results.append(
            {
                "case_id": case_id,
                "shape": dict(shape),
                "config_id": measured.get("config_id"),
                "correctness_passed": bool(measured.get("passed"))
                and bool(measured.get("execution_matches")),
                "max_tolerance_ratio": measured.get("max_tolerance_ratio"),
                "baseline_status": _baseline_status(measured),
                "baseline_median_ms": measured.get("baseline_median_ms"),
                "deployed_median_ms": measured.get("median_ms"),
                "deployed_p90_ms": measured.get("p90_ms"),
                "speedup": measured.get("speedup"),
                "peak_memory_bytes": measured.get("peak_memory_bytes"),
                "estimated_model_flops": measured.get("estimated_model_flops"),
                "latency_kind": measured.get("latency_kind"),
                "output_digest": measured.get("output_digest"),
                "error": measured.get("error"),
            }
        )

    resident_status = str(resident_benchmark.get("status", "failed"))
    shape14_status = str(shape14_benchmark.get("status", "failed"))
    group_statuses = (resident_status, shape14_status)
    status = (
        "completed"
        if all(value == "completed" for value in group_statuses)
        else "interrupted"
        if "interrupted" in group_statuses
        else "completed_with_failures"
    )
    return {
        "schema_version": 1,
        "status": status,
        "groups": {
            "resident": {
                "status": resident_status,
                "started_at": resident_benchmark.get("started_at"),
                "finished_at": resident_benchmark.get("finished_at"),
                "elapsed_seconds": resident_benchmark.get("elapsed_seconds"),
                "progress": resident_benchmark.get("progress"),
                "error": resident_benchmark.get("error"),
            },
            "shape14": {
                "status": shape14_status,
                "started_at": shape14_benchmark.get("started_at"),
                "finished_at": shape14_benchmark.get("finished_at"),
                "elapsed_seconds": shape14_benchmark.get("elapsed_seconds"),
                "progress": shape14_benchmark.get("progress"),
                "error": shape14_benchmark.get("error"),
            },
        },
        "hardware": hardware,
        "measurement": {
            "preset": resident_benchmark.get("preset", shape14_benchmark.get("preset")),
            "variant": resident_benchmark.get(
                "variant", shape14_benchmark.get("variant")
            ),
            "speedup_definition": "baseline_median_ms / deployed_median_ms",
            "aggregate_definition": (
                "geometric mean of speedups for official_01 through official_13"
            ),
            "shape_14_baseline": (
                "Dense SxS baseline is not materialized; Shape 14 reports deployed "
                "latency, peak memory, and correctness only."
            ),
            "shape_14_latency": (
                "Smoke is a scaled model-compute estimate; formal/final runs one "
                "complete logical batch with distinct streamed microbatches."
            ),
        },
        "resident_geomean_speedup": resident_benchmark.get("resident_geomean_speedup"),
        "shapes": results,
    }


def _number(value: object, digits: int = 3) -> str:
    if not isinstance(value, int | float):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _gib(value: object) -> str:
    if not isinstance(value, int | float):
        return "N/A"
    return f"{float(value) / (1024**3):.2f}"


def render_markdown(report: dict[str, Any]) -> str:
    hardware = report["hardware"]
    groups = report["groups"]
    geomean = report.get("resident_geomean_speedup")
    resident_shapes = [
        shape for shape in report["shapes"] if shape["case_id"] != "official_14"
    ]
    shape14 = next(
        (shape for shape in report["shapes"] if shape["case_id"] == "official_14"),
        None,
    )
    lines = [
        "# Final Performance Results",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Resident task: `{groups['resident'].get('status')}`",
        f"- Shape 14 task: `{groups['shape14'].get('status')}`",
        f"- Device: {hardware.get('name', 'N/A')}",
        f"- Compute capability: {hardware.get('compute_capability', 'N/A')}",
        (
            f"- PyTorch / CUDA: {hardware.get('torch_version', 'N/A')} / "
            f"{hardware.get('cuda_version', 'N/A')}"
        ),
        f"- Measurement preset: `{report['measurement'].get('preset')}`",
        f"- Completed at: {report.get('completed_at', report.get('finished_at'))}",
        "- Speedup: baseline median latency / deployed median latency",
        f"- Shapes 01-13 geometric mean speedup: {_number(geomean, 4)}x",
        "",
        "## Shapes 01-13",
        "",
        (
            "| Shape | Baseline median (ms) | Deployed median (ms) | "
            "Deployed P90 (ms) | Speedup | Peak VRAM (GiB) | Correct |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for shape in resident_shapes:
        speedup = _number(shape.get("speedup"), 3)
        if speedup != "N/A":
            speedup += "x"
        lines.append(
            "| {case_id} | {baseline} | {median} | {p90} | {speedup} | "
            "{memory} | {correct} |".format(
                case_id=shape["case_id"],
                baseline=_number(shape.get("baseline_median_ms")),
                median=_number(shape.get("deployed_median_ms")),
                p90=_number(shape.get("deployed_p90_ms")),
                speedup=speedup,
                memory=_gib(shape.get("peak_memory_bytes")),
                correct="PASS" if shape.get("correctness_passed") else "FAIL",
            )
        )
    lines.extend(["", "## Shape 14", ""])
    if shape14 is None:
        lines.append("No Shape 14 result was produced.")
    else:
        lines.extend(
            [
                (
                    "| Shape | Latency kind | Deployed median (ms) | "
                    "Deployed P90 (ms) | Peak VRAM (GiB) | Correct |"
                ),
                "| --- | --- | ---: | ---: | ---: | :---: |",
                (
                    "| {case_id} | {latency_kind} | {median} | {p90} | "
                    "{memory} | {correct} |"
                ).format(
                    case_id=shape14["case_id"],
                    latency_kind=shape14.get("latency_kind") or "N/A",
                    median=_number(shape14.get("deployed_median_ms")),
                    p90=_number(shape14.get("deployed_p90_ms")),
                    memory=_gib(shape14.get("peak_memory_bytes")),
                    correct=("PASS" if shape14.get("correctness_passed") else "FAIL"),
                ),
            ]
        )
    lines.extend(
        [
            "",
            (
                "Shape 14 uses the memory-efficient streamed path and is excluded "
                "from the Shapes 01-13 speedup geometric mean."
            ),
            (
                "Its final/formal latency covers distinct streamed microbatches; "
                "smoke latency is explicitly reported as a model-compute estimate."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _timestamp() -> tuple[str, str]:
    now = datetime.now(UTC)
    return (
        now.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        now.strftime("%Y%m%dT%H%M%S.%fZ"),
    )


def _prepare_working_directory() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _, run_id = _timestamp()
    working_directory = WORKING_ROOT / run_id
    working_directory.mkdir(parents=True)
    return working_directory


def _completed_output_paths() -> tuple[str, Path, Path]:
    completed_at, run_id = _timestamp()
    run_directory = OUTPUT_ROOT / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    return (
        completed_at,
        run_directory / "final_performance.json",
        run_directory / "final_performance.md",
    )


def _run(command: list[str]) -> int:
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


def _benchmark_command(
    *,
    group: str,
    preset: str,
    device: str,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "cli.py"),
        "benchmark",
        "--group",
        group,
        "--preset",
        preset,
        "--device",
        device,
        "--output",
        str(output),
    ]


def _load_group_summary(
    *,
    path: Path,
    group: str,
    preset: str,
    exit_code: int,
) -> dict[str, Any]:
    if path.exists():
        return _load_json(path)
    return {
        "status": "failed",
        "preset": preset,
        "shapes": [],
        "error": f"{group} benchmark exited with code {exit_code} without a summary",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure resident Shapes and Shape 14 as independent tasks."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--preset",
        choices=("smoke", "formal", "final"),
        default="final",
        help="Use final for the submitted result; smoke is only a quick check.",
    )
    args = parser.parse_args(argv)

    working_directory = _prepare_working_directory()
    hardware_path = working_directory / "hardware.json"
    probe_code = _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "cli.py"),
            "probe",
            "--device",
            args.device,
            "--output",
            str(hardware_path),
        ]
    )
    if probe_code != 0:
        return probe_code

    resident_root = working_directory / "resident"
    resident_code = _run(
        _benchmark_command(
            group="resident",
            preset=args.preset,
            device=args.device,
            output=resident_root,
        )
    )
    resident_summary = _load_group_summary(
        path=resident_root / "summary.json",
        group="resident",
        preset=args.preset,
        exit_code=resident_code,
    )

    shape14_root = working_directory / "shape14"
    shape14_code = _run(
        _benchmark_command(
            group="shape14",
            preset=args.preset,
            device=args.device,
            output=shape14_root,
        )
    )
    shape14_summary = _load_group_summary(
        path=shape14_root / "summary.json",
        group="shape14",
        preset=args.preset,
        exit_code=shape14_code,
    )

    report = build_final_report(
        resident_summary,
        shape14_summary,
        _load_json(hardware_path),
        _load_json(PROJECT_ROOT / "official" / "test_shapes.json"),
    )
    completed_at, json_output, markdown_output = _completed_output_paths()
    report["completed_at"] = completed_at
    json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text(render_markdown(report), encoding="utf-8")
    shutil.rmtree(working_directory)
    try:
        WORKING_ROOT.rmdir()
    except OSError:
        pass
    print(f"JSON result: {json_output}")
    print(f"Readable result: {markdown_output}")
    return 1 if resident_code != 0 or shape14_code != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
