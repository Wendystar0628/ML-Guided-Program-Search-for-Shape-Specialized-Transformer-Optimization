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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = Path(__file__).resolve().parent / "result"
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
    benchmark: dict[str, Any],
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
    measured_shapes = benchmark.get("shapes")
    if not isinstance(measured_shapes, list):
        raise TypeError("benchmark summary has no shapes array")

    results: list[dict[str, Any]] = []
    for measured in measured_shapes:
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
                "error": measured.get("error"),
            }
        )

    return {
        "schema_version": 1,
        "status": benchmark.get("status"),
        "started_at": benchmark.get("started_at"),
        "finished_at": benchmark.get("finished_at"),
        "elapsed_seconds": benchmark.get("elapsed_seconds"),
        "hardware": hardware,
        "measurement": {
            "preset": benchmark.get("preset"),
            "variant": benchmark.get("variant"),
            "speedup_definition": "baseline_median_ms / deployed_median_ms",
            "aggregate_definition": (
                "geometric mean of speedups for official_01 through official_13"
            ),
            "shape_14_baseline": (
                "Dense SxS baseline is not materialized; Shape 14 reports deployed "
                "latency, peak memory, and correctness only."
            ),
        },
        "resident_geomean_speedup": benchmark.get("resident_geomean_speedup"),
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
    geomean = report.get("resident_geomean_speedup")
    lines = [
        "# Final Performance Results",
        "",
        f"- Status: `{report.get('status')}`",
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
        (
            "| Shape | Baseline median (ms) | Deployed median (ms) | "
            "Deployed P90 (ms) | Speedup | Peak VRAM (GiB) | Correct |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for shape in report["shapes"]:
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
    lines.extend(
        [
            "",
            (
                "Shape 14 uses the memory-efficient streamed path. Its dense SxS "
                "baseline is not materialized on the target device, so it is "
                "excluded from the speedup geometric mean."
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure all official Shapes for the final competition report."
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
    benchmark_root = working_directory / "benchmark"
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

    benchmark_code = _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "cli.py"),
            "benchmark",
            "--group",
            "all",
            "--preset",
            args.preset,
            "--device",
            args.device,
            "--output",
            str(benchmark_root),
        ]
    )
    summary_path = benchmark_root / "summary.json"
    if not summary_path.exists():
        return benchmark_code or 1

    report = build_final_report(
        _load_json(summary_path),
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
    return benchmark_code


if __name__ == "__main__":
    raise SystemExit(main())
