"""Fresh-process benchmark scheduling and compact progress summaries."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .configuration import resolve_config
from .device_queue import IsolatedProcessError, run_in_fresh_process
from .measure import measure_config
from .protocols import (
    MeasurementProtocol,
    RunVariant,
    load_resident_shapes,
    load_shape,
    write_json,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_run_directory(root: Path) -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return root / run_id


def _measure_one_shape(
    project_root: Path,
    case_id: str,
    config_path: Path | None,
    variant: RunVariant,
    preset: str,
    device: str,
) -> dict[str, Any]:
    shape = load_shape(project_root, case_id)
    result = measure_config(
        shape,
        resolve_config(
            config_path,
            shape,
            variant,
            device,
            project_root=project_root,
        ),
        variant,
        MeasurementProtocol.for_benchmark(preset, case_id),
        device,
        include_baseline=not shape.streamed,
    )
    return result.to_dict()


def _resident_geomean(
    project_root: Path,
    results: list[dict[str, Any]],
) -> float | None:
    required = tuple(shape.case_id for shape in load_resident_shapes(project_root))
    by_case = {str(item.get("case_id")): item for item in results}
    if any(case_id not in by_case for case_id in required):
        return None
    speedups: list[float] = []
    for case_id in required:
        item = by_case[case_id]
        speedup = item.get("speedup")
        if item.get("status", "passed") != "passed" or not item.get("passed"):
            return None
        if not isinstance(speedup, int | float):
            return None
        value = float(speedup)
        if not math.isfinite(value) or value <= 0.0:
            return None
        speedups.append(value)
    return math.exp(sum(math.log(value) for value in speedups) / len(speedups))


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteResult:
    path: Path
    summary: dict[str, Any]
    exit_code: int


def run_benchmark_suite(
    *,
    project_root: Path,
    case_ids: tuple[str, ...],
    config_path: Path | None,
    variant: RunVariant,
    preset: str,
    device: str,
    output_directory: Path,
) -> BenchmarkSuiteResult:
    """Measure official shapes serially, with a fresh process for every shape."""

    if not case_ids:
        raise ValueError("benchmark requires at least one shape")
    if config_path is not None and len(case_ids) != 1:
        raise ValueError("--config requires exactly one --case-id")
    summary_path = output_directory / "summary.json"
    if summary_path.exists():
        raise ValueError(f"benchmark output already exists: {summary_path}")
    output_directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "run_id": output_directory.name,
        "status": "running",
        "preset": preset,
        "device": device,
        "variant": variant.to_dict(),
        "started_at": _utc_now(),
        "finished_at": None,
        "elapsed_seconds": 0.0,
        "progress": {
            "completed": 0,
            "total": len(case_ids),
            "passed": 0,
        },
        "current_case_id": None,
        "resident_geomean_speedup": None,
        "shapes": results,
    }
    write_json(summary_path, summary)

    try:
        for case_id in case_ids:
            summary["current_case_id"] = case_id
            summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
            write_json(summary_path, summary)
            try:
                result = run_in_fresh_process(
                    _measure_one_shape,
                    project_root,
                    case_id,
                    config_path,
                    variant,
                    preset,
                    device,
                )
                result["status"] = "passed" if result.get("passed") else "failed"
            except IsolatedProcessError as exc:
                result = {
                    "case_id": case_id,
                    "status": "failed",
                    "passed": False,
                    "error": str(exc),
                }
            results.append(result)
            progress = summary["progress"]
            progress["completed"] = len(results)
            progress["passed"] = sum(item.get("passed") is True for item in results)
            summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
            write_json(summary_path, summary)
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        summary["current_case_id"] = None
        summary["finished_at"] = _utc_now()
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_json(summary_path, summary)
        raise

    failures = [item for item in results if item.get("passed") is not True]
    summary["status"] = "completed_with_failures" if failures else "completed"
    summary["current_case_id"] = None
    summary["finished_at"] = _utc_now()
    summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
    summary["resident_geomean_speedup"] = _resident_geomean(
        project_root,
        results,
    )
    write_json(summary_path, summary)
    return BenchmarkSuiteResult(
        path=summary_path,
        summary=summary,
        exit_code=1 if failures else 0,
    )


__all__ = ["BenchmarkSuiteResult", "new_run_directory", "run_benchmark_suite"]
