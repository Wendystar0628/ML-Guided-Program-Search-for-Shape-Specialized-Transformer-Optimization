"""Run benchmarks in a fresh process and persist one compact result."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from runner.contracts import (
    MeasurementProtocol,
    WorkloadCase,
    atomic_write_json,
    load_json,
    new_run_id,
    utc_now,
    validate_official_snapshot,
)


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _run_worker(
    project_root: Path,
    request: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    cache_root = project_root / ".cache" / "runner"
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_root) as temporary_directory:
        temporary_root = Path(temporary_directory)
        request_path = temporary_root / "request.json"
        response_path = temporary_root / "response.json"
        atomic_write_json(request_path, request)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "runner.worker",
                str(request_path),
                str(response_path),
            ],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _stop_process(process)
            return {
                "status": "timeout",
                "environment": None,
                "correctness": None,
                "performance": None,
                "failure": {
                    "kind": "timeout",
                    "message": f"benchmark exceeded {timeout_seconds:g} seconds",
                },
            }
        except KeyboardInterrupt:
            _stop_process(process)
            return {
                "status": "interrupted",
                "environment": None,
                "correctness": None,
                "performance": None,
                "failure": {
                    "kind": "interrupted",
                    "message": "benchmark interrupted by the user",
                },
            }

        if response_path.is_file():
            return load_json(response_path)
        message = (
            stderr.strip()[-4000:] or f"worker exited with code {process.returncode}"
        )
        return {
            "status": "failed",
            "environment": None,
            "correctness": None,
            "performance": None,
            "failure": {"kind": "worker_failed", "message": message},
        }


def run_managed_benchmark(
    project_root: Path,
    *,
    workload_set_id: str,
    case: WorkloadCase,
    protocol: MeasurementProtocol,
    device: str,
) -> tuple[dict[str, Any], Path]:
    project_root = project_root.resolve()
    run_id = new_run_id()
    created_at = utc_now()
    snapshot = validate_official_snapshot(project_root)
    request = {
        "project_root": str(project_root),
        "case": case.as_dict(),
        "protocol": protocol.as_dict(),
        "device": device,
    }
    response = _run_worker(project_root, request, protocol.timeout_seconds)
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": created_at,
        "completed_at": utc_now(),
        "status": response["status"],
        "official_snapshot_sha256": snapshot["sha256"],
        "solution_source_sha256": response.get("solution_source_sha256"),
        "workload_set_id": workload_set_id,
        "case": case.as_dict(),
        "protocol": protocol.as_dict(),
        "environment": response.get("environment"),
        "correctness": response.get("correctness"),
        "performance": response.get("performance"),
        "failure": response.get("failure"),
    }
    result_path = project_root / "results" / "runs" / f"{run_id}.json"
    atomic_write_json(result_path, result)
    return result, result_path
