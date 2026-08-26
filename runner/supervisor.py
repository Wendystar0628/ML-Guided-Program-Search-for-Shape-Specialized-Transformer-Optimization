"""Run requests in fresh processes and persist compact result documents."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import psutil

from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    WorkloadCase,
    atomic_write_json,
    load_json,
    new_run_id,
    utc_now,
    validate_official_snapshot,
)

_ALLOWED_OUTCOMES = {
    "success",
    "invalid_output",
    "unsupported",
    "build_error",
    "oom",
    "timeout",
    "cancelled",
    "runtime_error",
}


def _legacy_status(outcome: str) -> str:
    if outcome == "invalid_output":
        return "correctness_failed"
    if outcome == "cancelled":
        return "interrupted"
    return outcome


def _failure_response(
    outcome: str,
    *,
    stage: str,
    failure_type: str,
    message: str,
    exit_code: int | None,
) -> dict[str, Any]:
    return {
        "outcome": outcome,
        "status": _legacy_status(outcome),
        "solution_source_sha256": None,
        "environment": None,
        "correctness": None,
        "performance": None,
        "profile": None,
        "probe": None,
        "path": None,
        "failure": {
            "stage": stage,
            "type": failure_type,
            "message": message,
            "exit_code": exit_code,
        },
    }


def _stop_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    processes: list[psutil.Process] = []
    try:
        root = psutil.Process(process.pid)
        processes.extend(root.children(recursive=True))
        processes.append(root)
        for child in processes[:-1]:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        try:
            root.terminate()
        except psutil.NoSuchProcess:
            pass
        _, alive = psutil.wait_procs(processes, timeout=5)
        for remaining in alive:
            try:
                remaining.kill()
            except psutil.NoSuchProcess:
                pass
        if alive:
            psutil.wait_procs(alive, timeout=5)
    except (psutil.Error, OSError):
        pass
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _worker_response_error(
    response: dict[str, Any],
    run_kind: str,
) -> str | None:
    outcome = response.get("outcome")
    if outcome not in _ALLOWED_OUTCOMES:
        return f"worker returned unsupported outcome: {outcome!r}"
    if outcome != "success":
        if not isinstance(response.get("failure"), dict):
            return "failed worker response is missing failure details"
        return None
    if response.get("failure") is not None:
        return "successful worker response contains failure details"
    if not isinstance(response.get("environment"), dict):
        return "successful worker response is missing environment details"
    required_payload = {
        "benchmark": "performance",
        "profile": "profile",
        "probe": "probe",
    }.get(run_kind)
    if required_payload is None:
        return f"unsupported worker run_kind: {run_kind!r}"
    if not isinstance(response.get(required_payload), dict):
        return f"successful {run_kind} response is missing {required_payload}"
    if run_kind in {"benchmark", "profile"} and not isinstance(
        response.get("path"), dict
    ):
        return f"successful {run_kind} response is missing target path"
    if run_kind == "benchmark" and not isinstance(response.get("correctness"), dict):
        return "successful benchmark response is missing correctness"
    return None


def _run_worker_inner(
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
        popen_options: dict[str, Any] = {
            "cwd": project_root,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if os.name == "nt":
            popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_options["start_new_session"] = True
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "runner.worker",
                str(request_path),
                str(response_path),
            ],
            **popen_options,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _stop_process_tree(process)
            return _failure_response(
                "timeout",
                stage="worker",
                failure_type="TimeoutExpired",
                message=f"runner worker exceeded {timeout_seconds:g} seconds",
                exit_code=process.returncode,
            )
        except KeyboardInterrupt:
            _stop_process_tree(process)
            return _failure_response(
                "cancelled",
                stage="worker",
                failure_type="KeyboardInterrupt",
                message="runner worker was cancelled by the user",
                exit_code=process.returncode,
            )

        if response_path.is_file():
            try:
                response = load_json(response_path)
            except ContractError as exc:
                return _failure_response(
                    "runtime_error",
                    stage="worker_response",
                    failure_type=type(exc).__name__,
                    message=str(exc),
                    exit_code=process.returncode,
                )
            outcome = response.get("outcome")
            if not isinstance(outcome, str):
                legacy = response.get("status")
                outcome = {
                    "correctness_failed": "invalid_output",
                    "interrupted": "cancelled",
                    "failed": "runtime_error",
                }.get(str(legacy), str(legacy or "runtime_error"))
                response["outcome"] = outcome
            response.setdefault("status", _legacy_status(outcome))
            failure = response.get("failure")
            if isinstance(failure, dict):
                failure["exit_code"] = process.returncode
            response_error = _worker_response_error(
                response, str(request.get("run_kind", "benchmark"))
            )
            if response_error is not None:
                return _failure_response(
                    "runtime_error",
                    stage="worker_response",
                    failure_type="InvalidWorkerResponse",
                    message=response_error,
                    exit_code=process.returncode,
                )
            if outcome == "success" and process.returncode != 0:
                message = stderr.strip()[-4000:] or stdout.strip()[-4000:]
                return _failure_response(
                    "runtime_error",
                    stage="worker_exit",
                    failure_type="UnexpectedExitCode",
                    message=message or f"worker exited with code {process.returncode}",
                    exit_code=process.returncode,
                )
            return response

        message = stderr.strip()[-4000:] or stdout.strip()[-4000:]
        return _failure_response(
            "runtime_error",
            stage="worker",
            failure_type="WorkerFailed",
            message=message or f"worker exited with code {process.returncode}",
            exit_code=process.returncode,
        )


def _run_worker(
    project_root: Path,
    request: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        return _run_worker_inner(project_root, request, timeout_seconds)
    except KeyboardInterrupt:
        return _failure_response(
            "cancelled",
            stage="worker_start",
            failure_type="KeyboardInterrupt",
            message="runner request was cancelled before the worker started",
            exit_code=None,
        )
    except (ContractError, OSError, ValueError, subprocess.SubprocessError) as exc:
        return _failure_response(
            "runtime_error",
            stage="worker_start",
            failure_type=type(exc).__name__,
            message=str(exc),
            exit_code=None,
        )


def _persist_result(
    project_root: Path,
    result: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    result_path = project_root / "results" / "runs" / f"{result['run_id']}.json"
    atomic_write_json(result_path, result)
    return result, result_path


def _workload_record(
    workload_set_id: str,
    case: WorkloadCase,
    workload_sha256: str | None,
) -> dict[str, Any]:
    return {
        "set_id": workload_set_id,
        "case_id": case.case_id,
        "sha256": workload_sha256,
        "signature": case.as_dict(),
    }


def run_managed_benchmark(
    project_root: Path,
    *,
    workload_set_id: str,
    case: WorkloadCase,
    protocol: MeasurementProtocol,
    device: str,
    target: str = "solution",
    workload_sha256: str | None = None,
) -> tuple[dict[str, Any], Path]:
    project_root = project_root.resolve()
    run_id = new_run_id()
    created_at = utc_now()
    snapshot = validate_official_snapshot(project_root)
    request = {
        "run_kind": "benchmark",
        "project_root": str(project_root),
        "case": case.as_dict(),
        "protocol": protocol.as_dict(),
        "device": device,
        "target": target,
    }
    response = _run_worker(project_root, request, protocol.timeout_seconds)
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": created_at,
        "completed_at": utc_now(),
        "run_kind": "benchmark",
        "preset": protocol.preset,
        "target": target,
        "outcome": response["outcome"],
        "status": response["status"],
        "official_snapshot_sha256": snapshot["sha256"],
        "solution_source_sha256": response.get("solution_source_sha256"),
        "workload": _workload_record(workload_set_id, case, workload_sha256),
        "workload_set_id": workload_set_id,
        "case": case.as_dict(),
        "protocol": protocol.as_dict(),
        "environment": response.get("environment"),
        "correctness": response.get("correctness"),
        "performance": response.get("performance"),
        "path": response.get("path") or {"requested": target, "resolved": None},
        "execution_path": response.get("execution_path"),
        "failure": response.get("failure"),
    }
    return _persist_result(project_root, result)


def run_managed_profile(
    project_root: Path,
    *,
    workload_set_id: str,
    case: WorkloadCase,
    protocol: MeasurementProtocol,
    device: str,
    target: str,
    workload_sha256: str | None = None,
) -> tuple[dict[str, Any], Path]:
    project_root = project_root.resolve()
    run_id = new_run_id()
    created_at = utc_now()
    snapshot = validate_official_snapshot(project_root)
    request = {
        "run_kind": "profile",
        "project_root": str(project_root),
        "case": case.as_dict(),
        "protocol": protocol.as_dict(),
        "device": device,
        "target": target,
    }
    response = _run_worker(project_root, request, protocol.timeout_seconds)
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": created_at,
        "completed_at": utc_now(),
        "run_kind": "profile",
        "preset": protocol.preset,
        "target": target,
        "outcome": response["outcome"],
        "status": response["status"],
        "official_snapshot_sha256": snapshot["sha256"],
        "solution_source_sha256": response.get("solution_source_sha256"),
        "workload": _workload_record(workload_set_id, case, workload_sha256),
        "workload_set_id": workload_set_id,
        "case": case.as_dict(),
        "protocol": protocol.as_dict(),
        "environment": response.get("environment"),
        "correctness": response.get("correctness"),
        "profile": response.get("profile"),
        "path": response.get("path") or {"requested": target, "resolved": None},
        "execution_path": response.get("execution_path"),
        "failure": response.get("failure"),
    }
    return _persist_result(project_root, result)


def run_managed_probe(
    project_root: Path,
    *,
    device: str,
    timeout_seconds: float = 30.0,
) -> tuple[dict[str, Any], Path]:
    if timeout_seconds <= 0:
        raise ContractError("timeout_seconds must be positive")
    project_root = project_root.resolve()
    run_id = new_run_id()
    created_at = utc_now()
    response = _run_worker(
        project_root,
        {"run_kind": "probe", "device": device},
        timeout_seconds,
    )
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": created_at,
        "completed_at": utc_now(),
        "run_kind": "probe",
        "preset": None,
        "target": "device",
        "outcome": response["outcome"],
        "status": response["status"],
        "environment": response.get("environment"),
        "probe": response.get("probe"),
        "path": response.get("path") or {"requested": device, "resolved": None},
        "failure": response.get("failure"),
    }
    return _persist_result(project_root, result)
