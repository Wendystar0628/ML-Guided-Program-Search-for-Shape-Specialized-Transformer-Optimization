"""Run requests in fresh processes and persist compact result documents."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
import threading
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
from runner.result_contracts import (
    compact_correctness,
    compact_performance,
    parse_worker_response,
    validate_benchmark_performance,
    validate_correctness,
)


def _validate_timeout_seconds(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
        or float(value) > threading.TIMEOUT_MAX
    ):
        raise ContractError(
            "timeout_seconds must be finite, positive, and supported by this platform"
        )
    return float(value)


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
        "solution_source_sha256": None,
        "environment": None,
        "correctness": None,
        "performance": None,
        "profile": None,
        "probe": None,
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


def _run_worker_inner(
    project_root: Path,
    request: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    timeout_seconds = _validate_timeout_seconds(timeout_seconds)
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
        except (OSError, ValueError, OverflowError, subprocess.SubprocessError) as exc:
            _stop_process_tree(process)
            return _failure_response(
                "runtime_error",
                stage="worker",
                failure_type=type(exc).__name__,
                message=str(exc),
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
            failure = response.get("failure")
            if isinstance(failure, dict):
                failure["exit_code"] = process.returncode
            try:
                response = parse_worker_response(
                    response,
                    run_kind=str(request.get("run_kind", "benchmark")),
                )
            except ContractError as exc:
                return _failure_response(
                    "runtime_error",
                    stage="worker_response",
                    failure_type="InvalidWorkerResponse",
                    message=str(exc),
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
    *,
    result_dir: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    output_dir = result_dir or project_root / "results" / "runs"
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    result_path = output_dir.resolve() / f"{result['run_id']}.json"
    atomic_write_json(result_path, result)
    return result, result_path


def _compact_failure(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    compact = {
        "stage": str(value.get("stage", "unknown")),
        "type": str(value.get("type", "RuntimeError")),
        "message": str(value.get("message", ""))[-1000:],
    }
    if value.get("exit_code") is not None:
        compact["exit_code"] = value["exit_code"]
    return compact


def _success_contract_error(result: dict[str, Any]) -> str | None:
    if result.get("outcome") != "success":
        return None
    environment = result.get("environment")
    if not isinstance(environment, dict) or not isinstance(
        environment.get("device"), str
    ):
        return "missing compact environment"

    run_kind = result.get("run_kind")
    if run_kind in {"benchmark", "profile"}:
        if not isinstance(result.get("workload"), dict):
            return "missing compact workload"
        if not isinstance(result.get("protocol"), dict):
            return "missing compact protocol"
        source = result.get("source")
        if not isinstance(source, dict) or not isinstance(
            source.get("official_sha256"), str
        ):
            return "missing compact source identity"
        if result.get("target") == "solution" and not isinstance(
            source.get("solution_sha256"), str
        ):
            return "missing Solution source identity"
        if not isinstance(result.get("execution_path"), dict):
            return "missing compact execution path"

    if run_kind == "benchmark":
        performance = result.get("performance")
        protocol = result["protocol"]
        repeats = protocol.get("repeats")
        rounds = protocol.get("rounds")
        expected_timer = (
            "cuda_event"
            if result["environment"]["device"].startswith("cuda")
            else "perf_counter_ns"
        )
        _parsed_performance, performance_error = validate_benchmark_performance(
            performance,
            target=str(result.get("target")),
            repeats=repeats,
            rounds=rounds,
            expected_timer=expected_timer,
        )
        if performance_error is not None:
            return f"invalid compact benchmark performance: {performance_error}"
        if result.get("target") == "solution":
            correctness = result.get("correctness")
            correctness_error = validate_correctness(
                correctness, expected_trials=protocol.get("accuracy_trials")
            )
            if correctness_error is not None:
                return f"invalid compact correctness: {correctness_error}"
    elif run_kind == "profile":
        profile = result.get("profile")
        if not isinstance(profile, dict) or not profile.get("operator_hotspots"):
            return "missing compact profile hotspots"
        if result.get("target") == "solution":
            correctness = result.get("correctness")
            protocol = result["protocol"]
            correctness_error = validate_correctness(
                correctness, expected_trials=protocol.get("accuracy_trials")
            )
            if correctness_error is not None:
                return f"invalid compact correctness: {correctness_error}"
    elif run_kind == "probe":
        probe = result.get("probe")
        if not isinstance(probe, dict):
            return "missing compact probe"
        if probe.get("device_operation_passed") is not True:
            return "device operation failed"
    else:
        return f"unsupported compact run kind: {run_kind!r}"
    return None


def _enforce_success_contract(result: dict[str, Any]) -> None:
    error = _success_contract_error(result)
    if error is None:
        return
    result["outcome"] = "runtime_error"
    result.pop("performance", None)
    result.pop("profile", None)
    result["failure"] = {
        "stage": "result_compaction",
        "type": "InvalidWorkerResponse",
        "message": error,
    }


def _workload_record(
    workload_set_id: str,
    case: WorkloadCase,
    workload_sha256: str | None,
) -> dict[str, Any]:
    return {
        "set_id": workload_set_id,
        "sha256": workload_sha256,
        "case": case.as_dict(),
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
    sweep_id: str | None = None,
    tuning_id: str | None = None,
    candidate_id: str | None = None,
    result_dir: Path | None = None,
    solution_policy: str | None = None,
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
    if solution_policy is not None and target == "solution":
        request["solution_policy"] = solution_policy
    response = _run_worker(project_root, request, protocol.timeout_seconds)
    source = {"official_sha256": snapshot["sha256"]}
    if response.get("solution_source_sha256") is not None:
        source["solution_sha256"] = response["solution_source_sha256"]
    result: dict[str, Any] = {
        "schema_version": 2,
        "run_id": run_id,
    }
    if sweep_id is not None:
        result["sweep_id"] = sweep_id
    if tuning_id is not None or candidate_id is not None:
        if not tuning_id or not candidate_id:
            raise ContractError("tuning_id and candidate_id must be provided together")
        result["tuning"] = {
            "id": tuning_id,
            "candidate_id": candidate_id,
        }
    result.update(
        {
            "created_at": created_at,
            "run_kind": "benchmark",
            "target": target,
            "requested_device": device,
            "outcome": response["outcome"],
            "workload": _workload_record(workload_set_id, case, workload_sha256),
            "source": source,
        }
    )
    correctness = compact_correctness(response.get("correctness"))
    if target == "solution" and correctness is not None:
        result["correctness"] = correctness
    if response["outcome"] == "success":
        performance = compact_performance(response.get("performance"), target)
        if performance is not None:
            result["performance"] = performance
    execution_path = response.get("execution_path")
    if isinstance(execution_path, dict):
        result["execution_path"] = execution_path
    environment = response.get("environment")
    if isinstance(environment, dict):
        result["environment"] = environment
    result["protocol"] = protocol.as_dict()
    failure = _compact_failure(response.get("failure"))
    if failure is not None:
        result["failure"] = failure
    _enforce_success_contract(result)
    return _persist_result(project_root, result, result_dir=result_dir)


def run_managed_profile(
    project_root: Path,
    *,
    workload_set_id: str,
    case: WorkloadCase,
    protocol: MeasurementProtocol,
    device: str,
    target: str,
    workload_sha256: str | None = None,
    solution_policy: str | None = None,
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
    if solution_policy is not None and target == "solution":
        request["solution_policy"] = solution_policy
    response = _run_worker(project_root, request, protocol.timeout_seconds)
    source = {"official_sha256": snapshot["sha256"]}
    if response.get("solution_source_sha256") is not None:
        source["solution_sha256"] = response["solution_source_sha256"]
    result: dict[str, Any] = {
        "schema_version": 2,
        "run_id": run_id,
        "created_at": created_at,
        "run_kind": "profile",
        "target": target,
        "requested_device": device,
        "outcome": response["outcome"],
        "workload": _workload_record(workload_set_id, case, workload_sha256),
        "source": source,
    }
    correctness = compact_correctness(response.get("correctness"))
    if correctness is not None:
        result["correctness"] = correctness
    if response["outcome"] == "success":
        profile = response.get("profile")
        if isinstance(profile, dict):
            result["profile"] = profile
    execution_path = response.get("execution_path")
    if isinstance(execution_path, dict):
        result["execution_path"] = execution_path
    environment = response.get("environment")
    if isinstance(environment, dict):
        result["environment"] = environment
    result["protocol"] = protocol.as_dict()
    failure = _compact_failure(response.get("failure"))
    if failure is not None:
        result["failure"] = failure
    _enforce_success_contract(result)
    return _persist_result(project_root, result)


def run_managed_probe(
    project_root: Path,
    *,
    device: str,
    timeout_seconds: float = 30.0,
    matmul_precision: str = "high",
    allow_tf32: bool = True,
    probe_mode: str = "diagnostic",
) -> tuple[dict[str, Any], Path]:
    timeout_seconds = _validate_timeout_seconds(timeout_seconds)
    if matmul_precision not in {"highest", "high", "medium"}:
        raise ContractError(f"unsupported matmul precision: {matmul_precision}")
    if not isinstance(allow_tf32, bool):
        raise ContractError("allow_tf32 must be a boolean")
    if probe_mode not in {"routing", "diagnostic"}:
        raise ContractError(f"unsupported probe mode: {probe_mode}")
    project_root = project_root.resolve()
    run_id = new_run_id()
    created_at = utc_now()
    response = _run_worker(
        project_root,
        {
            "run_kind": "probe",
            "device": device,
            "probe_mode": probe_mode,
            "matmul_precision": matmul_precision,
            "allow_tf32": allow_tf32,
        },
        timeout_seconds,
    )
    result: dict[str, Any] = {
        "schema_version": 2,
        "run_id": run_id,
        "created_at": created_at,
        "run_kind": "probe",
        "requested_device": device,
        "outcome": response["outcome"],
    }
    probe = response.get("probe")
    if isinstance(probe, dict):
        result["probe"] = probe
    environment = response.get("environment")
    if isinstance(environment, dict):
        result["environment"] = environment
    failure = _compact_failure(response.get("failure"))
    if failure is not None:
        result["failure"] = failure
    _enforce_success_contract(result)
    return _persist_result(project_root, result)
