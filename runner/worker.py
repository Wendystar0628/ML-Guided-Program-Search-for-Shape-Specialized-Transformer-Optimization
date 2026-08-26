"""Fresh-process worker for one managed runner request."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from runner.contracts import ContractError, atomic_write_json, load_json
from runner.execution import execute_benchmark, execute_profile
from runner.probe import execute_probe


def _failure(exc: BaseException, run_kind: str) -> dict[str, Any]:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        outcome = "oom"
    elif isinstance(exc, KeyboardInterrupt):
        outcome = "cancelled"
    elif isinstance(exc, ContractError):
        if run_kind == "probe":
            outcome = "unsupported"
        elif run_kind == "request":
            outcome = "runtime_error"
        else:
            outcome = "build_error"
    else:
        outcome = "runtime_error"
    return {
        "outcome": outcome,
        "solution_source_sha256": None,
        "environment": None,
        "correctness": None,
        "performance": None,
        "profile": None,
        "probe": None,
        "failure": {
            "stage": run_kind,
            "type": type(exc).__name__,
            "message": str(exc),
            "exit_code": None,
        },
    }


def execute_request(request: dict[str, Any]) -> dict[str, Any]:
    run_kind = str(request.get("run_kind", "benchmark"))
    try:
        if run_kind == "benchmark":
            return execute_benchmark(request)
        if run_kind == "profile":
            return execute_profile(request)
        if run_kind == "probe":
            return execute_probe(request)
        raise ContractError(f"unsupported run_kind: {run_kind}")
    except BaseException as exc:  # noqa: BLE001 - this is the worker boundary.
        return _failure(exc, run_kind)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Internal managed runner worker")
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args(argv)

    try:
        request = load_json(args.request)
    except BaseException as exc:  # noqa: BLE001 - this is the process boundary.
        response = _failure(exc, "request")
    else:
        response = execute_request(request)
    atomic_write_json(args.response, response)
    outcome = response.get("outcome")
    if outcome == "success":
        return 0
    if outcome == "invalid_output":
        return 2
    if outcome == "cancelled":
        return 130
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
