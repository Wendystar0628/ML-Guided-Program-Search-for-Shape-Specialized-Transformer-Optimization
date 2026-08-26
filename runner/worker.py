"""Fresh-process worker for one benchmark run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from runner.contracts import atomic_write_json
from runner.execution import execute_benchmark


def _failure(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        kind = "cuda_out_of_memory"
    elif isinstance(exc, KeyboardInterrupt):
        kind = "interrupted"
    else:
        kind = type(exc).__name__
    return {
        "status": "failed",
        "solution_source_sha256": None,
        "environment": None,
        "correctness": None,
        "performance": None,
        "failure": {"kind": kind, "message": str(exc)},
    }


def execute_request(request: dict[str, Any]) -> dict[str, Any]:
    try:
        return execute_benchmark(request)
    except BaseException as exc:  # noqa: BLE001 - this is the worker boundary.
        return _failure(exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Internal benchmark worker")
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args(argv)

    request = json.loads(args.request.read_text(encoding="utf-8"))
    response = execute_request(request)
    atomic_write_json(args.response, response)
    return 0 if response["status"] in {"success", "correctness_failed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
