"""Tests for worker supervision and compact persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from runner import supervisor
from runner.contracts import ContractError
from tests.support.runner_fixtures import PROJECT_ROOT, tiny_case, tiny_protocol


def test_managed_failure_persists_only_known_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_worker(
        project_root: Path,
        request: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        del project_root, request, timeout_seconds
        return {
            "outcome": "timeout",
            "environment": None,
            "probe": None,
            "failure": {
                "stage": "worker",
                "type": "TimeoutExpired",
                "message": "worker exceeded its time limit",
                "exit_code": None,
            },
        }

    monkeypatch.setattr(supervisor, "_run_worker", fake_run_worker)

    result, result_path = supervisor.run_managed_probe(
        tmp_path,
        device="cuda:0",
        timeout_seconds=1.0,
    )

    assert set(result) == {
        "schema_version",
        "run_id",
        "created_at",
        "run_kind",
        "requested_device",
        "outcome",
        "failure",
    }
    assert result["outcome"] == "timeout"
    assert result["failure"] == {
        "stage": "worker",
        "type": "TimeoutExpired",
        "message": "worker exceeded its time limit",
    }
    assert json.loads(result_path.read_text(encoding="utf-8")) == result


def test_correctness_failure_persists_summary_not_trials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_worker(
        project_root: Path,
        request: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        del project_root, request, timeout_seconds
        return {
            "outcome": "invalid_output",
            "solution_source_sha256": "fixture-solution-hash",
            "environment": {
                "device": "cpu",
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
            },
            "correctness": {
                "passed": False,
                "trial_count": 1,
                "failed_elements": 3,
                "max_abs_error": 0.25,
                "max_relative_error": 2.0,
                "trials": [
                    {
                        "seed": 17,
                        "passed": False,
                        "failed_elements": 3,
                        "max_abs_error": 0.25,
                        "max_relative_error": 2.0,
                        "error": "ContractError: output shape mismatch",
                    }
                ],
            },
            "execution_path": {"qkv_projection": "packed"},
            "failure": {
                "stage": "correctness",
                "type": "CorrectnessError",
                "message": "Solution failed the correctness contract",
                "exit_code": None,
            },
        }

    monkeypatch.setattr(supervisor, "_run_worker", fake_run_worker)
    result_dir = tmp_path / "runs"

    result, result_path = supervisor.run_managed_benchmark(
        PROJECT_ROOT,
        workload_set_id="tiny_test_fixture",
        case=tiny_case(),
        protocol=tiny_protocol(),
        device="cpu",
        target="solution",
        workload_sha256="fixture-hash",
        result_dir=result_dir,
    )

    assert result["outcome"] == "invalid_output"
    assert result["correctness"] == {
        "passed": False,
        "trial_count": 1,
        "failed_elements": 3,
        "max_abs_error": 0.25,
        "max_relative_error": 2.0,
        "diagnostic": "ContractError: output shape mismatch",
    }
    assert "performance" not in result
    assert "trials" not in result["correctness"]
    assert "samples_ms" not in json.dumps(result)
    assert result_path.parent == result_dir
    assert json.loads(result_path.read_text(encoding="utf-8")) == result


def test_probe_rejects_failed_device_operation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_worker(
        project_root: Path,
        request: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        del project_root, request, timeout_seconds
        return {
            "outcome": "success",
            "environment": {"device": "cpu", "torch": torch.__version__},
            "probe": {
                "device_operation_passed": False,
                "sdpa": {"available": False, "reason": "cuda_required"},
            },
            "failure": None,
        }

    monkeypatch.setattr(supervisor, "_run_worker", fake_run_worker)

    result, _ = supervisor.run_managed_probe(tmp_path, device="cpu")

    assert result["outcome"] == "runtime_error"
    assert result["probe"]["device_operation_passed"] is False
    assert result["failure"] == {
        "stage": "result_compaction",
        "type": "InvalidWorkerResponse",
        "message": "device operation failed",
    }


@pytest.mark.parametrize("timeout", [0.0, float("nan"), float("inf")])
def test_probe_rejects_invalid_timeout_before_start(
    tmp_path: Path,
    timeout: float,
) -> None:
    with pytest.raises(ContractError, match="timeout_seconds must be finite"):
        supervisor.run_managed_probe(
            tmp_path,
            device="cpu",
            timeout_seconds=timeout,
        )

    assert not (tmp_path / "results").exists()


def test_probe_rejects_invalid_mode_before_start(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="unsupported probe mode"):
        supervisor.run_managed_probe(
            tmp_path,
            device="cpu",
            probe_mode="unknown",
        )

    assert not (tmp_path / "results").exists()
