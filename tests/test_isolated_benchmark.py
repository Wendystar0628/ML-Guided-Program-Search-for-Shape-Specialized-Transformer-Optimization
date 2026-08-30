from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

import benchmarking.suite as suite_module
from benchmarking.device_queue import (
    DeviceLease,
    DeviceLeaseTimeout,
    run_in_fresh_process,
)
from benchmarking.protocols import RunVariant

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _current_process_id() -> int:
    return os.getpid()


def _passed_result(case_id: str, speedup: float = 2.0) -> dict[str, object]:
    return {
        "case_id": case_id,
        "config_id": f"config-{case_id}",
        "passed": True,
        "max_tolerance_ratio": 0.25,
        "median_ms": 1.0,
        "p90_ms": 1.1,
        "baseline_median_ms": speedup,
        "speedup": speedup,
        "peak_memory_bytes": 1024,
        "execution_matches": True,
    }


def test_suite_runs_each_shape_serially_in_a_fresh_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[object, str]] = []

    def run_worker(target: object, *args: object) -> dict[str, object]:
        case_id = str(args[1])
        calls.append((target, case_id))
        return _passed_result(case_id)

    monkeypatch.setattr(suite_module, "run_in_fresh_process", run_worker)
    case_ids = ("official_01", "official_02", "official_03")

    result = suite_module.run_benchmark_suite(
        project_root=PROJECT_ROOT,
        case_ids=case_ids,
        config_path=None,
        variant=RunVariant(),
        preset="smoke",
        device="cuda:0",
        output_directory=tmp_path / "run",
    )

    assert calls == [
        (suite_module._measure_one_shape, case_id) for case_id in case_ids
    ]
    assert [item["case_id"] for item in result.summary["shapes"]] == list(case_ids)
    assert result.summary["progress"] == {
        "completed": 3,
        "total": 3,
        "passed": 3,
    }
    assert result.summary["status"] == "completed"
    assert result.exit_code == 0


def test_fresh_process_uses_a_different_process() -> None:
    assert run_in_fresh_process(_current_process_id) != os.getpid()


def test_worker_failure_is_recorded_and_does_not_block_the_next_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def run_worker(_target: object, *args: object) -> dict[str, object]:
        case_id = str(args[1])
        calls.append(case_id)
        if case_id == "official_01":
            raise suite_module.IsolatedProcessError("worker exited with code 7")
        return _passed_result(case_id)

    monkeypatch.setattr(suite_module, "run_in_fresh_process", run_worker)

    result = suite_module.run_benchmark_suite(
        project_root=PROJECT_ROOT,
        case_ids=("official_01", "official_02"),
        config_path=None,
        variant=RunVariant(),
        preset="smoke",
        device="cuda:0",
        output_directory=tmp_path / "run",
    )

    assert calls == ["official_01", "official_02"]
    assert result.exit_code == 1
    assert result.summary["status"] == "completed_with_failures"
    assert result.summary["progress"] == {
        "completed": 2,
        "total": 2,
        "passed": 1,
    }
    assert result.summary["resident_geomean_speedup"] is None
    assert result.summary["shapes"][0] == {
        "case_id": "official_01",
        "status": "failed",
        "passed": False,
        "error": "worker exited with code 7",
    }
    assert result.summary["shapes"][1]["case_id"] == "official_02"
    persisted = json.loads(result.path.read_text(encoding="utf-8"))
    assert persisted == result.summary


def test_resident_geomean_requires_a_complete_passing_shape_set() -> None:
    speedups = [1.0 + index / 10.0 for index in range(1, 14)]
    complete = [
        _passed_result(f"official_{index:02d}", speedup)
        for index, speedup in enumerate(speedups, start=1)
    ]
    expected = math.exp(sum(math.log(value) for value in speedups) / len(speedups))

    assert suite_module._resident_geomean(PROJECT_ROOT, complete) == pytest.approx(
        expected
    )
    assert suite_module._resident_geomean(PROJECT_ROOT, complete[:-1]) is None

    failed = [dict(item) for item in complete]
    failed[5]["passed"] = False
    failed[5]["status"] = "failed"
    assert suite_module._resident_geomean(PROJECT_ROOT, failed) is None


def test_device_lease_excludes_a_second_process_and_recovers_after_crash(
    tmp_path: Path,
) -> None:
    script = """
import sys
import time
from pathlib import Path

from benchmarking.device_queue import DeviceLease

lease = DeviceLease(device="cuda:0", root=Path(sys.argv[1]), timeout_seconds=1.0)
lease.__enter__()
print("locked", flush=True)
time.sleep(60)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "locked"

        with pytest.raises(DeviceLeaseTimeout), DeviceLease(
            device="cuda:0",
            root=tmp_path,
            timeout_seconds=0.05,
        ):
            pass

        process.kill()
        process.wait(timeout=5)

        with DeviceLease(device="cuda:0", root=tmp_path, timeout_seconds=1.0):
            pass
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
