"""Focused tests for the unified intermediate result layout."""

from __future__ import annotations

import json
from pathlib import Path

from runner import locking, supervisor
from runner.result_layout import (
    final_performance_path,
    final_results_root,
    intermediate_results_dir,
    intermediate_results_root,
)


def test_intermediate_categories_share_one_root(tmp_path: Path) -> None:
    root = intermediate_results_root(tmp_path)

    assert root == tmp_path.resolve() / "results" / "intermediate"
    for category in (
        "runs",
        "sweeps",
        "tuning",
        "probes",
        "profiles",
        "streamed",
        "calibration",
        "experiments",
    ):
        assert intermediate_results_dir(tmp_path, category) == root / category


def test_default_run_persistence_uses_intermediate_runs(tmp_path: Path) -> None:
    result = {"run_id": "example-run", "outcome": "success"}

    persisted, result_path = supervisor._persist_result(tmp_path, result)

    assert persisted == result
    assert result_path == (
        tmp_path.resolve() / "results" / "intermediate" / "runs" / "example-run.json"
    )
    assert json.loads(result_path.read_text(encoding="utf-8")) == result


def test_all_runtime_locks_share_intermediate_lock_directory(tmp_path: Path) -> None:
    bundle_root = tmp_path / "verified_hardware" / "example_gpu"
    expected = tmp_path.resolve() / "results" / "intermediate" / ".locks"

    assert locking._device_measurement_lock_path(tmp_path, "cuda:0").parent == expected
    assert locking.hardware_bundle_lock_path(tmp_path, "example_gpu").parent == expected
    assert locking.bundle_lock_path(bundle_root).parent == expected


def test_final_performance_is_grouped_by_hardware(tmp_path: Path) -> None:
    assert final_results_root(tmp_path) == tmp_path.resolve() / "results" / "final"
    assert final_performance_path(tmp_path, "fixture_gpu") == (
        tmp_path.resolve() / "results" / "final" / "fixture_gpu.json"
    )
