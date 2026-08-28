from __future__ import annotations

import pytest

from runner.contracts import load_workload_set
from runner.resource_guard import (
    ResourceGuardError,
    ensure_local_benchmark_allowed,
    local_benchmark_shapes,
)
from tests.support.runner_fixtures import PROJECT_ROOT, WORKLOAD_SET_ID, official_shape


def test_default_local_selection_is_exactly_official_01_through_13() -> None:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)

    assert [shape.case_id for shape in local_benchmark_shapes(workload.shapes)] == [
        f"official_{index:02d}" for index in range(1, 14)
    ]


def test_official_14_is_rejected_before_tensor_allocation() -> None:
    with pytest.raises(ResourceGuardError, match="official_14"):
        ensure_local_benchmark_allowed(official_shape("official_14"))


def test_official_13_remains_available_to_explicit_runs() -> None:
    ensure_local_benchmark_allowed(official_shape("official_13"))
