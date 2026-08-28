from __future__ import annotations

from dataclasses import replace

import pytest

from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    RunVariant,
    load_workload_set,
)
from runner.workload_execution import (
    all_benchmark_shapes,
    effective_protocol,
    estimate_dense_attention_bytes,
    plan_workload_execution,
    route_eligible_shapes,
)
from tests.support.runner_fixtures import PROJECT_ROOT, WORKLOAD_SET_ID, official_shape


def test_execution_mode_is_derived_from_memory_not_case_id() -> None:
    original = official_shape("official_14")
    renamed = replace(original, case_id="same_shape_with_another_name")

    original_plan = plan_workload_execution(original)
    renamed_plan = plan_workload_execution(renamed)

    assert original_plan == renamed_plan
    assert original_plan.execution_mode == "batch_streamed"
    assert original_plan.reference_kind == "internal_query_block"
    assert original_plan.validation_microbatch_size == 1
    assert original_plan.timing_microbatch_candidates == (1, 2, 4, 8, 16, 32)
    assert original_plan.formal_eligible is False


def test_attention_limit_boundary_selects_execution_mode() -> None:
    shape = official_shape("official_13")
    variant = RunVariant()
    estimated = estimate_dense_attention_bytes(shape, variant)

    resident = plan_workload_execution(
        shape,
        variant,
        resident_attention_limit_bytes=estimated,
    )
    streamed = plan_workload_execution(
        shape,
        variant,
        resident_attention_limit_bytes=estimated - 1,
    )

    assert resident.execution_mode == "resident"
    assert resident.reference_kind == "live_baseline"
    assert resident.validation_microbatch_size is None
    assert resident.timing_microbatch_candidates == ()
    assert resident.formal_eligible is True
    assert streamed.execution_mode == "batch_streamed"
    assert streamed.validation_microbatch_size == 1
    assert streamed.timing_microbatch_candidates == (1, 2, 4, 8, 16, 32)


@pytest.mark.parametrize(
    ("batch_size", "expected"),
    [
        (1, (1,)),
        (3, (1,)),
        (8, (1, 2, 4, 8)),
        (24, (1, 2, 4, 8)),
        (64, (1, 2, 4, 8, 16, 32)),
    ],
)
def test_streamed_schedule_contains_only_supported_batch_divisors(
    batch_size: int,
    expected: tuple[int, ...],
) -> None:
    shape = replace(
        official_shape("official_14"),
        case_id=f"batch_{batch_size}",
        batch_size=batch_size,
    )

    plan = plan_workload_execution(shape, resident_attention_limit_bytes=1)

    assert plan.validation_microbatch_size == 1
    assert plan.timing_microbatch_candidates == expected


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5])
def test_attention_limit_must_be_a_positive_integer(invalid: object) -> None:
    with pytest.raises(ContractError, match="positive integer"):
        plan_workload_execution(
            official_shape("official_01"),
            resident_attention_limit_bytes=invalid,  # type: ignore[arg-type]
        )


def test_streamed_smoke_protocol_uses_one_bounded_measurement() -> None:
    protocol = MeasurementProtocol.for_preset("smoke", timeout_seconds=10.0)
    plan = plan_workload_execution(official_shape("official_14"))

    result = effective_protocol(protocol, plan)

    assert result.accuracy_trials == 1
    assert result.warmup == 1
    assert result.repeats == 1
    assert result.rounds == 1
    assert result.timeout_seconds == 300.0
    assert result.seed == protocol.seed
    assert result.rtol == protocol.rtol
    assert result.atol == protocol.atol


def test_streamed_formal_protocol_keeps_three_independent_rounds() -> None:
    protocol = MeasurementProtocol.for_preset("formal", timeout_seconds=600.0)
    plan = plan_workload_execution(official_shape("official_14"))

    result = effective_protocol(protocol, plan)

    assert result.accuracy_trials == 1
    assert result.warmup == 2
    assert result.repeats == 1
    assert result.rounds == 3
    assert result.timeout_seconds == 1200.0


def test_resident_protocol_is_not_rewritten() -> None:
    protocol = MeasurementProtocol.for_preset("formal")
    plan = plan_workload_execution(official_shape("official_01"))

    assert effective_protocol(protocol, plan) is protocol


def test_benchmark_and_route_scopes_are_separate() -> None:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)

    benchmark_shapes = all_benchmark_shapes(workload.shapes)
    route_shapes = route_eligible_shapes(workload.shapes)

    assert [shape.case_id for shape in benchmark_shapes] == [
        f"official_{index:02d}" for index in range(1, 15)
    ]
    assert [shape.case_id for shape in route_shapes] == [
        f"official_{index:02d}" for index in range(1, 14)
    ]


def test_route_scope_follows_plan_eligibility_not_a_fixed_exclusion() -> None:
    workload = load_workload_set(PROJECT_ROOT, WORKLOAD_SET_ID)

    route_shapes = route_eligible_shapes(
        workload.shapes,
        resident_attention_limit_bytes=10**15,
    )

    assert route_shapes == workload.shapes
