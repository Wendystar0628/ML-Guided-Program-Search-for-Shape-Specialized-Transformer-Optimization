from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

import benchmarking.ablation as ablation_module
from benchmarking.ablation import (
    ABLATION_FAMILIES,
    DEFAULT_ABLATION_SHAPES,
    AblationFamily,
    ablation_protocol,
    build_ablation_candidate,
    run_component_ablation_suite,
    write_component_ablation_csv,
)
from benchmarking.protocols import load_shape
from solution.config import (
    ConfigSpec,
    InitialNormBackend,
    PrecisionPlan,
    ProjectionBackend,
    RuntimeBackend,
)
from solution.plan import ExecutionContext
from solution.plan_builder import HardwareCapabilities, PlanBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deployed_for(case_id: str) -> ConfigSpec:
    shapes = json.loads(
        (PROJECT_ROOT / "official" / "test_shapes.json").read_text(encoding="utf-8")
    )["ordered_shapes"]
    shape = next(value for value in shapes if value["case_id"] == case_id)
    document = json.loads(
        (PROJECT_ROOT / "deployment" / "deployed_configs.json").read_text(
            encoding="utf-8"
        )
    )
    entries = document["bundles"][0]["entries"]
    keys = ("batch_size", "qkv_dim", "heads", "seq_len", "layers", "causal", "ffn_dim")
    entry = next(
        value
        for value in entries
        if all(value["shape"][key] == shape[key] for key in keys)
    )
    return ConfigSpec.from_dict(entry["config"])


def _hardware() -> HardwareCapabilities:
    return HardwareCapabilities(
        device_type="cuda",
        compute_capability=(8, 9),
        shared_memory_per_block=100_000,
        mem_efficient_sdp=True,
        cudnn_sdp=True,
        cudnn_available=True,
        torch_compile=True,
        triton_shape13_attention=True,
        triton_dh8_attention=True,
        triton_residual_norm=True,
        triton_mixed_residual_norm=True,
        triton_initial_norm=True,
        triton_exact_gelu=True,
        triton_streaming_dh64_attention=True,
        triton_qkv_native_bhsd=True,
        triton_attention_output_projection=True,
        triton_linear_exact_gelu=True,
        triton_d32_residual_norm=True,
        triton_masked_norm=True,
        triton_linear_residual_norm=True,
        triton_fused_ffn_residual_norm=True,
    )


def _context(case_id: str) -> ExecutionContext:
    shape = load_shape(PROJECT_ROOT, case_id)
    return ExecutionContext(
        batch_size=shape.batch_size,
        seq_len=shape.seq_len,
        d_model=shape.d_model,
        num_heads=shape.num_heads,
        causal=shape.causal,
        device=torch.device("cuda:0"),
        dtype=torch.float32,
        training=False,
        grad_enabled=False,
        input_contiguous=True,
        has_valid_token_mask=False,
        mask_compatible=True,
        ffn_dim=shape.ffn_dim,
        num_layers=shape.num_layers,
    )


def test_protocol_compacts_only_the_expensive_shape() -> None:
    regular = ablation_protocol("official_02")
    expensive = ablation_protocol("official_06")

    assert (
        regular.accuracy_trials,
        regular.warmup,
        regular.repeats,
        regular.rounds,
    ) == (1, 2, 5, 5)
    assert (
        expensive.accuracy_trials,
        expensive.warmup,
        expensive.repeats,
        expensive.rounds,
    ) == (1, 2, 5, 3)


def test_retained_performance_is_exact_ratio_of_aggregate_medians() -> None:
    slowdown, retained = ablation_module._aggregate_ablation_effect(
        deployed_median_ms=2.0,
        ablated_median_ms=5.0,
    )

    assert slowdown == pytest.approx(2.5)
    assert retained == pytest.approx(0.4)


def test_default_scope_covers_every_resident_shape_in_order() -> None:
    assert DEFAULT_ABLATION_SHAPES == tuple(
        f"official_{index:02d}" for index in range(1, 14)
    )


def test_runtime_ablation_is_immutable_and_clears_runtime_only_fields() -> None:
    shape = load_shape(PROJECT_ROOT, "official_02")
    deployed = _deployed_for(shape.case_id)
    original = deployed.to_dict()

    candidate = build_ablation_candidate(
        deployed,
        AblationFamily.RUNTIME,
        shape,
    )

    assert candidate is not None
    assert deployed.to_dict() == original
    assert candidate.config.schedule.runtime is RuntimeBackend.EAGER
    assert candidate.config.schedule.compile_mode is None
    assert candidate.config.schedule.batch_tile_size is None
    assert candidate.config.schedule.microbatch_size is None
    assert not candidate.config.schedule.reuse_unchanged_input
    assert any(
        item["field"] == "schedule.runtime" for item in candidate.changed_fields
    )


def test_runtime_dependency_closure_is_explicit_for_shape_07() -> None:
    shape = load_shape(PROJECT_ROOT, "official_07")
    candidate = build_ablation_candidate(
        _deployed_for(shape.case_id),
        AblationFamily.RUNTIME,
        shape,
    )

    assert candidate is not None
    assert candidate.variant_kind == "dependency_closure"
    assert candidate.config.program.initial_norm is InitialNormBackend.TORCH


def test_shape_06_runtime_is_capacity_excluded() -> None:
    shape = load_shape(PROJECT_ROOT, "official_06")
    assert (
        build_ablation_candidate(
            _deployed_for(shape.case_id),
            AblationFamily.RUNTIME,
            shape,
        )
        is None
    )


def test_shape_08_has_a_clean_projection_precision_ablation() -> None:
    shape = load_shape(PROJECT_ROOT, "official_08")
    candidate = build_ablation_candidate(
        _deployed_for(shape.case_id),
        AblationFamily.PROJECTION,
        shape,
    )

    assert candidate is not None
    assert candidate.variant_kind == "atomic"
    assert candidate.config.program.precision_plan is PrecisionPlan.INPUT_DTYPE
    assert {
        candidate.config.program.qkv_projection,
        candidate.config.program.attention_output_projection,
        candidate.config.program.ffn_input_projection,
        candidate.config.program.ffn_output_projection,
    } == {ProjectionBackend.INPUT_DTYPE}


@pytest.mark.parametrize(
    "case_id",
    tuple(f"official_{index:02d}" for index in range(1, 14)),
)
def test_every_generated_candidate_is_a_legal_plan(case_id: str) -> None:
    shape = load_shape(PROJECT_ROOT, case_id)
    deployed = _deployed_for(case_id)
    builder = PlanBuilder()
    assert builder.evaluate(deployed, _context(case_id), _hardware()).accepted

    for family in ABLATION_FAMILIES:
        candidate = build_ablation_candidate(deployed, family, shape)
        if candidate is None:
            continue
        result = builder.evaluate(candidate.config, _context(case_id), _hardware())
        assert result.accepted, (
            case_id,
            family,
            [violation.to_dict() for violation in result.violations],
        )


def test_suite_uses_fixed_serial_case_family_order_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case_ids = ("official_02", "official_08", "official_13")
    families = (AblationFamily.RUNTIME,)
    deployed = {case_id: _deployed_for(case_id) for case_id in case_ids}
    calls: list[str] = []

    monkeypatch.setattr(
        ablation_module,
        "_deployed_configs",
        lambda *_args, **_kwargs: deployed,
    )

    def run_worker(_target: object, *args: object) -> dict[str, object]:
        shape = args[0]
        calls.append(shape.case_id)
        return {
            "status": "measured",
            "deployed": {
                "median_ms": 1.0,
                "p90_ms": 1.1,
                "passed": True,
                "execution_matches": True,
                "max_tolerance_ratio": 0.1,
                "peak_memory_bytes": 100,
            },
            "ablated": {
                "median_ms": 2.0,
                "p90_ms": 2.1,
                "passed": True,
                "execution_matches": True,
                "max_tolerance_ratio": 0.1,
                "peak_memory_bytes": 100,
            },
            "ablation_slowdown": 2.0,
            "retained_performance_fraction": 0.5,
            "paired_ablation_slowdowns": [2.0] * 5,
        }

    monkeypatch.setattr(ablation_module, "run_in_fresh_process", run_worker)
    result = run_component_ablation_suite(
        project_root=PROJECT_ROOT,
        case_ids=case_ids,
        families=families,
        output_directory=tmp_path / "run",
    )

    assert calls == ["official_02", "official_08"]
    assert [item["status"] for item in result.summary["comparisons"]] == [
        "measured",
        "measured",
        "not_applicable",
    ]
    assert result.summary["progress"] == {
        "completed": 3,
        "total": 3,
        "measured": 2,
    }
    assert result.exit_code == 0


def test_suite_marks_active_coupled_projection_as_not_isolatable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case_id = "official_01"
    deployed = _deployed_for(case_id)
    monkeypatch.setattr(
        ablation_module,
        "_deployed_configs",
        lambda *_args, **_kwargs: {case_id: deployed},
    )

    result = run_component_ablation_suite(
        project_root=PROJECT_ROOT,
        case_ids=(case_id,),
        families=(AblationFamily.PROJECTION,),
        output_directory=tmp_path / "run",
    )

    comparison = result.summary["comparisons"][0]
    assert comparison["status"] == "not_isolatable"
    assert comparison["variant_kind"] == "dependency_coupled"
    assert result.summary["schema_version"] == 2


def test_plot_csv_keeps_measured_values_and_marks_non_measurements(
    tmp_path: Path,
) -> None:
    summary = {
        "comparisons": [
            {
                "case_id": "official_02",
                "mechanism_id": "runtime_schedule",
                "mechanism_label": "Runtime schedule",
                "status": "measured",
                "variant_kind": "atomic",
                "ablation_slowdown": 1.5,
                "retained_performance_fraction": 2.0 / 3.0,
                "deployed": {"median_ms": 1.0},
                "ablated": {"median_ms": 1.5},
                "protocol": {"repeats": 5, "rounds": 5},
                "completed_at": "2026-09-01T00:00:00+00:00",
                "note": "measured",
            },
            {
                "case_id": "official_13",
                "mechanism_id": "runtime_schedule",
                "mechanism_label": "Runtime schedule",
                "status": "not_applicable",
                "variant_kind": "not_applicable",
                "completed_at": "2026-09-01T00:00:01+00:00",
                "note": "already eager",
            },
        ]
    }
    path = write_component_ablation_csv(summary, tmp_path / "ablation.csv")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert rows[0]["ablation_slowdown"] == "1.5"
    assert float(rows[0]["retained_performance_fraction"]) == pytest.approx(2.0 / 3.0)
    assert rows[0]["timed_samples_per_config"] == "25"
    assert rows[0]["correctness_passed"] == "true"
    assert rows[1]["ablation_slowdown"] == ""
    assert rows[1]["retained_performance_fraction"] == ""
    assert rows[1]["correctness_passed"] == "false"
