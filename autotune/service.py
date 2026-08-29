"""Thin bridge from generated configurations to direct GPU measurements."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from benchmarking.measure import BenchmarkResult, measure_config, measure_paired_configs
from benchmarking.protocols import (
    MeasurementProtocol,
    RunVariant,
    TransformerShape,
    load_shape,
)
from deployment.registry import (
    EnvironmentFingerprint,
    ShapeFingerprint,
    iter_deployed_configs,
    publish_deployed_config,
    resolve_deployed_config,
)
from solution.config import ConfigSpec
from solution.plan import ExecutionContext
from solution.plan_builder import HardwareCapabilities, PlanBuilder

from .engine import SearchBudget, SearchEngine, SearchRequest, SearchResult
from .evaluator import (
    RESIDENT_PROTOCOLS,
    STREAMED_PROTOCOLS,
    ConstraintVector,
    EvaluationScope,
    Fidelity,
    PairedMeasurement,
    TrialMeasurement,
    classify_infeasible_exception,
    execution_signatures_match,
    memory_constraint,
    normalized_accuracy_constraint,
)
from .space import ProgramSearchSpace
from .storage import SearchStorage

MIN_PROMOTION_SPEEDUP = 1.02
MAX_CROSS_SHAPE_WARM_STARTS = 4


def execution_context(
    shape: TransformerShape,
    variant: RunVariant,
    device: torch.device,
) -> ExecutionContext:
    return ExecutionContext(
        batch_size=shape.batch_size,
        seq_len=shape.seq_len,
        d_model=shape.d_model,
        num_heads=shape.num_heads,
        causal=shape.causal,
        device=device,
        dtype={
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[variant.dtype],
        training=False,
        grad_enabled=False,
        input_contiguous=True,
        has_valid_token_mask=variant.padding_ratio > 0.0,
        mask_compatible=True,
        ffn_dim=shape.ffn_dim,
        num_layers=shape.num_layers,
    )


def _protocol(scope: EvaluationScope, fidelity: Fidelity) -> MeasurementProtocol:
    source = (
        STREAMED_PROTOCOLS if scope is EvaluationScope.STREAMED else RESIDENT_PROTOCOLS
    )[fidelity]
    return MeasurementProtocol(
        accuracy_trials=source.accuracy_trials,
        warmup=source.warmup,
        repeats=source.repeats,
        rounds=source.rounds,
        full_logical_batch=source.full_logical_batch,
    )


class BenchmarkEvaluator:
    """Turn direct measurements into constrained Optuna observations."""

    def __init__(
        self,
        *,
        shape: TransformerShape,
        variant: RunVariant,
        device: torch.device,
        memory_budget_bytes: int | None,
    ) -> None:
        self.shape = shape
        self.variant = variant
        self.device = device
        self.scope = (
            EvaluationScope.STREAMED if shape.streamed else EvaluationScope.RESIDENT
        )
        self.memory_budget_bytes = memory_budget_bytes

    def _to_measurement(
        self,
        result: BenchmarkResult,
        fidelity: Fidelity,
    ) -> TrialMeasurement:
        safety_margin = 1.0 if fidelity is Fidelity.FORMAL else 0.9
        constraints = ConstraintVector(
            accuracy=normalized_accuracy_constraint(
                result.max_tolerance_ratio,
                safety_margin=safety_margin,
            ),
            execution_path=(
                0.0
                if execution_signatures_match(
                    result.expected_execution_signature,
                    result.actual_execution_signature,
                )
                else 1.0
            ),
            memory=memory_constraint(
                result.peak_memory_bytes,
                self.memory_budget_bytes,
            ),
            runtime=0.0,
        )
        failure = None if constraints.feasible else "constraint_violation"
        return TrialMeasurement(
            config_id=result.config.config_id,
            fidelity=fidelity,
            scope=self.scope,
            objective_ms=result.optimized.median_ms,
            median_ms=result.optimized.median_ms,
            p90_ms=result.optimized.p90_ms,
            peak_memory_bytes=result.peak_memory_bytes,
            max_tolerance_ratio=result.max_tolerance_ratio,
            expected_execution_signature=result.expected_execution_signature,
            actual_execution_signature=result.actual_execution_signature,
            constraints=constraints,
            failure_kind=failure,
        )

    def _known_infeasible(
        self,
        config: ConfigSpec,
        fidelity: Fidelity,
        exc: Exception,
    ) -> TrialMeasurement | None:
        failure_kind = classify_infeasible_exception(exc)
        if failure_kind is None:
            return None
        return TrialMeasurement.infeasible(
            config_id=config.config_id,
            fidelity=fidelity,
            scope=self.scope,
            penalty_ms=1_000_000_000.0,
            constraints=ConstraintVector(runtime=1.0),
            failure_kind=failure_kind,
            metrics={"message": str(exc)[:500]},
        )

    def evaluate(self, config: ConfigSpec, fidelity: Fidelity) -> TrialMeasurement:
        try:
            result = measure_config(
                self.shape,
                config,
                self.variant,
                _protocol(self.scope, fidelity),
                self.device,
                include_baseline=False,
            )
        except Exception as exc:
            infeasible = self._known_infeasible(config, fidelity, exc)
            if infeasible is None:
                raise
            return infeasible
        return self._to_measurement(result, fidelity)

    def compare(
        self,
        challenger: ConfigSpec,
        incumbent: ConfigSpec,
    ) -> PairedMeasurement:
        try:
            paired = measure_paired_configs(
                self.shape,
                challenger,
                incumbent,
                self.variant,
                _protocol(self.scope, Fidelity.FORMAL),
                self.device,
            )
        except Exception as exc:
            if classify_infeasible_exception(exc) is None:
                raise
            # Locate the infeasible program. This path is not used for a
            # performance promotion; it only preserves the constraint result.
            incumbent_result = self.evaluate(incumbent, Fidelity.FORMAL)
            challenger_result = self.evaluate(challenger, Fidelity.FORMAL)
            ratio = incumbent_result.objective_ms / challenger_result.objective_ms
            return PairedMeasurement(
                incumbent=incumbent_result,
                challenger=challenger_result,
                paired_ratios=(ratio,),
                exceeds_noise_margin=False,
            )
        incumbent_result = self._to_measurement(paired.incumbent, Fidelity.FORMAL)
        challenger_result = self._to_measurement(paired.challenger, Fidelity.FORMAL)
        return PairedMeasurement(
            incumbent=incumbent_result,
            challenger=challenger_result,
            paired_ratios=paired.paired_ratios,
            exceeds_noise_margin=paired.median_speedup >= MIN_PROMOTION_SPEEDUP,
        )


@dataclass(frozen=True, slots=True)
class SearchServiceRequest:
    project_root: Path
    case_ids: tuple[str, ...]
    device: str = "cuda:0"
    storage_root: Path | None = None
    budget_seconds: float = 900.0
    max_trials: int | None = None
    seed: int = 1234
    variant: RunVariant = field(default_factory=RunVariant)


@dataclass(frozen=True, slots=True)
class ShapeSearchResult:
    case_id: str
    search_result: SearchResult
    deployment_updated: bool

    @property
    def selected_config(self) -> ConfigSpec | None:
        return self.search_result.selected_config


@dataclass(frozen=True, slots=True)
class SearchServiceResult:
    shape_results: tuple[ShapeSearchResult, ...]

    @property
    def exit_code(self) -> int:
        if any(
            item.search_result.stop_reason == "interrupted"
            for item in self.shape_results
        ):
            return 130
        return (
            0
            if all(item.selected_config is not None for item in self.shape_results)
            else 1
        )


@dataclass(frozen=True, slots=True)
class _WarmStartCandidate:
    shape: ShapeFingerprint
    config: ConfigSpec
    source_priority: int
    source_order: int


def _same_shape_family(
    left: ShapeFingerprint,
    right: ShapeFingerprint,
) -> bool:
    """Match stable program dimensions while allowing batch/head transfer."""

    return (
        left.qkv_dim == right.qkv_dim
        and left.seq_len == right.seq_len
        and left.layers == right.layers
        and left.causal == right.causal
        and left.ffn_dim == right.ffn_dim
        and left.dtype == right.dtype
        and left.padding_ratio == right.padding_ratio
        and left.input_scale == right.input_scale
    )


def _ratio_distance(left: int, right: int) -> float:
    return abs(math.log2(left / right))


def _warm_start_sort_key(
    candidate: _WarmStartCandidate,
    target: ShapeFingerprint,
) -> tuple[bool, float, bool, float, int, int]:
    source = candidate.shape
    return (
        source.batch_size != target.batch_size,
        _ratio_distance(source.batch_size, target.batch_size),
        source.heads != target.heads,
        _ratio_distance(source.heads, target.heads),
        candidate.source_priority,
        candidate.source_order,
    )


def _compatible_family_warm_starts(
    *,
    candidates: list[_WarmStartCandidate],
    target: ShapeFingerprint,
    incumbent: ConfigSpec | None,
    search_space: ProgramSearchSpace,
    limit: int = MAX_CROSS_SHAPE_WARM_STARTS,
) -> tuple[ConfigSpec, ...]:
    """Select bounded, unique family configs accepted by this exact plan."""

    if limit <= 0:
        return ()
    compatible: list[ConfigSpec] = []
    seen = {incumbent.config_id} if incumbent is not None else set()
    ordered = sorted(
        (
            candidate
            for candidate in candidates
            if _same_shape_family(candidate.shape, target)
        ),
        key=lambda candidate: _warm_start_sort_key(candidate, target),
    )
    for candidate in ordered:
        config = candidate.config
        if config.config_id in seen:
            continue
        # ``accepted`` invokes PlanBuilder for this exact target context. The
        # final plan includes accepted warm starts as required branches.
        if not search_space.accepted(config):
            continue
        seen.add(config.config_id)
        compatible.append(config)
        if len(compatible) >= limit:
            break
    return tuple(compatible)


def _transferable_formal_configs(result: SearchResult) -> tuple[ConfigSpec, ...]:
    """Keep only formally feasible results, with the selected winner first."""

    values: list[ConfigSpec] = []
    selected = result.selected_config if result.has_deployable_selection else None
    if selected is not None:
        values.append(selected)
    ranked = sorted(
        (
            (config, measurement)
            for config, measurement in zip(
                result.formal_configs,
                result.formal_measurements,
                strict=True,
            )
            if measurement.fidelity is Fidelity.FORMAL and measurement.feasible
        ),
        key=lambda item: item[1].objective_ms,
    )
    values.extend(config for config, _ in ranked)
    unique: list[ConfigSpec] = []
    seen: set[str] = set()
    for config in values:
        if config.config_id in seen:
            continue
        seen.add(config.config_id)
        unique.append(config)
    return tuple(unique)


def _shape_key(shape: TransformerShape, variant: RunVariant) -> ShapeFingerprint:
    return ShapeFingerprint(
        batch_size=shape.batch_size,
        qkv_dim=shape.d_model,
        heads=shape.num_heads,
        seq_len=shape.seq_len,
        layers=shape.num_layers,
        causal=shape.causal,
        ffn_dim=shape.ffn_dim,
        dtype=variant.dtype,
        padding_ratio=variant.padding_ratio,
        input_scale=variant.input_scale,
    )


class SearchService:
    def run(self, request: SearchServiceRequest) -> SearchServiceResult:
        device = torch.device(request.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("program search requires a CUDA device")
        hardware_key = EnvironmentFingerprint.detect(
            device,
            project_root=request.project_root,
        )
        capabilities = HardwareCapabilities.detect(device)
        properties = torch.cuda.get_device_properties(device)
        memory_budget = int(properties.total_memory * 0.95)
        storage = SearchStorage(
            request.storage_root or request.project_root / "search_state"
        )
        plan_builder = PlanBuilder()
        results: list[ShapeSearchResult] = []
        warm_start_candidates: list[_WarmStartCandidate] = []
        warm_start_order = 0
        for deployed_shape, config in iter_deployed_configs(hardware=hardware_key):
            warm_start_candidates.append(
                _WarmStartCandidate(
                    shape=deployed_shape,
                    config=config,
                    source_priority=2,
                    source_order=warm_start_order,
                )
            )
            warm_start_order += 1

        for case_id in request.case_ids:
            shape = load_shape(request.project_root, case_id)
            shape_key = _shape_key(shape, request.variant)
            incumbent = resolve_deployed_config(
                hardware=hardware_key,
                shape=shape_key,
            )
            scope = (
                EvaluationScope.STREAMED if shape.streamed else EvaluationScope.RESIDENT
            )
            evaluator = BenchmarkEvaluator(
                shape=shape,
                variant=request.variant,
                device=device,
                memory_budget_bytes=memory_budget,
            )
            engine = SearchEngine(
                storage=storage,
                evaluator=evaluator,
                plan_builder=plan_builder,
            )
            search_request = SearchRequest(
                case_id=case_id,
                execution_context=execution_context(
                    shape,
                    request.variant,
                    device,
                ),
                hardware=capabilities,
                scope=scope,
                environment=(
                    f"{hardware_key.identity}-{request.variant.dtype}-"
                    f"padding{request.variant.padding_ratio:g}-"
                    f"scale{request.variant.input_scale:g}"
                ),
                budget=SearchBudget(
                    max_seconds=request.budget_seconds,
                    max_trials=request.max_trials,
                ),
                seed=request.seed,
                incumbent=incumbent,
            )
            search_plan = engine.plan(search_request)
            warm_starts = _compatible_family_warm_starts(
                candidates=warm_start_candidates,
                target=shape_key,
                incumbent=incumbent,
                search_space=search_plan.search_space,
            )
            search_result = engine.run(replace(search_request, warm_starts=warm_starts))
            selected = search_result.selected_config
            updated = (
                search_result.has_deployable_selection
                and selected is not None
                and selected != incumbent
            )
            if updated:
                publish_deployed_config(
                    hardware=hardware_key,
                    shape=shape_key,
                    config=selected,
                )
            results.append(ShapeSearchResult(case_id, search_result, updated))
            for config in _transferable_formal_configs(search_result):
                warm_start_candidates.append(
                    _WarmStartCandidate(
                        shape=shape_key,
                        config=config,
                        source_priority=(
                            0
                            if selected is not None
                            and config.config_id == selected.config_id
                            else 1
                        ),
                        source_order=warm_start_order,
                    )
                )
                warm_start_order += 1

        return SearchServiceResult(tuple(results))


__all__ = [
    "MAX_CROSS_SHAPE_WARM_STARTS",
    "MIN_PROMOTION_SPEEDUP",
    "BenchmarkEvaluator",
    "SearchService",
    "SearchServiceRequest",
    "SearchServiceResult",
    "ShapeSearchResult",
    "execution_context",
]
