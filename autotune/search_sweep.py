"""Thin bridge from generated configurations to direct GPU measurements."""

from __future__ import annotations

import math
from collections.abc import Callable
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
from solution.config import ConfigSpec, portable_config, portable_streamed_config
from solution.plan import ExecutionContext
from solution.plan_builder import HardwareCapabilities, PlanBuilder

from .evaluation import (
    RESIDENT_PROTOCOLS,
    STREAMED_PROTOCOLS,
    ConstraintVector,
    EvaluationScope,
    Fidelity,
    PairedMeasurement,
    TrialMeasurement,
    classify_infeasible_exception,
    execution_signatures_match,
    normalized_accuracy_constraint,
)
from .promotion import promotion_should_stop
from .search_engine import SearchBudget, SearchEngine, SearchRequest, SearchResult
from .search_space import ProgramSearchSpace
from .study_storage import SearchStorage

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
    ) -> None:
        self.shape = shape
        self.variant = variant
        self.device = device
        self.scope = (
            EvaluationScope.STREAMED if shape.streamed else EvaluationScope.RESIDENT
        )

    def _to_measurement(
        self,
        result: BenchmarkResult,
        fidelity: Fidelity,
    ) -> TrialMeasurement:
        constraints = ConstraintVector(
            accuracy=normalized_accuracy_constraint(result.max_tolerance_ratio),
            execution_path=(
                0.0
                if execution_signatures_match(
                    result.expected_execution_signature,
                    result.actual_execution_signature,
                )
                else 1.0
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
                stop_when=promotion_should_stop,
            )
        except Exception as exc:
            if classify_infeasible_exception(exc) is None:
                raise
            # Preserve feasibility without inventing paired timing evidence.
            incumbent_result = self.evaluate(incumbent, Fidelity.FORMAL)
            challenger_result = self.evaluate(challenger, Fidelity.FORMAL)
            return PairedMeasurement(
                incumbent=incumbent_result,
                challenger=challenger_result,
                paired_ratios=(),
            )
        incumbent_result = self._to_measurement(paired.incumbent, Fidelity.FORMAL)
        challenger_result = self._to_measurement(paired.challenger, Fidelity.FORMAL)
        return PairedMeasurement(
            incumbent=incumbent_result,
            challenger=challenger_result,
            paired_ratios=paired.paired_ratios,
        )


@dataclass(frozen=True, slots=True)
class SearchSweepRequest:
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
class SearchSweepResult:
    shape_results: tuple[ShapeSearchResult, ...]

    @property
    def exit_code(self) -> int:
        if any(
            item.search_result.stop_reason == "interrupted"
            for item in self.shape_results
        ):
            return 130
        # Budget-limited screening is a normal, resumable search outcome. Its
        # observations are persisted and the optimization loop should continue
        # from them instead of treating a missing Formal selection as failure.
        return 0


ShapeObserver = Callable[[ShapeSearchResult], None]


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


def _transferable_formal_config(result: SearchResult) -> ConfigSpec | None:
    """Share only the formally approved deployment with later shapes."""

    return result.selected_config if result.deployment_approved else None


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


class SearchSweep:
    """Run one bounded search pass across an explicit group of shapes."""

    def __init__(self, observer: ShapeObserver | None = None) -> None:
        self.observer = observer

    def run(self, request: SearchSweepRequest) -> SearchSweepResult:
        device = torch.device(request.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("program search requires a CUDA device")
        hardware_key = EnvironmentFingerprint.detect(
            device,
            project_root=request.project_root,
        )
        capabilities = HardwareCapabilities.detect(device)
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
            if incumbent is None:
                incumbent = (
                    portable_streamed_config() if shape.streamed else portable_config()
                )
            scope = (
                EvaluationScope.STREAMED if shape.streamed else EvaluationScope.RESIDENT
            )
            evaluator = BenchmarkEvaluator(
                shape=shape,
                variant=request.variant,
                device=device,
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
                    # Shape 14's 100k-token Formal run dominates wall time.
                    # Close its best Screen challenger instead of replaying the
                    # resident Top-8 promotion breadth.
                    enhanced_top_k=(
                        1 if scope is EvaluationScope.STREAMED else 8
                    ),
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
                search_result.deployment_approved
                and selected is not None
                and selected != incumbent
            )
            if updated:
                publish_deployed_config(
                    hardware=hardware_key,
                    shape=shape_key,
                    config=selected,
                )
            shape_result = ShapeSearchResult(case_id, search_result, updated)
            results.append(shape_result)
            if self.observer is not None:
                self.observer(shape_result)
            if search_result.stop_reason == "interrupted":
                break
            transferable = _transferable_formal_config(search_result)
            if transferable is not None:
                warm_start_candidates.append(
                    _WarmStartCandidate(
                        shape=shape_key,
                        config=transferable,
                        source_priority=0,
                        source_order=warm_start_order,
                    )
                )
                warm_start_order += 1

        return SearchSweepResult(tuple(results))


__all__ = [
    "MAX_CROSS_SHAPE_WARM_STARTS",
    "BenchmarkEvaluator",
    "SearchSweep",
    "SearchSweepRequest",
    "SearchSweepResult",
    "ShapeObserver",
    "ShapeSearchResult",
    "execution_context",
]
