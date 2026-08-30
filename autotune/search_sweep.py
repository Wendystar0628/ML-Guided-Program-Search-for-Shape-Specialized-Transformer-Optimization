"""Thin bridge from generated configurations to direct GPU measurements."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from benchmarking.device_queue import run_in_fresh_process
from benchmarking.measure import BenchmarkResult, measure_config, measure_paired_configs
from benchmarking.protocols import (
    MeasurementProtocol,
    RunVariant,
    TransformerShape,
    load_shape,
    load_shapes,
)
from deployment.environment import ImplementationScope, stable_digest
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
from .evidence_identity import evidence_identity
from .meta_warmstart import (
    WarmStartCandidate,
    best_screen_candidates,
    load_study_summaries,
    select_meta_warm_starts,
)
from .promotion import promotion_should_stop
from .search_engine import SearchBudget, SearchEngine, SearchRequest, SearchResult
from .study_storage import SearchStorage, scoped_search_root


def _measurement_environment_identity(
    hardware: EnvironmentFingerprint,
    variant: RunVariant,
) -> str:
    """Identify measurement conditions without rounding variant values."""

    return stable_digest(
        {
            "hardware_runtime": hardware.measurement_identity,
            "variant": variant.to_dict(),
        }
    )


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


def _protocol(
    scope: EvaluationScope,
    fidelity: Fidelity,
    case_id: str,
) -> MeasurementProtocol:
    source = (
        STREAMED_PROTOCOLS if scope is EvaluationScope.STREAMED else RESIDENT_PROTOCOLS
    )[fidelity]
    accuracy_trials = source.accuracy_trials
    warmup = source.warmup
    if scope is EvaluationScope.RESIDENT and case_id == "official_06":
        accuracy_trials = 1
        warmup = 1 if fidelity is Fidelity.SCREEN else 2
    return MeasurementProtocol(
        accuracy_trials=accuracy_trials,
        warmup=warmup,
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
                _protocol(self.scope, fidelity, self.shape.case_id),
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
                _protocol(self.scope, Fidelity.FORMAL, self.shape.case_id),
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
    scope: ImplementationScope = ImplementationScope.RESIDENT
    device: str = "cuda:0"
    storage_root: Path | None = None
    budget_seconds: float = 900.0
    max_trials: int | None = None
    seed: int = 1234
    variant: RunVariant = field(default_factory=RunVariant)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", ImplementationScope(self.scope))


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


def _implementation_scope(shape: TransformerShape) -> ImplementationScope:
    return (
        ImplementationScope.SHAPE14 if shape.streamed else ImplementationScope.RESIDENT
    )


class SearchSweep:
    """Run one bounded search pass across an explicit group of shapes."""

    def __init__(
        self,
        observer: ShapeObserver | None = None,
        *,
        isolate_shapes: bool = True,
    ) -> None:
        self.observer = observer
        self.isolate_shapes = isolate_shapes

    def run(self, request: SearchSweepRequest) -> SearchSweepResult:
        if self.isolate_shapes:
            results: list[ShapeSearchResult] = []
            for case_id in request.case_ids:
                shape_result = run_in_fresh_process(
                    _run_one_shape_worker,
                    replace(request, case_ids=(case_id,)),
                )
                results.append(shape_result)
                if self.observer is not None:
                    self.observer(shape_result)
                if shape_result.search_result.stop_reason == "interrupted":
                    break
            return SearchSweepResult(tuple(results))
        return self._run_in_process(request)

    def _run_in_process(self, request: SearchSweepRequest) -> SearchSweepResult:
        device = torch.device(request.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("program search requires a CUDA device")
        requested_shapes = {
            case_id: load_shape(request.project_root, case_id)
            for case_id in request.case_ids
        }
        mismatched = tuple(
            case_id
            for case_id, shape in requested_shapes.items()
            if _implementation_scope(shape) is not request.scope
        )
        if mismatched:
            joined = ", ".join(mismatched)
            raise ValueError(
                f"search scope {request.scope.value!r} does not match: {joined}"
            )
        hardware_key = EnvironmentFingerprint.detect(
            device,
            project_root=request.project_root,
            scope=request.scope,
        )
        capabilities = HardwareCapabilities.detect(device)
        storage = SearchStorage(
            scoped_search_root(
                request.project_root,
                request.scope.value,
                request.storage_root,
            )
        )
        evidence = evidence_identity(request.project_root, scope=request.scope)
        plan_builder = PlanBuilder()
        environment = _measurement_environment_identity(
            hardware_key,
            request.variant,
        )
        enhanced_identity = stable_digest(
            {
                "environment": environment,
                "evidence": evidence.enhanced,
            }
        )
        known_shapes = tuple(
            shape
            for shape in load_shapes(request.project_root)
            if _implementation_scope(shape) is request.scope
        )
        shape_by_fingerprint = {
            _shape_key(shape, request.variant): shape for shape in known_shapes
        }
        study_summaries = load_study_summaries(storage)
        results: list[ShapeSearchResult] = []
        warm_start_candidates: list[WarmStartCandidate] = []
        warm_start_order = 0
        for deployed_shape, config in iter_deployed_configs(hardware=hardware_key):
            source_shape = shape_by_fingerprint.get(deployed_shape)
            if source_shape is None:
                continue
            warm_start_candidates.append(
                WarmStartCandidate(
                    shape=source_shape,
                    variant=request.variant,
                    config=config,
                    evidence_priority=1,
                    source_order=warm_start_order,
                )
            )
            warm_start_order += 1
        for source_shape in known_shapes:
            candidates = best_screen_candidates(
                study_summaries,
                shape=source_shape,
                variant=request.variant,
                environment=environment,
                search_identity=evidence.search,
                source_order=warm_start_order,
            )
            warm_start_candidates.extend(candidates)
            warm_start_order += len(candidates)

        for case_id in request.case_ids:
            shape = requested_shapes[case_id]
            shape_key = _shape_key(shape, request.variant)
            incumbent = resolve_deployed_config(
                hardware=hardware_key,
                shape=shape_key,
            )
            if incumbent is None:
                incumbent = (
                    portable_streamed_config() if shape.streamed else portable_config()
                )
            evaluation_scope = (
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
                scope=evaluation_scope,
                environment=environment,
                search_identity=evidence.search,
                enhanced_identity=enhanced_identity,
                promotion_identity=evidence.promotion,
                budget=SearchBudget(
                    max_seconds=request.budget_seconds,
                    max_trials=request.max_trials,
                    # Shape 14's 100k-token Formal run dominates wall time.
                    # Close its best Screen challenger instead of replaying the
                    # resident Top-8 promotion breadth.
                    enhanced_top_k=(
                        1 if evaluation_scope is EvaluationScope.STREAMED else 8
                    ),
                ),
                seed=request.seed,
                incumbent=incumbent,
            )
            search_plan = engine.plan(search_request)
            warm_starts = select_meta_warm_starts(
                candidates=warm_start_candidates,
                target=shape,
                variant=request.variant,
                reference_shapes=known_shapes,
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
                    WarmStartCandidate(
                        shape=shape,
                        variant=request.variant,
                        config=transferable,
                        evidence_priority=0,
                        source_order=warm_start_order,
                    )
                )
                warm_start_order += 1

        return SearchSweepResult(tuple(results))


def _run_one_shape_worker(request: SearchSweepRequest) -> ShapeSearchResult:
    result = SearchSweep(isolate_shapes=False).run(request)
    if len(result.shape_results) != 1:
        raise RuntimeError("isolated search worker did not return exactly one shape")
    return result.shape_results[0]


__all__ = [
    "BenchmarkEvaluator",
    "SearchSweep",
    "SearchSweepRequest",
    "SearchSweepResult",
    "ShapeObserver",
    "ShapeSearchResult",
    "execution_context",
]
