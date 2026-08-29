"""Thin bridge from generated configurations to direct GPU measurements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from optimizer.engine import SearchBudget, SearchEngine, SearchRequest, SearchResult
from optimizer.evaluation import (
    RESIDENT_PROTOCOLS,
    STREAMED_PROTOCOLS,
    ConstraintVector,
    EvaluationScope,
    Fidelity,
    PairedMeasurement,
    TrialMeasurement,
    execution_signatures_match,
    memory_constraint,
    normalized_accuracy_constraint,
)
from optimizer.storage import SearchStorage
from runner.benchmark import measure_config
from runner.contracts import (
    MeasurementProtocol,
    RunVariant,
    TransformerShape,
    load_shape,
)
from solution.config import ConfigSpec
from solution.config_compiler import ConfigCompiler, HardwareCapabilities
from solution.deployed_configs import (
    HardwareFingerprint,
    ShapeFingerprint,
    publish_deployed_config,
    resolve_deployed_config,
)
from solution.execution_plan import ExecutionContext

MIN_PROMOTION_SPEEDUP = 1.02


def compilation_context(
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


class RunnerSearchEvaluator:
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
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            return TrialMeasurement.infeasible(
                config_id=config.config_id,
                fidelity=fidelity,
                scope=self.scope,
                penalty_ms=1_000_000_000.0,
                constraints=ConstraintVector(runtime=1.0),
                failure_kind=type(exc).__name__,
                metrics={"message": str(exc)[:500]},
            )

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
            config_id=config.config_id,
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

    def compare(
        self,
        challenger: ConfigSpec,
        incumbent: ConfigSpec,
    ) -> PairedMeasurement:
        incumbent_result = self.evaluate(incumbent, Fidelity.FORMAL)
        challenger_result = self.evaluate(challenger, Fidelity.FORMAL)
        speedup = incumbent_result.objective_ms / challenger_result.objective_ms
        return PairedMeasurement(
            incumbent=incumbent_result,
            challenger=challenger_result,
            speedup=speedup,
            exceeds_noise_margin=speedup >= MIN_PROMOTION_SPEEDUP,
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
    variant: RunVariant = RunVariant()


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
        hardware_key = HardwareFingerprint.detect(device)
        capabilities = HardwareCapabilities.detect(device)
        properties = torch.cuda.get_device_properties(device)
        memory_budget = int(properties.total_memory * 0.95)
        storage = SearchStorage(
            request.storage_root
            or request.project_root / "results" / "intermediate" / "search"
        )
        compiler = ConfigCompiler()
        results: list[ShapeSearchResult] = []

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
            evaluator = RunnerSearchEvaluator(
                shape=shape,
                variant=request.variant,
                device=device,
                memory_budget_bytes=memory_budget,
            )
            engine = SearchEngine(
                storage=storage,
                evaluator=evaluator,
                compiler=compiler,
            )
            search_result = engine.run(
                SearchRequest(
                    case_id=case_id,
                    compilation_context=compilation_context(
                        shape,
                        request.variant,
                        device,
                    ),
                    hardware=capabilities,
                    scope=scope,
                    environment=(
                        f"{hardware_key.device_name}-{hardware_key.compute_capability}-"
                        f"{request.variant.dtype}-padding{request.variant.padding_ratio:g}-"
                        f"scale{request.variant.input_scale:g}"
                    ),
                    budget=SearchBudget(
                        max_seconds=request.budget_seconds,
                        max_trials=request.max_trials,
                    ),
                    seed=request.seed,
                    incumbent=incumbent,
                )
            )
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

        return SearchServiceResult(tuple(results))


__all__ = [
    "MIN_PROMOTION_SPEEDUP",
    "RunnerSearchEvaluator",
    "SearchService",
    "SearchServiceRequest",
    "SearchServiceResult",
    "ShapeSearchResult",
    "compilation_context",
]
