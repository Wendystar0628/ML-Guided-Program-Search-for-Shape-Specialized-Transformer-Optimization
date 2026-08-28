"""Execute benchmark and profile requests using the official primitives."""

from __future__ import annotations

import importlib
import math
import statistics
import sys
import tempfile
import types
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import nn

from official import torch_transformer_benchmark as official
from project_identity import solution_implementation_hash
from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    RunVariant,
    TransformerShape,
)
from runner.probe import collect_environment
from runner.resource_guard import ResourceGuardError, ensure_local_benchmark_allowed
from runner.result_contracts import WorkerRequest


def load_solution_module(project_root: Path) -> ModuleType:
    """Load the current Solution without depending on the caller's working directory."""

    solution_root = (project_root / "solution").resolve()
    source_path = solution_root / "transformer.py"
    if not source_path.is_file():
        raise ContractError(f"Solution entry file is missing: {source_path}")

    package_name = f"_benchmark_solution_{uuid.uuid4().hex}"
    bytecode_cache = tempfile.TemporaryDirectory(prefix="solution-bytecode-")
    previous_pycache_prefix = sys.pycache_prefix
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.pycache_prefix = bytecode_cache.name
    sys.dont_write_bytecode = True
    package = types.ModuleType(package_name)
    package.__path__ = [str(solution_root)]
    package.__bytecode_cache__ = bytecode_cache
    sys.modules[package_name] = package
    module_name = f"{package_name}.transformer"
    try:
        try:
            module = importlib.import_module(module_name)
        finally:
            sys.pycache_prefix = previous_pycache_prefix
            sys.dont_write_bytecode = previous_dont_write_bytecode
    except BaseException:
        for loaded_name in tuple(sys.modules):
            if loaded_name == package_name or loaded_name.startswith(
                f"{package_name}."
            ):
                sys.modules.pop(loaded_name, None)
        bytecode_cache.cleanup()
        raise

    solution_class = getattr(module, "UserOptimizedTransformer", None)
    if not isinstance(solution_class, type) or not issubclass(
        solution_class, nn.Module
    ):
        for loaded_name in tuple(sys.modules):
            if loaded_name == package_name or loaded_name.startswith(
                f"{package_name}."
            ):
                sys.modules.pop(loaded_name, None)
        bytecode_cache.cleanup()
        raise ContractError(
            "Solution must export an nn.Module named UserOptimizedTransformer"
        )
    return module


def _assert_unchanged(name: str, value: torch.Tensor, snapshot: torch.Tensor) -> None:
    if (
        value.shape != snapshot.shape
        or value.dtype != snapshot.dtype
        or value.device != snapshot.device
        or not torch.equal(value, snapshot)
    ):
        raise ContractError(f"Solution modified {name} in place")


def _validate_output(
    output: Any,
    reference: torch.Tensor,
    inputs: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise ContractError("Solution output must be a Tensor")
    if output.shape != reference.shape:
        raise ContractError(
            f"output shape mismatch: expected={tuple(reference.shape)}, "
            f"actual={tuple(output.shape)}"
        )
    if output.device != inputs.device:
        raise ContractError(
            f"output device mismatch: expected={inputs.device}, actual={output.device}"
        )
    if output.dtype != inputs.dtype:
        raise ContractError(
            f"output dtype mismatch: expected={inputs.dtype}, actual={output.dtype}"
        )
    return output


def run_correctness(
    baseline: nn.Module,
    solution: nn.Module,
    config: official.TransformerConfig,
    variant: RunVariant,
    protocol: MeasurementProtocol,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    trial_count = 0
    all_passed = True
    failed_elements = 0
    failed_elements_known = True
    max_abs_errors: list[float] = []
    max_relative_errors: list[float] = []
    diagnostic: str | None = None
    with torch.inference_mode():
        for trial_index in range(protocol.accuracy_trials):
            seed = protocol.seed + trial_index
            trial_count += 1
            reference: torch.Tensor | None = None
            inputs, valid_mask = official.generate_random_case(
                config=config,
                device=device,
                dtype=dtype,
                seed=seed,
                padding_ratio=variant.padding_ratio,
                input_scale=variant.input_scale,
            )
            input_snapshot = inputs.clone()
            mask_snapshot = valid_mask.clone()
            try:
                reference = baseline(inputs, valid_mask)
                _assert_unchanged("input", inputs, input_snapshot)
                _assert_unchanged("valid_token_mask", valid_mask, mask_snapshot)
                solution_output = solution(inputs, valid_mask)
                _assert_unchanged("input", inputs, input_snapshot)
                _assert_unchanged("valid_token_mask", valid_mask, mask_snapshot)
                solution_output = _validate_output(solution_output, reference, inputs)
                result = official.compare_outputs(
                    reference,
                    solution_output,
                    rtol=protocol.rtol,
                    atol=protocol.atol,
                )
                all_passed = all_passed and result.passed
                failed_elements += int(result.failed_elements)
                if math.isfinite(result.max_abs_error):
                    max_abs_errors.append(float(result.max_abs_error))
                if math.isfinite(result.max_relative_error):
                    max_relative_errors.append(float(result.max_relative_error))
            except (AssertionError, ContractError, TypeError, ValueError) as exc:
                all_passed = False
                failed_elements_known = False
                if diagnostic is None:
                    diagnostic = f"seed {seed}: {type(exc).__name__}: {exc}"

    passed = trial_count == protocol.accuracy_trials and all_passed
    summary: dict[str, Any] = {
        "passed": passed,
        "trial_count": trial_count,
    }
    if failed_elements_known:
        summary["failed_elements"] = failed_elements
    if max_abs_errors:
        summary["max_abs_error"] = max(max_abs_errors)
    if max_relative_errors:
        summary["max_relative_error"] = max(max_relative_errors)
    if not passed and diagnostic is not None:
        summary["diagnostic"] = diagnostic[-500:]
    return summary


def _measurement_stats(
    samples: list[float],
    *,
    repeats: int,
    rounds: int,
) -> dict[str, Any]:
    expected_count = repeats * rounds
    if len(samples) != expected_count:
        raise ContractError(
            f"expected {expected_count} timing samples, received {len(samples)}"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
        for value in samples
    ):
        raise ContractError("timing samples must be finite positive numbers")
    normalized = [float(value) for value in samples]
    median = statistics.median(normalized)
    if not math.isfinite(median) or median <= 0:
        raise ContractError("timing median must be a finite positive number")
    p90 = official.percentile(normalized, 0.9)
    return {
        "sample_count": expected_count,
        "median_ms": median,
        "p90_ms": p90,
    }


def run_performance(
    baseline: nn.Module,
    solution: nn.Module,
    config: official.TransformerConfig,
    variant: RunVariant,
    protocol: MeasurementProtocol,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    inputs, valid_mask = official.generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=protocol.seed + 100000,
        padding_ratio=variant.padding_ratio,
        input_scale=variant.input_scale,
    )
    input_snapshot = inputs.clone()
    mask_snapshot = valid_mask.clone()
    official.warmup_model(baseline, inputs, valid_mask, protocol.warmup, device)
    official.warmup_model(solution, inputs, valid_mask, protocol.warmup, device)

    baseline_samples: list[float] = []
    solution_samples: list[float] = []
    for round_index in range(protocol.rounds):
        if round_index % 2 == 0:
            baseline_samples.extend(
                official.benchmark_once(
                    baseline, inputs, valid_mask, protocol.repeats, device
                )
            )
            solution_samples.extend(
                official.benchmark_once(
                    solution, inputs, valid_mask, protocol.repeats, device
                )
            )
        else:
            solution_samples.extend(
                official.benchmark_once(
                    solution, inputs, valid_mask, protocol.repeats, device
                )
            )
            baseline_samples.extend(
                official.benchmark_once(
                    baseline, inputs, valid_mask, protocol.repeats, device
                )
            )

    _assert_unchanged("input", inputs, input_snapshot)
    _assert_unchanged("valid_token_mask", valid_mask, mask_snapshot)
    baseline_stats = _measurement_stats(
        baseline_samples,
        repeats=protocol.repeats,
        rounds=protocol.rounds,
    )
    solution_stats = _measurement_stats(
        solution_samples,
        repeats=protocol.repeats,
        rounds=protocol.rounds,
    )
    speedup = baseline_stats["median_ms"] / solution_stats["median_ms"]
    if not math.isfinite(speedup) or speedup <= 0:
        raise ContractError("speedup must be a finite positive number")
    return {
        "timer": "cuda_event" if device.type == "cuda" else "perf_counter_ns",
        "baseline": baseline_stats,
        "target": solution_stats,
        "speedup": speedup,
    }


def run_baseline_performance(
    baseline: nn.Module,
    config: official.TransformerConfig,
    variant: RunVariant,
    protocol: MeasurementProtocol,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    inputs, valid_mask = official.generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=protocol.seed + 100000,
        padding_ratio=variant.padding_ratio,
        input_scale=variant.input_scale,
    )
    input_snapshot = inputs.clone()
    mask_snapshot = valid_mask.clone()
    official.warmup_model(baseline, inputs, valid_mask, protocol.warmup, device)
    samples: list[float] = []
    for _ in range(protocol.rounds):
        samples.extend(
            official.benchmark_once(
                baseline, inputs, valid_mask, protocol.repeats, device
            )
        )
    _assert_unchanged("input", inputs, input_snapshot)
    _assert_unchanged("valid_token_mask", valid_mask, mask_snapshot)
    stats = _measurement_stats(
        samples,
        repeats=protocol.repeats,
        rounds=protocol.rounds,
    )
    return {
        "timer": "cuda_event" if device.type == "cuda" else "perf_counter_ns",
        "baseline": stats,
    }


def _config_for_shape(shape: TransformerShape) -> official.TransformerConfig:
    config = official.TransformerConfig(
        batch_size=shape.batch_size,
        seq_len=shape.seq_len,
        d_model=shape.d_model,
        num_heads=shape.num_heads,
        ffn_dim=shape.ffn_dim,
        num_layers=shape.num_layers,
        causal=shape.causal,
    )
    config.validate()
    return config


def _configure_runtime(protocol: MeasurementProtocol, device: torch.device) -> None:
    torch.manual_seed(protocol.seed)
    torch.set_float32_matmul_precision(protocol.matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(protocol.seed)
        torch.backends.cuda.matmul.allow_tf32 = protocol.allow_tf32
        torch.backends.cudnn.allow_tf32 = protocol.allow_tf32


def _load_solution_source(project_root: Path) -> tuple[ModuleType, str]:
    source_hash_before_load = solution_implementation_hash(project_root / "solution")
    solution_module = load_solution_module(project_root)
    measured_source_hash = solution_implementation_hash(project_root / "solution")
    if measured_source_hash != source_hash_before_load:
        raise ContractError("Solution source changed while it was being loaded")
    return solution_module, measured_source_hash


def _build_solution(
    solution_module: ModuleType,
    config: official.TransformerConfig,
    solution_policy: str | None,
) -> nn.Module:
    solution = solution_module.UserOptimizedTransformer(config)
    if solution_policy is None:
        return solution
    configure = getattr(solution, "configure_runtime_policy", None)
    if not callable(configure):
        raise ContractError(
            "Solution does not expose configure_runtime_policy() for an explicit "
            "solution_policy request"
        )
    configure(policy=solution_policy)
    return solution


@dataclass(frozen=True)
class PreparedExecution:
    """Models and immutable run context shared by timing and profiling."""

    request: WorkerRequest
    project_root: Path
    shape: TransformerShape
    variant: RunVariant
    protocol: MeasurementProtocol
    device: torch.device
    dtype: torch.dtype
    config: official.TransformerConfig
    environment: dict[str, Any]
    baseline: nn.Module
    target_model: nn.Module
    reporting_solution: nn.Module | None
    solution_source_sha256: str | None
    execution_path: dict[str, Any]


class _PreparationFailure(Exception):
    """Preserve the failed preparation stage and any context already collected."""

    def __init__(
        self,
        stage: str,
        cause: BaseException,
        *,
        environment: dict[str, Any] | None,
        solution_hash: str | None,
        execution_path: dict[str, Any] | None,
    ) -> None:
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause
        self.environment = environment
        self.solution_hash = solution_hash
        self.execution_path = execution_path


def prepare_execution(
    request: WorkerRequest | Mapping[str, Any],
    *,
    expected_run_kind: str | None = None,
) -> PreparedExecution:
    """Build one validated execution context for benchmark or profile work."""

    stage = "request"
    environment: dict[str, Any] | None = None
    measured_source_hash: str | None = None
    execution_path: dict[str, Any] | None = None
    try:
        parsed_request = (
            request
            if isinstance(request, WorkerRequest)
            else WorkerRequest.from_dict(request)
        )
        if parsed_request.run_kind not in {"benchmark", "profile"}:
            raise ContractError(
                "prepare_execution accepts only benchmark or profile requests"
            )
        if (
            expected_run_kind is not None
            and parsed_request.run_kind != expected_run_kind
        ):
            raise ContractError(
                f"expected {expected_run_kind!r} request, received "
                f"{parsed_request.run_kind!r}"
            )

        assert parsed_request.project_root is not None
        assert parsed_request.shape is not None
        assert parsed_request.variant is not None
        assert parsed_request.protocol is not None
        assert parsed_request.target is not None
        project_root = parsed_request.project_root
        shape = parsed_request.shape
        variant = parsed_request.variant
        protocol = parsed_request.protocol
        target = parsed_request.target

        stage = "resource_guard"
        ensure_local_benchmark_allowed(shape)

        stage = "device"
        device = official.resolve_device(parsed_request.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        if device.type == "cpu" and protocol.preset != "smoke":
            raise ContractError("CPU execution is supported only by the smoke preset")
        environment = collect_environment(device)

        stage = "dtype"
        dtype = official.resolve_dtype(variant.dtype)
        _configure_runtime(protocol, device)
        config = _config_for_shape(shape)

        stage = "build_model"
        baseline = official.BaselineTransformer(config)
        solution: nn.Module | None = None
        if target == "solution":
            stage = "load_solution"
            solution_module, measured_source_hash = _load_solution_source(project_root)
            stage = "build_model"
            solution = _build_solution(
                solution_module,
                config,
                parsed_request.solution_policy,
            )
            stage = "copy_weights"
            weight_loader = getattr(solution_module, "copy_model_weights", None)
            if weight_loader is None:
                official.copy_model_weights(baseline, solution, strict=True)
            else:
                weight_loader(baseline, solution, strict=True)

        stage = "move_model"
        baseline = baseline.to(device=device, dtype=dtype).eval()
        if solution is not None:
            solution = solution.to(device=device, dtype=dtype).eval()
        reporting_solution = solution

        stage = "execution_plan"
        execution_path = _describe_execution_path(
            solution if solution is not None else baseline,
            shape=shape,
        )
        target_is_compiled = (
            protocol.compile_solution
            if target == "solution"
            else protocol.compile_baseline
        )
        execution_path["execution_mode"] = (
            "torch_compile" if target_is_compiled else "eager"
        )
        _validate_cuda_graph_composition(execution_path, protocol)
        if parsed_request.run_kind == "profile":
            _validate_profile_execution_path(execution_path)

        stage = "compile"
        baseline = official.maybe_compile(
            baseline,
            protocol.compile_baseline,
            protocol.compile_mode,
        )
        if solution is not None:
            target_model = official.maybe_compile(
                solution,
                protocol.compile_solution,
                protocol.compile_mode,
            )
        else:
            target_model = baseline

        return PreparedExecution(
            request=parsed_request,
            project_root=project_root,
            shape=shape,
            variant=variant,
            protocol=protocol,
            device=device,
            dtype=dtype,
            config=config,
            environment=environment,
            baseline=baseline,
            target_model=target_model,
            reporting_solution=reporting_solution,
            solution_source_sha256=measured_source_hash,
            execution_path=execution_path,
        )
    except BaseException as exc:
        raise _PreparationFailure(
            stage,
            exc,
            environment=environment,
            solution_hash=measured_source_hash,
            execution_path=execution_path,
        ) from exc


def _run_prepared_correctness(
    prepared: PreparedExecution,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Run the shared Solution comparator and refresh eager execution evidence."""

    execution_path = dict(prepared.execution_path)
    if prepared.reporting_solution is None:
        return None, execution_path

    observe_execution = not prepared.protocol.compile_solution and (
        _set_execution_observation(prepared.reporting_solution, True)
    )
    try:
        correctness = run_correctness(
            prepared.baseline,
            prepared.target_model,
            prepared.config,
            prepared.variant,
            prepared.protocol,
            prepared.device,
            prepared.dtype,
        )
    finally:
        if observe_execution:
            _set_execution_observation(prepared.reporting_solution, False)
    if observe_execution:
        execution_path = _describe_execution_path(
            prepared.reporting_solution,
            shape=prepared.shape,
        )
        execution_path["execution_mode"] = prepared.execution_path["execution_mode"]
    return correctness, execution_path


def _assert_solution_source_integrity(
    prepared: PreparedExecution,
    *,
    phase: str,
) -> None:
    if prepared.solution_source_sha256 is None:
        return
    current_hash = solution_implementation_hash(prepared.project_root / "solution")
    if current_hash != prepared.solution_source_sha256:
        raise ContractError(f"Solution source changed before {phase}")


def _invalid_output_response(
    prepared: PreparedExecution,
    correctness: dict[str, Any],
    execution_path: dict[str, Any],
    *,
    result_key: str,
) -> dict[str, Any]:
    response = {
        "outcome": "invalid_output",
        "solution_source_sha256": prepared.solution_source_sha256,
        "environment": prepared.environment,
        "correctness": correctness,
        result_key: None,
        "execution_path": execution_path,
        "failure": {
            "stage": "correctness",
            "type": "CorrectnessError",
            "message": "Solution failed the correctness contract",
            "exit_code": None,
        },
    }
    return response


def _exception_outcome(exc: BaseException, stage: str) -> str:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return "oom"
    if isinstance(exc, KeyboardInterrupt):
        return "cancelled"
    if isinstance(exc, ResourceGuardError) or stage in {
        "resource_guard",
        "device",
        "dtype",
    }:
        return "unsupported"
    if stage in {
        "load_solution",
        "build_model",
        "copy_weights",
        "move_model",
        "compile",
    }:
        return "build_error"
    return "runtime_error"


def _failure_response(
    exc: BaseException,
    stage: str,
    *,
    environment: dict[str, Any] | None,
    correctness: dict[str, Any] | None,
    performance: dict[str, Any] | None,
    solution_hash: str | None,
    execution_path: dict[str, Any] | None,
) -> dict[str, Any]:
    outcome = _exception_outcome(exc, stage)
    return {
        "outcome": outcome,
        "solution_source_sha256": solution_hash,
        "environment": environment,
        "correctness": correctness,
        "performance": performance,
        "execution_path": execution_path,
        "failure": {
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc),
            "exit_code": None,
        },
    }


def execute_benchmark(
    request: WorkerRequest | Mapping[str, Any],
) -> dict[str, Any]:
    stage = "request"
    environment: dict[str, Any] | None = None
    correctness: dict[str, Any] | None = None
    performance: dict[str, Any] | None = None
    measured_source_hash: str | None = None
    execution_path: dict[str, Any] | None = None
    try:
        prepared = prepare_execution(request, expected_run_kind="benchmark")
        environment = prepared.environment
        measured_source_hash = prepared.solution_source_sha256
        execution_path = dict(prepared.execution_path)

        if prepared.reporting_solution is not None:
            stage = "correctness"
            correctness, execution_path = _run_prepared_correctness(prepared)
            assert correctness is not None
            if not correctness["passed"]:
                stage = "source_integrity"
                _assert_solution_source_integrity(
                    prepared,
                    phase="correctness completed",
                )
                return _invalid_output_response(
                    prepared,
                    correctness,
                    execution_path,
                    result_key="performance",
                )

            stage = "timing"
            performance = run_performance(
                prepared.baseline,
                prepared.target_model,
                prepared.config,
                prepared.variant,
                prepared.protocol,
                prepared.device,
                prepared.dtype,
            )
        else:
            correctness = {
                "passed": True,
                "trial_count": 0,
                "failed_elements": 0,
                "max_abs_error": 0.0,
                "skipped": "baseline target has no comparison candidate",
            }
            stage = "timing"
            performance = run_baseline_performance(
                prepared.baseline,
                prepared.config,
                prepared.variant,
                prepared.protocol,
                prepared.device,
                prepared.dtype,
            )

        stage = "source_integrity"
        _assert_solution_source_integrity(
            prepared,
            phase="the benchmark completed",
        )

        return {
            "outcome": "success",
            "solution_source_sha256": measured_source_hash,
            "environment": environment,
            "correctness": correctness,
            "performance": performance,
            "execution_path": execution_path,
            "failure": None,
        }
    except _PreparationFailure as failure:
        return _failure_response(
            failure.cause,
            failure.stage,
            environment=failure.environment,
            correctness=None,
            performance=None,
            solution_hash=failure.solution_hash,
            execution_path=failure.execution_path,
        )
    except BaseException as exc:  # noqa: BLE001 - worker execution boundary.
        return _failure_response(
            exc,
            stage,
            environment=environment,
            correctness=correctness,
            performance=performance,
            solution_hash=measured_source_hash,
            execution_path=execution_path,
        )


def _profile_time(event: Any, device: torch.device) -> float:
    if device.type == "cuda":
        return float(
            getattr(
                event,
                "self_device_time_total",
                getattr(event, "self_cuda_time_total", 0.0),
            )
            or 0.0
        )
    return float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)


def _describe_execution_path(
    model: nn.Module,
    *,
    shape: TransformerShape,
) -> dict[str, Any]:
    description: dict[str, Any]
    describe = getattr(model, "describe_execution_path", None)
    if callable(describe):
        value = describe()
        description = dict(value) if isinstance(value, dict) else {}
    else:
        description = {
            "requested_policy": "official-baseline",
            "selected_policy": "official-baseline",
            "qkv_projection": "separate",
            "attention_backend": "official_explicit",
            "runtime_wrapper": "eager",
            "block_backend": "torch",
            "causal_mask": "per_forward" if shape.causal else "none",
            "valid_token_mask": "direct_key_mask",
            "fallback_reasons": [],
        }
    return description


def _set_execution_observation(model: nn.Module, enabled: bool) -> bool:
    """Toggle optional eager branch observation without requiring the API."""

    setter = getattr(model, "set_execution_observation", None)
    if not callable(setter):
        return False
    setter(enabled)
    return True


def _validate_cuda_graph_composition(
    execution_path: dict[str, Any],
    protocol: MeasurementProtocol,
) -> None:
    """Reject compiling a Solution-owned CUDA Graph route."""

    if execution_path.get("runtime_wrapper") != "cuda_graph":
        return
    if protocol.compile_solution:
        raise ContractError("the graph policy cannot combine with torch.compile")


def _validate_profile_execution_path(execution_path: dict[str, Any]) -> None:
    """Keep operator profiling on an eager path with visible ATen work."""

    if execution_path.get("runtime_wrapper") == "cuda_graph":
        raise ContractError(
            "the graph policy hides per-operator profile work; select an eager "
            "policy for operator profiling"
        )


def _observed_attention_backend(events: list[Any]) -> str | None:
    keys = {str(event.key).lower() for event in events}
    if any("cudnn_attention" in key for key in keys):
        return "cudnn"
    if any("efficient_attention" in key for key in keys):
        return "efficient"
    if any("flash_attention" in key for key in keys):
        return "flash"
    if any("scaled_dot_product_attention_math" in key for key in keys):
        return "math"
    return None


def execute_profile(
    request: WorkerRequest | Mapping[str, Any],
) -> dict[str, Any]:
    stage = "request"
    environment: dict[str, Any] | None = None
    correctness: dict[str, Any] | None = None
    measured_source_hash: str | None = None
    execution_path: dict[str, Any] | None = None
    try:
        prepared = prepare_execution(request, expected_run_kind="profile")
        environment = prepared.environment
        measured_source_hash = prepared.solution_source_sha256
        execution_path = dict(prepared.execution_path)

        if prepared.reporting_solution is not None:
            stage = "correctness"
            correctness, execution_path = _run_prepared_correctness(prepared)
            assert correctness is not None
            if not correctness["passed"]:
                stage = "source_integrity"
                _assert_solution_source_integrity(
                    prepared,
                    phase="correctness completed",
                )
                return _invalid_output_response(
                    prepared,
                    correctness,
                    execution_path,
                    result_key="profile",
                )
        else:
            correctness = None

        stage = "profile"
        inputs, valid_mask = official.generate_random_case(
            config=prepared.config,
            device=prepared.device,
            dtype=prepared.dtype,
            seed=prepared.protocol.seed + 100000,
            padding_ratio=prepared.variant.padding_ratio,
            input_scale=prepared.variant.input_scale,
        )
        input_snapshot = inputs.clone()
        mask_snapshot = valid_mask.clone()
        official.warmup_model(
            prepared.target_model,
            inputs,
            valid_mask,
            prepared.protocol.warmup,
            prepared.device,
        )
        activities = [torch.profiler.ProfilerActivity.CPU]
        if prepared.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        iterations = min(max(prepared.protocol.repeats, 1), 10)
        with (
            torch.profiler.profile(
                activities=activities,
                record_shapes=True,
            ) as profiler,
            torch.inference_mode(),
        ):
            for _ in range(iterations):
                prepared.target_model(inputs, valid_mask)
        if prepared.device.type == "cuda":
            torch.cuda.synchronize(prepared.device)
        _assert_unchanged("input", inputs, input_snapshot)
        _assert_unchanged("valid_token_mask", valid_mask, mask_snapshot)

        events = list(profiler.key_averages(group_by_input_shape=True))
        observed_attention_backend = _observed_attention_backend(events)
        if (
            observed_attention_backend is None
            and execution_path.get("attention_backend") == "official_explicit"
        ):
            observed_attention_backend = "explicit"
        operator_events = [
            event for event in events if str(event.key).startswith("aten::")
        ]
        operator_events.sort(
            key=lambda event: _profile_time(event, prepared.device), reverse=True
        )
        total_operator_self_time_us = math.fsum(
            max(_profile_time(event, prepared.device), 0.0) for event in operator_events
        )
        if (
            not math.isfinite(total_operator_self_time_us)
            or total_operator_self_time_us <= 0
        ):
            raise ContractError("profiler did not record positive ATen self time")
        operator_hotspots: list[dict[str, Any]] = []
        for event in operator_events:
            operation_time = _profile_time(event, prepared.device)
            calls = int(event.count)
            if not math.isfinite(operation_time) or operation_time <= 0 or calls <= 0:
                continue
            hotspot: dict[str, Any] = {
                "name": str(event.key),
                "calls_per_forward": round(calls / iterations, 6),
                "self_time_us_per_forward": round(operation_time / iterations, 6),
                "share_pct": round(
                    operation_time / total_operator_self_time_us * 100.0,
                    3,
                ),
            }
            input_shapes = getattr(event, "input_shapes", [])
            if isinstance(input_shapes, (list, tuple)):
                tensor_shapes = [
                    list(shape)
                    for shape in input_shapes
                    if isinstance(shape, (list, tuple)) and shape
                ]
                if tensor_shapes:
                    hotspot["input_shapes"] = tensor_shapes
            operator_hotspots.append(hotspot)
            if len(operator_hotspots) == 8:
                break

        stage = "source_integrity"
        _assert_solution_source_integrity(
            prepared,
            phase="profiling completed",
        )

        return {
            "outcome": "success",
            "solution_source_sha256": measured_source_hash,
            "environment": environment,
            "correctness": correctness,
            "profile": {
                "iterations": iterations,
                "time_basis": "self_device_us_per_forward"
                if prepared.device.type == "cuda"
                else "self_cpu_us_per_forward",
                "total_self_time_us_per_forward": round(
                    total_operator_self_time_us / iterations,
                    6,
                ),
                "operator_hotspots": operator_hotspots,
                **(
                    {"observed_attention_backend": observed_attention_backend}
                    if observed_attention_backend is not None
                    else {}
                ),
            },
            "execution_path": execution_path,
            "failure": None,
        }
    except _PreparationFailure as failure:
        response = _failure_response(
            failure.cause,
            failure.stage,
            environment=failure.environment,
            correctness=None,
            performance=None,
            solution_hash=failure.solution_hash,
            execution_path=failure.execution_path,
        )
        response["profile"] = None
        return response
    except BaseException as exc:  # noqa: BLE001 - worker execution boundary.
        response = _failure_response(
            exc,
            stage,
            environment=environment,
            correctness=correctness,
            performance=None,
            solution_hash=measured_source_hash,
            execution_path=execution_path,
        )
        response["profile"] = None
        return response
