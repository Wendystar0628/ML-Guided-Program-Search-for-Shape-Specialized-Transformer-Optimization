"""Execute benchmark and profile requests using the official primitives."""

from __future__ import annotations

import importlib
import math
import statistics
import sys
import tempfile
import types
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import nn

from official import torch_transformer_benchmark as official
from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    WorkloadCase,
    solution_source_hash,
)
from runner.probe import collect_environment


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


def _legacy_status(outcome: str) -> str:
    if outcome == "invalid_output":
        return "correctness_failed"
    if outcome == "cancelled":
        return "interrupted"
    return outcome


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _accuracy_record(seed: int, result: official.AccuracyResult) -> dict[str, Any]:
    return {
        "seed": seed,
        "passed": result.passed,
        "total_elements": result.total_elements,
        "failed_elements": result.failed_elements,
        "max_abs_error": _finite(result.max_abs_error),
        "max_relative_error": _finite(result.max_relative_error),
        "mean_abs_error": _finite(result.mean_abs_error),
    }


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
    case: WorkloadCase,
    protocol: MeasurementProtocol,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    with torch.inference_mode():
        for trial_index in range(protocol.accuracy_trials):
            seed = protocol.seed + trial_index
            reference: torch.Tensor | None = None
            inputs, valid_mask = official.generate_random_case(
                config=config,
                device=device,
                dtype=dtype,
                seed=seed,
                padding_ratio=case.padding_ratio,
                input_scale=case.input_scale,
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
                trials.append(_accuracy_record(seed, result))
            except (AssertionError, ContractError, TypeError, ValueError) as exc:
                trials.append(
                    {
                        "seed": seed,
                        "passed": False,
                        "total_elements": reference.numel()
                        if reference is not None
                        else None,
                        "failed_elements": None,
                        "max_abs_error": None,
                        "max_relative_error": None,
                        "mean_abs_error": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    finite_abs = [
        trial["max_abs_error"]
        for trial in trials
        if isinstance(trial.get("max_abs_error"), (int, float))
    ]
    finite_relative = [
        trial["max_relative_error"]
        for trial in trials
        if isinstance(trial.get("max_relative_error"), (int, float))
    ]
    failed_values = [trial.get("failed_elements") for trial in trials]
    failed_elements = (
        sum(int(value) for value in failed_values)
        if all(isinstance(value, int) for value in failed_values)
        else None
    )
    return {
        "passed": len(trials) == protocol.accuracy_trials
        and all(trial["passed"] for trial in trials),
        "trial_count": len(trials),
        "trials": trials,
        "failed_elements": failed_elements,
        "max_abs_error": max(finite_abs, default=None),
        "max_relative_error": max(finite_relative, default=None),
    }


def _measurement_stats(samples: list[float], expected_count: int) -> dict[str, Any]:
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
    return {"samples_ms": normalized, "median_ms": median}


def run_performance(
    baseline: nn.Module,
    solution: nn.Module,
    config: official.TransformerConfig,
    case: WorkloadCase,
    protocol: MeasurementProtocol,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    inputs, valid_mask = official.generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=protocol.seed + 100000,
        padding_ratio=case.padding_ratio,
        input_scale=case.input_scale,
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
    expected_count = protocol.repeats * protocol.rounds
    baseline_stats = _measurement_stats(baseline_samples, expected_count)
    solution_stats = _measurement_stats(solution_samples, expected_count)
    speedup = baseline_stats["median_ms"] / solution_stats["median_ms"]
    if not math.isfinite(speedup) or speedup <= 0:
        raise ContractError("speedup must be a finite positive number")
    return {
        "timer": "cuda_event" if device.type == "cuda" else "perf_counter_ns",
        "baseline": baseline_stats,
        "target": solution_stats,
        "solution": solution_stats,
        "speedup": speedup,
    }


def run_baseline_performance(
    baseline: nn.Module,
    config: official.TransformerConfig,
    case: WorkloadCase,
    protocol: MeasurementProtocol,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    inputs, valid_mask = official.generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=protocol.seed + 100000,
        padding_ratio=case.padding_ratio,
        input_scale=case.input_scale,
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
    stats = _measurement_stats(samples, protocol.repeats * protocol.rounds)
    return {
        "timer": "cuda_event" if device.type == "cuda" else "perf_counter_ns",
        "baseline": stats,
        "target": stats,
        "solution": None,
        "speedup": None,
    }


def _config_for_case(case: WorkloadCase) -> official.TransformerConfig:
    config = official.TransformerConfig(
        batch_size=case.batch_size,
        seq_len=case.seq_len,
        d_model=case.d_model,
        num_heads=case.num_heads,
        ffn_dim=case.ffn_dim,
        num_layers=case.num_layers,
        causal=case.causal,
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
    source_hash_before_load = solution_source_hash(project_root / "solution")
    solution_module = load_solution_module(project_root)
    measured_source_hash = solution_source_hash(project_root / "solution")
    if measured_source_hash != source_hash_before_load:
        raise ContractError("Solution source changed while it was being loaded")
    return solution_module, measured_source_hash


def _exception_outcome(exc: BaseException, stage: str) -> str:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return "oom"
    if isinstance(exc, KeyboardInterrupt):
        return "cancelled"
    if stage in {"device", "dtype"}:
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
    target: str,
    execution_path: dict[str, Any] | None,
) -> dict[str, Any]:
    outcome = _exception_outcome(exc, stage)
    return {
        "outcome": outcome,
        "status": _legacy_status(outcome),
        "solution_source_sha256": solution_hash,
        "environment": environment,
        "correctness": correctness,
        "performance": performance,
        "path": {"requested": target, "resolved": target},
        "execution_path": execution_path,
        "failure": {
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc),
            "exit_code": None,
        },
    }


def execute_benchmark(request: dict[str, Any]) -> dict[str, Any]:
    stage = "request"
    environment: dict[str, Any] | None = None
    correctness: dict[str, Any] | None = None
    performance: dict[str, Any] | None = None
    measured_source_hash: str | None = None
    execution_path: dict[str, Any] | None = None
    target = str(request.get("target", "solution"))
    try:
        if target not in {"baseline", "solution"}:
            raise ContractError(f"unsupported benchmark target: {target}")
        project_root = Path(request["project_root"]).resolve()
        case = WorkloadCase.from_dict(request["case"])
        protocol = MeasurementProtocol(**request["protocol"])
        protocol.validate()

        stage = "device"
        requested_device = str(request["device"])
        device = official.resolve_device(requested_device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        if device.type == "cpu" and protocol.preset != "smoke":
            raise ContractError("CPU execution is supported only by the smoke preset")
        environment = collect_environment(device, requested_device)

        stage = "dtype"
        dtype = official.resolve_dtype(case.dtype)
        _configure_runtime(protocol, device)
        config = _config_for_case(case)

        stage = "build_model"
        baseline = official.BaselineTransformer(config)
        solution: nn.Module | None = None
        if target == "solution":
            stage = "load_solution"
            solution_module, measured_source_hash = _load_solution_source(project_root)
            stage = "build_model"
            solution = solution_module.UserOptimizedTransformer(config)
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

        execution_path = _describe_execution_path(
            solution if solution is not None else baseline,
            case=case,
            protocol=protocol,
            target=target,
        )

        stage = "compile"
        baseline = official.maybe_compile(
            baseline, protocol.compile_baseline, protocol.compile_mode
        )
        if solution is not None:
            solution = official.maybe_compile(
                solution, protocol.compile_solution, protocol.compile_mode
            )

        if solution is not None:
            stage = "correctness"
            correctness = run_correctness(
                baseline, solution, config, case, protocol, device, dtype
            )
            if not correctness["passed"]:
                if measured_source_hash is not None:
                    stage = "source_integrity"
                    if (
                        solution_source_hash(project_root / "solution")
                        != measured_source_hash
                    ):
                        raise ContractError(
                            "Solution source changed before correctness completed"
                        )
                outcome = "invalid_output"
                return {
                    "outcome": outcome,
                    "status": _legacy_status(outcome),
                    "solution_source_sha256": measured_source_hash,
                    "environment": environment,
                    "correctness": correctness,
                    "performance": None,
                    "path": {"requested": target, "resolved": target},
                    "execution_path": execution_path,
                    "failure": {
                        "stage": "correctness",
                        "type": "CorrectnessError",
                        "message": "Solution failed the correctness contract",
                        "exit_code": None,
                    },
                }

            stage = "timing"
            performance = run_performance(
                baseline, solution, config, case, protocol, device, dtype
            )
        else:
            correctness = {
                "passed": True,
                "trial_count": 0,
                "trials": [],
                "failed_elements": 0,
                "max_abs_error": 0.0,
                "max_relative_error": 0.0,
                "skipped": "baseline target has no comparison candidate",
            }
            stage = "timing"
            performance = run_baseline_performance(
                baseline, config, case, protocol, device, dtype
            )

        if measured_source_hash is not None:
            stage = "source_integrity"
            source_hash_after_run = solution_source_hash(project_root / "solution")
            if source_hash_after_run != measured_source_hash:
                raise ContractError(
                    "Solution source changed before the benchmark completed"
                )

        return {
            "outcome": "success",
            "status": "success",
            "solution_source_sha256": measured_source_hash,
            "environment": environment,
            "correctness": correctness,
            "performance": performance,
            "path": {"requested": target, "resolved": target},
            "execution_path": execution_path,
            "failure": None,
        }
    except BaseException as exc:  # noqa: BLE001 - worker execution boundary.
        return _failure_response(
            exc,
            stage,
            environment=environment,
            correctness=correctness,
            performance=performance,
            solution_hash=measured_source_hash,
            target=target,
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
    case: WorkloadCase,
    protocol: MeasurementProtocol,
    target: str,
) -> dict[str, Any]:
    description: dict[str, Any]
    describe = getattr(model, "describe_execution_path", None)
    if callable(describe):
        value = describe()
        description = dict(value) if isinstance(value, dict) else {}
    else:
        description = {
            "qkv_projection": "separate",
            "attention_policy": "official_explicit",
            "selected_attention_backend": "explicit",
            "causal_mask": "per_forward" if case.causal else "none",
        }
    description.update(
        {
            "target": target,
            "dtype": case.dtype,
            "head_dim": case.d_model // case.num_heads,
            "causal": case.causal,
            "mask_kind": "prefix_padding" if case.padding_ratio > 0 else "all_valid",
            "compile": {
                "enabled": protocol.compile_solution
                if target == "solution"
                else protocol.compile_baseline,
                "mode": protocol.compile_mode,
            },
        }
    )
    return description


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


def execute_profile(request: dict[str, Any]) -> dict[str, Any]:
    stage = "request"
    environment: dict[str, Any] | None = None
    correctness: dict[str, Any] | None = None
    measured_source_hash: str | None = None
    execution_path: dict[str, Any] | None = None
    target = str(request.get("target", "solution"))
    try:
        if target not in {"baseline", "solution"}:
            raise ContractError(f"unsupported profile target: {target}")
        project_root = Path(request["project_root"]).resolve()
        case = WorkloadCase.from_dict(request["case"])
        protocol = MeasurementProtocol(**request["protocol"])
        protocol.validate()

        stage = "device"
        requested_device = str(request["device"])
        device = official.resolve_device(requested_device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        environment = collect_environment(device, requested_device)

        stage = "dtype"
        dtype = official.resolve_dtype(case.dtype)
        _configure_runtime(protocol, device)
        config = _config_for_case(case)

        stage = "build_model"
        baseline = official.BaselineTransformer(config)
        solution: nn.Module | None = None
        if target == "solution":
            stage = "load_solution"
            solution_module, measured_source_hash = _load_solution_source(project_root)
            stage = "build_model"
            solution = solution_module.UserOptimizedTransformer(config)
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

        execution_path = _describe_execution_path(
            solution if solution is not None else baseline,
            case=case,
            protocol=protocol,
            target=target,
        )

        stage = "compile"
        if target == "baseline":
            model = official.maybe_compile(
                baseline, protocol.compile_baseline, protocol.compile_mode
            )
        else:
            assert solution is not None
            baseline = official.maybe_compile(
                baseline, protocol.compile_baseline, protocol.compile_mode
            )
            model = official.maybe_compile(
                solution, protocol.compile_solution, protocol.compile_mode
            )

        if target == "solution":
            stage = "correctness"
            correctness = run_correctness(
                baseline, model, config, case, protocol, device, dtype
            )
            if not correctness["passed"]:
                if measured_source_hash is not None:
                    stage = "source_integrity"
                    if (
                        solution_source_hash(project_root / "solution")
                        != measured_source_hash
                    ):
                        raise ContractError(
                            "Solution source changed before correctness completed"
                        )
                outcome = "invalid_output"
                return {
                    "outcome": outcome,
                    "status": _legacy_status(outcome),
                    "solution_source_sha256": measured_source_hash,
                    "environment": environment,
                    "correctness": correctness,
                    "profile": None,
                    "path": {"requested": target, "resolved": target},
                    "execution_path": execution_path,
                    "failure": {
                        "stage": "correctness",
                        "type": "CorrectnessError",
                        "message": "Solution failed the correctness contract",
                        "exit_code": None,
                    },
                }
        else:
            correctness = None

        stage = "profile"
        inputs, valid_mask = official.generate_random_case(
            config=config,
            device=device,
            dtype=dtype,
            seed=protocol.seed + 100000,
            padding_ratio=case.padding_ratio,
            input_scale=case.input_scale,
        )
        input_snapshot = inputs.clone()
        mask_snapshot = valid_mask.clone()
        official.warmup_model(model, inputs, valid_mask, protocol.warmup, device)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        iterations = min(max(protocol.repeats, 1), 10)
        with (
            torch.profiler.profile(
                activities=activities,
                record_shapes=True,
            ) as profiler,
            torch.inference_mode(),
        ):
            for _ in range(iterations):
                model(inputs, valid_mask)
                profiler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        _assert_unchanged("input", inputs, input_snapshot)
        _assert_unchanged("valid_token_mask", valid_mask, mask_snapshot)

        events = list(profiler.key_averages())
        observed_attention_backend = _observed_attention_backend(events)
        if (
            observed_attention_backend is None
            and execution_path.get("selected_attention_backend") == "explicit"
        ):
            observed_attention_backend = "explicit"
        events.sort(key=lambda event: _profile_time(event, device), reverse=True)
        top_ops = [
            {
                "name": str(event.key),
                "calls": int(event.count),
                "self_cpu_time_total_us": _finite(
                    float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
                ),
                "self_device_time_total_us": _finite(
                    _profile_time(event, torch.device("cuda"))
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            for event in events[:15]
        ]

        if measured_source_hash is not None:
            stage = "source_integrity"
            source_hash_after_run = solution_source_hash(project_root / "solution")
            if source_hash_after_run != measured_source_hash:
                raise ContractError(
                    "Solution source changed before profiling completed"
                )

        return {
            "outcome": "success",
            "status": "success",
            "solution_source_sha256": measured_source_hash,
            "environment": environment,
            "correctness": correctness,
            "profile": {
                "profiler": "torch.profiler",
                "iterations": iterations,
                "sort_by": "self_device_time_total_us"
                if device.type == "cuda"
                else "self_cpu_time_total_us",
                "observed_attention_backend": observed_attention_backend,
                "top_ops_are_non_additive": True,
                "top_ops": top_ops,
            },
            "path": {"requested": target, "resolved": target},
            "execution_path": execution_path,
            "failure": None,
        }
    except BaseException as exc:  # noqa: BLE001 - worker execution boundary.
        response = _failure_response(
            exc,
            stage,
            environment=environment,
            correctness=correctness,
            performance=None,
            solution_hash=measured_source_hash,
            target=target,
            execution_path=execution_path,
        )
        response["profile"] = None
        return response
