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


class _CudaGraphSolution(nn.Module):
    """Bounded eager CUDA Graph wrapper used only by the tuning runner."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.train(module.training)
        self._signature: tuple[object, ...] | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_input: torch.Tensor | None = None
        self._static_mask: torch.Tensor | None = None
        self._static_output: torch.Tensor | None = None

    @staticmethod
    def _input_signature(
        value: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> tuple[object, ...]:
        mask_signature = None
        if valid_mask is not None:
            mask_signature = (
                valid_mask.device,
                valid_mask.dtype,
                tuple(valid_mask.shape),
                tuple(valid_mask.stride()),
            )
        return (
            value.device,
            value.dtype,
            tuple(value.shape),
            tuple(value.stride()),
            mask_signature,
        )

    def _capture(
        self,
        value: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> None:
        if not value.is_cuda:
            raise ContractError("the eager CUDA Graph candidate requires CUDA input")
        if self.training or torch.is_grad_enabled():
            raise ContractError(
                "the eager CUDA Graph candidate requires eval inference mode"
            )
        self._signature = self._input_signature(value, valid_mask)
        self._static_input = value.detach().clone()
        self._static_mask = None if valid_mask is None else valid_mask.detach().clone()

        current_stream = torch.cuda.current_stream(value.device)
        capture_stream = torch.cuda.Stream(device=value.device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream), torch.inference_mode():
            for _ in range(3):
                self.module(self._static_input, self._static_mask)
        current_stream.wait_stream(capture_stream)

        graph = torch.cuda.CUDAGraph()
        with (
            torch.inference_mode(),
            torch.cuda.graph(
                graph,
                stream=capture_stream,
            ),
        ):
            static_output = self.module(self._static_input, self._static_mask)
        current_stream.wait_stream(capture_stream)
        self._graph = graph
        self._static_output = static_output

    def forward(
        self,
        value: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        signature = self._input_signature(value, valid_mask)
        if self._graph is None:
            self._capture(value, valid_mask)
        elif signature != self._signature:
            raise ContractError(
                "the eager CUDA Graph candidate received a different input signature"
            )

        assert self._static_input is not None
        assert self._static_output is not None
        assert self._graph is not None
        self._static_input.copy_(value)
        if valid_mask is not None:
            assert self._static_mask is not None
            self._static_mask.copy_(valid_mask)
        self._graph.replay()
        return self._static_output.clone()


def run_correctness(
    baseline: nn.Module,
    solution: nn.Module,
    config: official.TransformerConfig,
    case: WorkloadCase,
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
    if not passed:
        if max_relative_errors:
            summary["max_relative_error"] = max(max_relative_errors)
        if diagnostic is not None:
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
    round_medians = [
        statistics.median(normalized[index * repeats : (index + 1) * repeats])
        for index in range(rounds)
    ]
    return {
        "sample_count": expected_count,
        "median_ms": median,
        "p90_ms": p90,
        "round_medians_ms": round_medians,
    }


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
    stats = _measurement_stats(
        samples,
        repeats=protocol.repeats,
        rounds=protocol.rounds,
    )
    return {
        "timer": "cuda_event" if device.type == "cuda" else "perf_counter_ns",
        "baseline": stats,
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
        if target == "baseline" and protocol.cuda_graph_solution:
            raise ContractError("CUDA Graph wrapping applies only to the Solution")

        stage = "device"
        requested_device = str(request["device"])
        device = official.resolve_device(requested_device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        if device.type == "cpu" and protocol.preset != "smoke":
            raise ContractError("CPU execution is supported only by the smoke preset")
        environment = collect_environment(device)

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
        )
        _validate_cuda_graph_composition(execution_path, protocol)

        if solution is not None and protocol.cuda_graph_solution:
            if device.type != "cuda":
                raise ContractError("the eager CUDA Graph candidate requires CUDA")
            solution = _CudaGraphSolution(solution).eval()
            execution_path["runtime_wrapper"] = "eager_cuda_graph"

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
                    "solution_source_sha256": measured_source_hash,
                    "environment": environment,
                    "correctness": correctness,
                    "performance": None,
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
                "failed_elements": 0,
                "max_abs_error": 0.0,
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
            "solution_source_sha256": measured_source_hash,
            "environment": environment,
            "correctness": correctness,
            "performance": performance,
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
    return description


def _validate_cuda_graph_composition(
    execution_path: dict[str, Any],
    protocol: MeasurementProtocol,
) -> None:
    """Reject nested or compiled use of a Solution-owned CUDA Graph route."""

    if execution_path.get("runtime_wrapper") != "solution_eager_cuda_graph":
        return
    if protocol.compile_solution:
        raise ContractError(
            "the Solution CUDA Graph route cannot combine with torch.compile; "
            "select the auto policy for compile screening"
        )
    if protocol.cuda_graph_solution:
        raise ContractError(
            "the Solution CUDA Graph route cannot be wrapped by another CUDA Graph"
        )


def _validate_profile_execution_path(execution_path: dict[str, Any]) -> None:
    """Keep operator profiling on an eager path with visible ATen work."""

    if execution_path.get("runtime_wrapper") == "solution_eager_cuda_graph":
        raise ContractError(
            "the Solution CUDA Graph route hides per-operator profile work; "
            "select the auto policy to profile its eager computation body"
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
        if protocol.cuda_graph_solution:
            raise ContractError(
                "the eager CUDA Graph candidate is supported only by benchmark runs"
            )

        stage = "device"
        requested_device = str(request["device"])
        device = official.resolve_device(requested_device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        environment = collect_environment(device)

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
        )
        _validate_cuda_graph_composition(execution_path, protocol)
        _validate_profile_execution_path(execution_path)

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
                    "solution_source_sha256": measured_source_hash,
                    "environment": environment,
                    "correctness": correctness,
                    "profile": None,
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
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        _assert_unchanged("input", inputs, input_snapshot)
        _assert_unchanged("valid_token_mask", valid_mask, mask_snapshot)

        events = list(profiler.key_averages(group_by_input_shape=True))
        observed_attention_backend = _observed_attention_backend(events)
        if (
            observed_attention_backend is None
            and execution_path.get("selected_attention_backend") == "explicit"
        ):
            observed_attention_backend = "explicit"
        operator_events = [
            event for event in events if str(event.key).startswith("aten::")
        ]
        operator_events.sort(
            key=lambda event: _profile_time(event, device), reverse=True
        )
        total_operator_self_time_us = math.fsum(
            max(_profile_time(event, device), 0.0) for event in operator_events
        )
        if (
            not math.isfinite(total_operator_self_time_us)
            or total_operator_self_time_us <= 0
        ):
            raise ContractError("profiler did not record positive ATen self time")
        operator_hotspots: list[dict[str, Any]] = []
        for event in operator_events:
            operation_time = _profile_time(event, device)
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

        if measured_source_hash is not None:
            stage = "source_integrity"
            source_hash_after_run = solution_source_hash(project_root / "solution")
            if source_hash_after_run != measured_source_hash:
                raise ContractError(
                    "Solution source changed before profiling completed"
                )

        return {
            "outcome": "success",
            "solution_source_sha256": measured_source_hash,
            "environment": environment,
            "correctness": correctness,
            "profile": {
                "iterations": iterations,
                "time_basis": "self_device_us_per_forward"
                if device.type == "cuda"
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
