"""Execute one baseline-versus-Solution benchmark using official primitives."""

from __future__ import annotations

import importlib
import math
import platform
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


def load_solution_module(project_root: Path) -> ModuleType:
    """Load the current Solution without depending on the caller's working directory."""

    solution_root = (project_root / "solution").resolve()
    source_path = solution_root / "transformer.py"
    if not source_path.is_file():
        raise ContractError(f"Solution entry file is missing: {source_path}")

    package_name = f"_benchmark_solution_{uuid.uuid4().hex}"
    bytecode_cache = tempfile.TemporaryDirectory(prefix="solution-bytecode-")
    sys.pycache_prefix = bytecode_cache.name
    sys.dont_write_bytecode = True
    package = types.ModuleType(package_name)
    package.__path__ = [str(solution_root)]
    package.__bytecode_cache__ = bytecode_cache
    sys.modules[package_name] = package
    module_name = f"{package_name}.transformer"
    try:
        module = importlib.import_module(module_name)
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
        raise ContractError(
            "Solution must export an nn.Module named UserOptimizedTransformer"
        )
    return module


def _environment(device: torch.device, requested_device: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "requested_device": requested_device,
        "resolved_device": str(device),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
    if device.type == "cuda":
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(index)
        value["gpu"] = {
            "index": index,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
        }
    return value


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
            inputs, valid_mask = official.generate_random_case(
                config=config,
                device=device,
                dtype=dtype,
                seed=seed,
                padding_ratio=case.padding_ratio,
                input_scale=protocol.input_scale,
            )
            reference = baseline(inputs, valid_mask)
            solution_output = solution(inputs, valid_mask)
            try:
                result = official.compare_outputs(
                    reference,
                    solution_output,
                    rtol=protocol.rtol,
                    atol=protocol.atol,
                )
                trials.append(_accuracy_record(seed, result))
            except AssertionError as exc:
                trials.append(
                    {
                        "seed": seed,
                        "passed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return {
        "passed": all(trial["passed"] for trial in trials),
        "trials": trials,
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
        input_scale=protocol.input_scale,
    )
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

    baseline_median = statistics.median(baseline_samples)
    solution_median = statistics.median(solution_samples)
    return {
        "timer": "cuda_event" if device.type == "cuda" else "perf_counter_ns",
        "baseline": {
            "samples_ms": baseline_samples,
            "median_ms": baseline_median,
        },
        "solution": {
            "samples_ms": solution_samples,
            "median_ms": solution_median,
        },
        "speedup": baseline_median / solution_median,
    }


def execute_benchmark(request: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(request["project_root"]).resolve()
    case = WorkloadCase.from_dict(request["case"])
    protocol = MeasurementProtocol(**request["protocol"])
    protocol.validate()
    requested_device = str(request["device"])
    device = official.resolve_device(requested_device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = official.resolve_dtype(case.dtype)

    torch.manual_seed(protocol.seed)
    torch.set_float32_matmul_precision(protocol.matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(protocol.seed)
        torch.backends.cuda.matmul.allow_tf32 = protocol.allow_tf32
        torch.backends.cudnn.allow_tf32 = protocol.allow_tf32

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

    baseline = official.BaselineTransformer(config)
    source_hash_before_load = solution_source_hash(project_root / "solution")
    solution_module = load_solution_module(project_root)
    measured_source_hash = solution_source_hash(project_root / "solution")
    if measured_source_hash != source_hash_before_load:
        raise ContractError("Solution source changed while it was being loaded")
    solution_class = solution_module.UserOptimizedTransformer
    solution = solution_class(config)
    weight_loader = getattr(solution_module, "copy_model_weights", None)
    if weight_loader is None:
        official.copy_model_weights(baseline, solution, strict=True)
    else:
        weight_loader(baseline, solution, strict=True)

    baseline = baseline.to(device=device, dtype=dtype).eval()
    solution = solution.to(device=device, dtype=dtype).eval()
    baseline = official.maybe_compile(
        baseline, protocol.compile_baseline, protocol.compile_mode
    )
    solution = official.maybe_compile(
        solution, protocol.compile_solution, protocol.compile_mode
    )

    correctness = run_correctness(
        baseline, solution, config, case, protocol, device, dtype
    )
    performance = None
    status = "correctness_failed"
    if correctness["passed"]:
        performance = run_performance(
            baseline, solution, config, case, protocol, device, dtype
        )
        status = "success"
    environment = _environment(device, requested_device)
    source_hash_after_run = solution_source_hash(project_root / "solution")
    failure = None
    if source_hash_after_run != measured_source_hash:
        status = "source_changed_during_run"
        failure = {
            "kind": "source_changed_during_run",
            "message": "Solution source changed before the benchmark completed",
        }
    return {
        "status": status,
        "solution_source_sha256": measured_source_hash,
        "environment": environment,
        "correctness": correctness,
        "performance": performance,
        "failure": failure,
    }
