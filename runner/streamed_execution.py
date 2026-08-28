"""Execute workloads whose full batch cannot reside on the selected GPU."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from official import torch_transformer_benchmark as official
from project_identity import solution_implementation_hash
from runner.candidates import (
    CandidateSpec,
    candidate_spec_for_policy,
    candidate_specs_for_execution_mode,
)
from runner.contracts import ContractError, RunVariant, TransformerShape
from runner.model_runtime import (
    build_solution,
    config_for_shape,
    configure_runtime,
    load_solution_source,
)
from runner.probe import collect_environment
from runner.result_contracts import WorkerRequest
from runner.workload_execution import STREAMED_POLICY_SELECTOR, WorkloadExecutionPlan

_COMPARATOR_CHUNK_ELEMENTS = 4 * 1024 * 1024
_SCHEDULE_TIMING_REPEATS = 3
_STREAMED_MEMORY_BUDGET_FRACTION = 0.70
_STREAMED_MEMORY_GROWTH_MARGIN = 1.15


@dataclass(frozen=True, slots=True)
class _SelectedSchedule:
    candidate_spec: CandidateSpec
    comparison: dict[str, Any]
    execution_fingerprint: tuple[Any, ...]
    execution_path: dict[str, Any]
    timing_microbatch_size: int
    estimated_logical_batch_ms: float


def _microbatch_config(
    config: official.TransformerConfig,
    microbatch_size: int,
) -> official.TransformerConfig:
    result = official.TransformerConfig(
        batch_size=microbatch_size,
        seq_len=config.seq_len,
        d_model=config.d_model,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        num_layers=config.num_layers,
        causal=config.causal,
    )
    result.validate()
    return result


def _microbatch_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _generate_microbatch(
    config: official.TransformerConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    input_scale: float,
) -> torch.Tensor:
    """Generate one deterministic batch slice without materializing full B."""

    inputs = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    inputs.mul_(input_scale)
    return inputs


def _copy_weights(
    solution_module: Any,
    baseline: nn.Module,
    model: nn.Module,
) -> None:
    loader = getattr(solution_module, "copy_model_weights", None)
    if loader is None:
        official.copy_model_weights(baseline, model, strict=True)
    else:
        loader(baseline, model, strict=True)


def _chunked_compare(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Apply the official OR tolerance without full-sized error tensors."""

    if reference.shape != candidate.shape:
        raise ContractError(
            f"output shape mismatch: reference={tuple(reference.shape)}, "
            f"candidate={tuple(candidate.shape)}"
        )
    if reference.dtype != candidate.dtype or reference.device != candidate.device:
        raise ContractError("candidate output dtype or device does not match reference")

    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    failed_elements = 0
    max_abs_error = 0.0
    max_relative_error = 0.0
    for start in range(0, reference_flat.numel(), _COMPARATOR_CHUNK_ELEMENTS):
        stop = min(start + _COMPARATOR_CHUNK_ELEMENTS, reference_flat.numel())
        reference_chunk = reference_flat[start:stop].float()
        candidate_chunk = candidate_flat[start:stop].float()
        finite = torch.isfinite(reference_chunk) & torch.isfinite(candidate_chunk)
        absolute_error = (candidate_chunk - reference_chunk).abs()
        passed = finite & (
            (absolute_error <= atol) | (absolute_error <= rtol * reference_chunk.abs())
        )
        failed_elements += int((~passed).sum().item())
        max_abs_error = max(max_abs_error, float(absolute_error.max().item()))
        relative_error = absolute_error / reference_chunk.abs().clamp_min(1e-12)
        max_relative_error = max(
            max_relative_error,
            float(relative_error.max().item()),
        )
    return {
        "passed": failed_elements == 0,
        "failed_elements": failed_elements,
        "max_abs_error": max_abs_error,
        "max_relative_error": max_relative_error,
        "compared_elements": reference.numel(),
    }


def _execution_fingerprint(path: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        path.get("selected_policy"),
        path.get("attention_backend"),
        path.get("runtime_wrapper"),
        path.get("residual_norm_backend"),
        tuple(path.get("fallback_reasons", ())),
    )


def _planned_candidate_matches(
    path: Mapping[str, Any],
    candidate_spec: CandidateSpec,
) -> bool:
    """Validate the immutable plan without requiring fresh runtime observation.

    Candidate screening has already executed the backend with observation
    enabled. Final timing reconfigures the selected policy, which deliberately
    clears that diagnostic state. The timed path therefore rechecks the same
    selected policy and registered plan fields; forced backend calls still fail
    explicitly if the real kernel becomes unavailable.
    """

    evidence = candidate_spec.evidence
    if path.get("requested_policy") != candidate_spec.solution_policy:
        return False
    accepted = evidence.selected_policies or frozenset({candidate_spec.solution_policy})
    if path.get("selected_policy") not in accepted:
        return False
    return all(
        path.get(expectation.field) in expectation.accepted_values
        for expectation in evidence.path_expectations
    )


def _validated_execution_path(
    model: nn.Module,
    candidate_spec: CandidateSpec,
    *,
    expected_fingerprint: tuple[Any, ...] | None = None,
    require_observed_evidence: bool = True,
) -> dict[str, Any]:
    """Reject silent policy fallback and execution-path drift."""

    describe = getattr(model, "describe_execution_path", None)
    if not callable(describe):
        raise ContractError("streamed candidate cannot report its execution path")
    raw_path = describe()
    if not isinstance(raw_path, Mapping):
        raise ContractError("streamed candidate returned an invalid execution path")
    path = dict(raw_path)
    evidence_matches = (
        candidate_spec.evidence_matches(path)
        if require_observed_evidence
        else _planned_candidate_matches(path, candidate_spec)
    )
    if not evidence_matches:
        raise ContractError(
            f"streamed candidate {candidate_spec.solution_policy!r} did not "
            + (
                "produce its registered execution evidence"
                if require_observed_evidence
                else "retain its registered execution plan"
            )
        )
    fingerprint = _execution_fingerprint(path)
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ContractError("streamed execution path changed between microbatches")
    return path


def _cuda_forward_ms(
    model: nn.Module,
    inputs: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    candidate_spec: CandidateSpec,
    expected_fingerprint: tuple[Any, ...],
    repeats: int,
) -> list[float]:
    samples: list[float] = []
    with torch.inference_mode():
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(inputs, valid_mask)
            end.record()
            torch.cuda.synchronize(inputs.device)
            if not bool(torch.isfinite(output).all().item()):
                raise ContractError("streamed candidate produced non-finite output")
            _validated_execution_path(
                model,
                candidate_spec,
                expected_fingerprint=expected_fingerprint,
            )
            elapsed = float(start.elapsed_time(end))
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ContractError("CUDA events reported invalid target latency")
            samples.append(elapsed)
            del output
    return samples


def _run_logical_batch(
    model: nn.Module,
    *,
    config: official.TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    input_scale: float,
    microbatch_count: int,
    candidate_spec: CandidateSpec,
    expected_fingerprint: tuple[Any, ...],
) -> tuple[float, int]:
    generator = _microbatch_generator(device, seed)
    valid_mask = torch.ones(
        config.batch_size,
        config.seq_len,
        device=device,
        dtype=torch.bool,
    )
    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    completed = 0
    with torch.inference_mode():
        for _ in range(microbatch_count):
            inputs = _generate_microbatch(
                config,
                device=device,
                dtype=dtype,
                generator=generator,
                input_scale=input_scale,
            )
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(inputs, valid_mask)
            end.record()
            if not bool(torch.isfinite(output).all().item()):
                raise ContractError("streamed candidate produced non-finite output")
            _validated_execution_path(
                model,
                candidate_spec,
                expected_fingerprint=expected_fingerprint,
                require_observed_evidence=False,
            )
            events.append((start, end))
            completed += 1
            del inputs, output
    torch.cuda.synchronize(device)
    elapsed = sum(float(start.elapsed_time(end)) for start, end in events)
    if completed != microbatch_count:
        raise ContractError(
            "streamed workload did not complete every required microbatch"
        )
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ContractError("CUDA events reported invalid target latency")
    del valid_mask
    return elapsed, completed


def _full_workload_samples(
    model: nn.Module,
    *,
    config: official.TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    input_scale: float,
    microbatch_count: int,
    candidate_spec: CandidateSpec,
    expected_fingerprint: tuple[Any, ...],
    repeats: int,
    rounds: int,
) -> tuple[list[float], int]:
    samples: list[float] = []
    completed = 0
    for _ in range(rounds):
        for _ in range(repeats):
            elapsed, pass_completed = _run_logical_batch(
                model,
                config=config,
                device=device,
                dtype=dtype,
                seed=seed,
                input_scale=input_scale,
                microbatch_count=microbatch_count,
                candidate_spec=candidate_spec,
                expected_fingerprint=expected_fingerprint,
            )
            samples.append(elapsed)
            completed += pass_completed
    return samples, completed


def _end_to_end_ms(
    model: nn.Module,
    *,
    config: official.TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    input_scale: float,
    microbatch_count: int,
    candidate_spec: CandidateSpec,
    expected_fingerprint: tuple[Any, ...],
) -> tuple[float, int]:
    """Measure a bounded host-streamed pass using reusable host buffers."""

    host_input = torch.empty(
        config.batch_size,
        config.seq_len,
        config.d_model,
        device="cpu",
        dtype=dtype,
    )
    host_mask = torch.ones(
        config.batch_size,
        config.seq_len,
        device="cpu",
        dtype=torch.bool,
    )
    host_output = torch.empty_like(host_input)
    device_input = torch.empty_like(host_input, device=device)
    device_mask = torch.empty_like(host_mask, device=device)
    generator = _microbatch_generator(torch.device("cpu"), seed)
    host_input.normal_(generator=generator)
    host_input.mul_(input_scale)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    completed = 0
    output: torch.Tensor | None = None
    with torch.inference_mode():
        for _ in range(microbatch_count):
            device_input.copy_(host_input)
            device_mask.copy_(host_mask)
            output = model(device_input, device_mask)
            host_output.copy_(output)
            completed += 1
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if completed != microbatch_count:
        raise ContractError(
            "end-to-end stream did not complete every required microbatch"
        )
    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise ContractError("end-to-end streamed timing was invalid")
    if not bool(torch.isfinite(host_output).all().item()):
        raise ContractError("streamed candidate produced non-finite output")
    _validated_execution_path(
        model,
        candidate_spec,
        expected_fingerprint=expected_fingerprint,
        require_observed_evidence=False,
    )
    del host_input, host_mask, host_output, device_input, device_mask, output
    return elapsed_ms, completed


def _candidate_specs(
    shape: TransformerShape,
    variant: RunVariant,
    execution_mode: str,
    requested_policy: str | None,
) -> tuple[CandidateSpec, ...]:
    normalized = (requested_policy or STREAMED_POLICY_SELECTOR).strip().lower()
    if normalized == STREAMED_POLICY_SELECTOR:
        specs = candidate_specs_for_execution_mode(shape, variant, execution_mode)
        if specs:
            return specs
        raise ContractError("no registered candidate supports batch-streamed execution")
    spec = candidate_spec_for_policy(
        shape,
        variant,
        normalized,
        deployable_only=True,
    )
    if spec is None or execution_mode not in spec.workload_execution_modes:
        choices = ", ".join(
            candidate.solution_policy
            for candidate in candidate_specs_for_execution_mode(
                shape,
                variant,
                execution_mode,
            )
        )
        raise ContractError(
            "batch-streamed execution requires "
            f"{STREAMED_POLICY_SELECTOR} or one of {choices}"
        )
    return (spec,)


def _selection_digest(observations: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(observations),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _streamed_memory_budget(device: torch.device) -> int | None:
    """Keep streamed screening below the WDDM/Linux paging cliff."""

    try:
        properties = torch.cuda.get_device_properties(device)
        total_memory = int(properties.total_memory)
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return None
    if total_memory <= 0:
        return None
    return int(total_memory * _STREAMED_MEMORY_BUDGET_FRACTION)


def _screen_timing_schedules(
    model: nn.Module,
    *,
    full_config: official.TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    timing_microbatch_candidates: Sequence[int],
    validated_candidates: Sequence[
        tuple[
            CandidateSpec,
            dict[str, Any],
            tuple[Any, ...],
            dict[str, Any],
        ]
    ],
) -> _SelectedSchedule | None:
    """Measure short policy/schedule combinations and return the best one."""

    best: _SelectedSchedule | None = None
    best_estimated_ms: float | None = None
    configure = model.configure_runtime_policy
    set_observation = getattr(model, "set_execution_observation", None)
    memory_budget = _streamed_memory_budget(device)

    for (
        candidate_spec,
        comparison,
        validation_fingerprint,
        candidate_result,
    ) in validated_candidates:
        schedules: list[dict[str, Any]] = []
        policy = candidate_spec.solution_policy
        persistent_allocated: int | None = None
        one_sample_incremental_peak: int | None = None
        for timing_microbatch_size in timing_microbatch_candidates:
            microbatch_count = full_config.batch_size // timing_microbatch_size
            if (
                memory_budget is not None
                and persistent_allocated is not None
                and one_sample_incremental_peak is not None
            ):
                projected_peak = persistent_allocated + int(
                    one_sample_incremental_peak
                    * timing_microbatch_size
                    * _STREAMED_MEMORY_GROWTH_MARGIN
                )
                if projected_peak > memory_budget:
                    schedules.append(
                        {
                            "timing_microbatch_size": timing_microbatch_size,
                            "microbatch_count": microbatch_count,
                            "passed": False,
                            "failure_kind": "memory_guard",
                            "projected_peak_allocated_bytes": projected_peak,
                            "memory_budget_bytes": memory_budget,
                        }
                    )
                    continue
            timing_config = _microbatch_config(
                full_config,
                timing_microbatch_size,
            )
            inputs: torch.Tensor | None = None
            valid_mask: torch.Tensor | None = None
            output: torch.Tensor | None = None
            configure(policy=policy)
            if callable(set_observation):
                set_observation(True)
            try:
                try:
                    persistent_before = int(torch.cuda.memory_allocated(device))
                    torch.cuda.reset_peak_memory_stats(device)
                except (AssertionError, RuntimeError, TypeError, ValueError):
                    persistent_before = None
                inputs, valid_mask = official.generate_random_case(
                    config=timing_config,
                    device=device,
                    dtype=dtype,
                    seed=seed,
                    padding_ratio=padding_ratio,
                    input_scale=input_scale,
                )
                with torch.inference_mode():
                    output = model(inputs, valid_mask)
                if not bool(torch.isfinite(output).all().item()):
                    raise ContractError(
                        "streamed schedule candidate produced non-finite output"
                    )
                candidate_path = _validated_execution_path(
                    model,
                    candidate_spec,
                    expected_fingerprint=validation_fingerprint,
                )
                fingerprint = _execution_fingerprint(candidate_path)
                timings = _cuda_forward_ms(
                    model,
                    inputs,
                    valid_mask,
                    candidate_spec=candidate_spec,
                    expected_fingerprint=fingerprint,
                    repeats=_SCHEDULE_TIMING_REPEATS,
                )
                forward_median_ms = statistics.median(timings)
                estimated_logical_batch_ms = forward_median_ms * microbatch_count
                if timing_microbatch_size == 1 and persistent_before is not None:
                    try:
                        peak_allocated = int(torch.cuda.max_memory_allocated(device))
                    except (AssertionError, RuntimeError, TypeError, ValueError):
                        peak_allocated = persistent_before
                    persistent_allocated = persistent_before
                    one_sample_incremental_peak = max(
                        peak_allocated - persistent_before,
                        1,
                    )
                schedule = {
                    "timing_microbatch_size": timing_microbatch_size,
                    "microbatch_count": microbatch_count,
                    "passed": True,
                    "forward_median_ms": forward_median_ms,
                    "estimated_logical_batch_ms": estimated_logical_batch_ms,
                }
                schedules.append(schedule)
                if (
                    best_estimated_ms is None
                    or estimated_logical_batch_ms < best_estimated_ms
                ):
                    best_estimated_ms = estimated_logical_batch_ms
                    best = _SelectedSchedule(
                        candidate_spec=candidate_spec,
                        comparison=comparison,
                        execution_fingerprint=fingerprint,
                        execution_path=candidate_path,
                        timing_microbatch_size=timing_microbatch_size,
                        estimated_logical_batch_ms=estimated_logical_batch_ms,
                    )
            except (ContractError, RuntimeError, ValueError) as exc:
                schedules.append(
                    {
                        "timing_microbatch_size": timing_microbatch_size,
                        "microbatch_count": microbatch_count,
                        "passed": False,
                        "failure_kind": (
                            "oom"
                            if isinstance(exc, torch.cuda.OutOfMemoryError)
                            else "unsupported"
                        ),
                        "error": f"{type(exc).__name__}: {exc}"[-500:],
                    }
                )
            finally:
                if callable(set_observation):
                    set_observation(False)
                del inputs, valid_mask, output
                torch.cuda.empty_cache()
        candidate_result["timing_schedules"] = schedules
    return best


def _failure(
    exc: BaseException,
    stage: str,
    *,
    environment: dict[str, Any] | None,
    solution_hash: str | None,
    execution_path: dict[str, Any] | None,
    plan: WorkloadExecutionPlan | None = None,
) -> dict[str, Any]:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        outcome = "oom"
    elif isinstance(exc, KeyboardInterrupt):
        outcome = "cancelled"
    elif isinstance(exc, ContractError) and stage in {
        "request",
        "device",
        "workload_execution",
    }:
        outcome = "unsupported"
    elif stage in {"load_solution", "build_model", "copy_weights", "move_model"}:
        outcome = "build_error"
    else:
        outcome = "runtime_error"
    return {
        "outcome": outcome,
        "solution_source_sha256": solution_hash,
        "environment": environment,
        "correctness": None,
        "performance": None,
        "execution_path": execution_path,
        "workload_execution": (
            None
            if plan is None
            else {
                "mode": plan.execution_mode,
                "validation_microbatch_size": plan.validation_microbatch_size,
                "timing_microbatch_candidates": list(plan.timing_microbatch_candidates),
                "estimated_dense_attention_bytes": (
                    plan.estimated_dense_attention_bytes
                ),
                "reference_kind": plan.reference_kind,
                "validation_level": "provisional",
            }
        ),
        "failure": {
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc),
            "exit_code": None,
        },
    }


def execute_streamed_benchmark(
    request: WorkerRequest,
    plan: WorkloadExecutionPlan,
) -> dict[str, Any]:
    """Run a target-only full-batch-equivalent benchmark on GPU microbatches."""

    stage = "request"
    environment: dict[str, Any] | None = None
    solution_hash: str | None = None
    execution_path: dict[str, Any] | None = None
    try:
        if request.run_kind != "benchmark" or request.comparison_mode != "target_only":
            raise ContractError(
                "streamed execution requires target_only benchmark mode"
            )
        if request.target != "solution":
            raise ContractError("the official dense baseline cannot run streamed")
        assert request.project_root is not None
        assert request.shape is not None
        assert request.variant is not None
        assert request.protocol is not None
        if (
            not plan.is_streamed
            or plan.validation_microbatch_size is None
            or not plan.timing_microbatch_candidates
        ):
            raise ContractError("streamed executor received a resident workload")
        if request.variant.dtype != "float32":
            raise ContractError(
                "streamed official execution currently requires float32"
            )
        if request.variant.padding_ratio != 0:
            raise ContractError(
                "streamed mixed attention currently requires no padding"
            )
        if not request.shape.causal:
            raise ContractError(
                "streamed mixed attention currently requires causal=True"
            )

        stage = "device"
        device = official.resolve_device(request.device)
        if device.type != "cuda":
            raise ContractError("batch-streamed execution requires a CUDA device")
        torch.cuda.set_device(device)
        environment = collect_environment(device)
        dtype = official.resolve_dtype(request.variant.dtype)
        configure_runtime(request.protocol, device)

        stage = "load_solution"
        solution_module, solution_hash = load_solution_source(request.project_root)
        config = config_for_shape(request.shape)
        validation_config = _microbatch_config(
            config,
            plan.validation_microbatch_size,
        )
        if request.shape.batch_size % plan.validation_microbatch_size != 0:
            raise ContractError(
                "streamed workload batch must divide evenly into validation batches"
            )
        if any(
            size <= 0 or request.shape.batch_size % size != 0
            for size in plan.timing_microbatch_candidates
        ):
            raise ContractError(
                "streamed timing candidates must be positive batch divisors"
            )
        candidate_specs = _candidate_specs(
            request.shape,
            request.variant,
            plan.execution_mode,
            request.solution_policy,
        )

        stage = "build_model"
        # Model parameters do not depend on B or S. Building from the B=1
        # validation config prevents constructors from materializing a full
        # Shape-14 batch or a full-batch attention helper.
        baseline = official.BaselineTransformer(validation_config)
        reference_model = build_solution(solution_module, validation_config, "safe")
        target_model = build_solution(
            solution_module,
            validation_config,
            "safe",
        )
        stage = "copy_weights"
        _copy_weights(solution_module, baseline, reference_model)
        _copy_weights(solution_module, baseline, target_model)
        del baseline

        stage = "move_model"
        reference_model = reference_model.to(device=device, dtype=dtype).eval()
        target_model = target_model.to(device=device, dtype=dtype).eval()

        stage = "correctness"
        inputs, valid_mask = official.generate_random_case(
            config=validation_config,
            device=device,
            dtype=dtype,
            seed=request.protocol.seed,
            padding_ratio=request.variant.padding_ratio,
            input_scale=request.variant.input_scale,
        )
        input_snapshot = inputs.clone()
        mask_snapshot = valid_mask.clone()
        reference_start = torch.cuda.Event(enable_timing=True)
        reference_end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            reference_start.record()
            reference = reference_model(inputs, valid_mask)
            reference_end.record()
        torch.cuda.synchronize(device)
        reference_latency_ms = float(reference_start.elapsed_time(reference_end))
        if not torch.equal(inputs, input_snapshot) or not torch.equal(
            valid_mask, mask_snapshot
        ):
            raise ContractError("reference path modified its inputs")

        candidate_results: list[dict[str, Any]] = []
        validated_candidates: list[
            tuple[
                CandidateSpec,
                dict[str, Any],
                tuple[Any, ...],
                dict[str, Any],
            ]
        ] = []
        for candidate_spec in candidate_specs:
            policy = candidate_spec.solution_policy
            configure = target_model.configure_runtime_policy
            configure(policy=policy)
            set_observation = getattr(target_model, "set_execution_observation", None)
            if callable(set_observation):
                set_observation(True)
            try:
                with torch.inference_mode():
                    candidate = target_model(inputs, valid_mask)
                candidate_path = _validated_execution_path(
                    target_model,
                    candidate_spec,
                )
                candidate_fingerprint = _execution_fingerprint(candidate_path)
                comparison = _chunked_compare(
                    reference,
                    candidate,
                    rtol=request.protocol.rtol,
                    atol=request.protocol.atol,
                )
                if not torch.equal(inputs, input_snapshot) or not torch.equal(
                    valid_mask, mask_snapshot
                ):
                    raise ContractError("candidate path modified its inputs")
                if not comparison["passed"]:
                    candidate_results.append(
                        {
                            "policy": policy,
                            "comparator_passed": False,
                            "failed_elements": comparison["failed_elements"],
                            "max_abs_error": comparison["max_abs_error"],
                        }
                    )
                    del candidate
                    continue
                candidate_result = {
                    "policy": policy,
                    "comparator_passed": True,
                    "failed_elements": comparison["failed_elements"],
                    "max_abs_error": comparison["max_abs_error"],
                    "attention_backend": candidate_path["attention_backend"],
                }
                candidate_results.append(candidate_result)
                validated_candidates.append(
                    (
                        candidate_spec,
                        comparison,
                        candidate_fingerprint,
                        candidate_result,
                    )
                )
                del candidate
            except (ContractError, RuntimeError, ValueError) as exc:
                candidate_results.append(
                    {
                        "policy": policy,
                        "comparator_passed": False,
                        "error": f"{type(exc).__name__}: {exc}"[-500:],
                    }
                )
            finally:
                if callable(set_observation):
                    set_observation(False)

        if not validated_candidates:
            return {
                "outcome": "invalid_output",
                "solution_source_sha256": solution_hash,
                "environment": environment,
                "correctness": {
                    "passed": False,
                    "trial_count": 1,
                    "diagnostic": "no streamed candidate passed the comparator",
                },
                "performance": None,
                "execution_path": None,
                "workload_execution": {
                    "mode": plan.execution_mode,
                    "validation_microbatch_size": (plan.validation_microbatch_size),
                    "timing_microbatch_candidates": list(
                        plan.timing_microbatch_candidates
                    ),
                    "candidate_screening": candidate_results,
                },
                "failure": {
                    "stage": "correctness",
                    "type": "CorrectnessError",
                    "message": "no streamed candidate passed the comparator",
                    "exit_code": None,
                },
            }

        del (
            reference_model,
            reference,
            input_snapshot,
            mask_snapshot,
            inputs,
            valid_mask,
        )
        torch.cuda.empty_cache()

        stage = "schedule_screening"
        selected = _screen_timing_schedules(
            target_model,
            full_config=config,
            device=device,
            dtype=dtype,
            seed=request.protocol.seed,
            padding_ratio=request.variant.padding_ratio,
            input_scale=request.variant.input_scale,
            timing_microbatch_candidates=plan.timing_microbatch_candidates,
            validated_candidates=validated_candidates,
        )
        if selected is None:
            return {
                "outcome": "runtime_error",
                "solution_source_sha256": solution_hash,
                "environment": environment,
                "correctness": {
                    "passed": True,
                    "trial_count": 1,
                    "failed_elements": 0,
                    "max_abs_error": max(
                        float(item[1]["max_abs_error"]) for item in validated_candidates
                    ),
                    "diagnostic": (
                        "candidates passed B=1 comparison but no timing "
                        "microbatch schedule completed"
                    ),
                },
                "performance": None,
                "execution_path": None,
                "workload_execution": {
                    "mode": plan.execution_mode,
                    "validation_microbatch_size": (plan.validation_microbatch_size),
                    "timing_microbatch_candidates": list(
                        plan.timing_microbatch_candidates
                    ),
                    "candidate_screening": candidate_results,
                },
                "failure": {
                    "stage": "schedule_screening",
                    "type": "ScheduleSelectionError",
                    "message": "no streamed policy and schedule combination ran",
                    "exit_code": None,
                },
            }

        selected_candidate_spec = selected.candidate_spec
        selected_comparison = selected.comparison
        selected_fingerprint = selected.execution_fingerprint
        execution_path = selected.execution_path
        timing_microbatch_size = selected.timing_microbatch_size
        estimated_logical_batch_ms = selected.estimated_logical_batch_ms
        selected_policy = selected_candidate_spec.solution_policy
        timing_config = _microbatch_config(config, timing_microbatch_size)
        microbatch_count = request.shape.batch_size // timing_microbatch_size
        selection_digest = _selection_digest(candidate_results)
        selection = {
            "method": "runtime_policy_and_microbatch_screen",
            "policy": selected_policy,
            "timing_microbatch_size": timing_microbatch_size,
            "microbatch_count": microbatch_count,
            "estimated_logical_batch_ms": estimated_logical_batch_ms,
            "evidence_sha256": selection_digest,
        }
        execution_path["execution_mode"] = "eager"

        configure = target_model.configure_runtime_policy
        configure(policy=selected_policy)

        correctness = {
            "passed": True,
            "trial_count": 1,
            "failed_elements": selected_comparison["failed_elements"],
            "max_abs_error": selected_comparison["max_abs_error"],
            "max_relative_error": selected_comparison["max_relative_error"],
            "compared_elements": selected_comparison["compared_elements"],
            "reference_kind": plan.reference_kind,
            "reference_scope": "validation_microbatch",
            "validation_level": "provisional",
            "reference_latency_ms": reference_latency_ms,
        }
        stage = "timing"
        warmup_completed = 0
        for _ in range(request.protocol.warmup):
            _, completed = _run_logical_batch(
                target_model,
                config=timing_config,
                device=device,
                dtype=dtype,
                seed=request.protocol.seed,
                input_scale=request.variant.input_scale,
                microbatch_count=microbatch_count,
                candidate_spec=selected_candidate_spec,
                expected_fingerprint=selected_fingerprint,
            )
            warmup_completed += completed
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        samples, timed_completed = _full_workload_samples(
            target_model,
            config=timing_config,
            device=device,
            dtype=dtype,
            seed=request.protocol.seed,
            input_scale=request.variant.input_scale,
            microbatch_count=microbatch_count,
            candidate_spec=selected_candidate_spec,
            expected_fingerprint=selected_fingerprint,
            repeats=request.protocol.repeats,
            rounds=request.protocol.rounds,
        )
        end_to_end_ms, end_to_end_completed = _end_to_end_ms(
            target_model,
            config=timing_config,
            device=device,
            dtype=dtype,
            seed=request.protocol.seed,
            input_scale=request.variant.input_scale,
            microbatch_count=microbatch_count,
            candidate_spec=selected_candidate_spec,
            expected_fingerprint=selected_fingerprint,
        )
        # Capture after the host/device streaming buffers have been exercised;
        # otherwise the persisted peak understates the real execution path.
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
        target_stats = {
            "sample_count": len(samples),
            "median_ms": statistics.median(samples),
            "p90_ms": official.percentile(samples, 0.9),
        }
        performance = {
            "comparison_mode": "target_only",
            "timer": "cuda_event",
            "target": target_stats,
            "peak_device_allocated_bytes": peak_bytes,
            "end_to_end_ms": end_to_end_ms,
        }

        stage = "source_integrity"
        if (
            solution_implementation_hash(request.project_root / "solution")
            != solution_hash
        ):
            raise ContractError("Solution source changed while benchmark was running")

        return {
            "outcome": "success",
            "solution_source_sha256": solution_hash,
            "environment": environment,
            "correctness": correctness,
            "performance": performance,
            "execution_path": execution_path,
            "workload_execution": {
                "mode": plan.execution_mode,
                "validation_microbatch_size": plan.validation_microbatch_size,
                "timing_microbatch_candidates": list(plan.timing_microbatch_candidates),
                "timing_microbatch_size": timing_microbatch_size,
                "microbatch_count": microbatch_count,
                "microbatches_per_sample": microbatch_count,
                "completed_logical_batches": len(samples),
                "completed_microbatches": timed_completed,
                "warmup_microbatches": warmup_completed,
                "end_to_end_microbatches": end_to_end_completed,
                "estimated_dense_attention_bytes": (
                    plan.estimated_dense_attention_bytes
                ),
                "reference_kind": plan.reference_kind,
                "reference_scope": "validation_microbatch",
                "validation_level": "provisional",
                "selection": selection,
                "candidate_screening": candidate_results,
            },
            "failure": None,
        }
    except BaseException as exc:  # noqa: BLE001 - worker execution boundary.
        return _failure(
            exc,
            stage,
            environment=environment,
            solution_hash=solution_hash,
            execution_path=execution_path,
            plan=plan,
        )


__all__ = ["execute_streamed_benchmark"]
