"""Finite, project-specific candidate screening for the performance mainline."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Any

from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    WorkloadCase,
    new_run_id,
)
from runner.supervisor import run_managed_benchmark

SOLUTION_POLICY_ENV = "TRANSFORMER_OPT_POLICY"
SOLUTION_POLICIES = ("auto", "torch", "triton", "padding", "packed")


@dataclass(frozen=True)
class TuningCandidate:
    """One bounded combination of model policy and compiler mode."""

    candidate_id: str
    solution_policy: str
    compile_solution: bool = False
    compile_mode: str = "default"


_EAGER_CANDIDATES = (
    TuningCandidate("eager-torch", "torch"),
    TuningCandidate("eager-auto", "auto"),
    TuningCandidate("eager-triton", "triton"),
)
_COMPILE_DEFAULT = TuningCandidate(
    "compile-default",
    "auto",
    True,
    "default",
)
_COMPILE_REDUCE_OVERHEAD = TuningCandidate(
    "compile-reduce-overhead",
    "auto",
    True,
    "reduce-overhead",
)
_PADDING_FUSION_CANDIDATE = TuningCandidate("padding-fused", "padding")
_PADDING_PACKED_CANDIDATE = TuningCandidate("padding-packed", "packed")
_WIDE_CANDIDATE = TuningCandidate(
    "compile-max-autotune",
    "auto",
    True,
    "max-autotune",
)


@contextmanager
def solution_policy(policy: str) -> Iterator[None]:
    """Temporarily select a Solution execution policy for a serial worker run."""

    if policy not in SOLUTION_POLICIES:
        raise ContractError(f"unsupported solution policy: {policy}")
    previous = os.environ.get(SOLUTION_POLICY_ENV)
    os.environ[SOLUTION_POLICY_ENV] = policy
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(SOLUTION_POLICY_ENV, None)
        else:
            os.environ[SOLUTION_POLICY_ENV] = previous


def candidates_for_case(case: WorkloadCase) -> tuple[TuningCandidate, ...]:
    """Return the small candidate set that is meaningful for one workload case."""

    candidates = list(_EAGER_CANDIDATES)
    if case.seq_len <= 128:
        candidates.extend((_COMPILE_DEFAULT, _COMPILE_REDUCE_OVERHEAD))
    if case.padding_ratio > 0 or case.case_id.startswith("mask_s512_"):
        candidates.extend((_PADDING_FUSION_CANDIDATE, _PADDING_PACKED_CANDIDATE))
    if case.d_model >= 1024 or case.ffn_dim >= 4096:
        if _COMPILE_DEFAULT not in candidates:
            candidates.append(_COMPILE_DEFAULT)
        candidates.append(_WIDE_CANDIDATE)
    return tuple(candidates)


def select_candidates(
    case: WorkloadCase,
    requested_ids: Sequence[str] | None,
) -> tuple[TuningCandidate, ...]:
    """Select explicit candidates while rejecting names that do not fit the case."""

    available = {item.candidate_id: item for item in candidates_for_case(case)}
    if not requested_ids:
        return tuple(available.values())
    unknown = sorted(set(requested_ids) - set(available))
    if unknown:
        raise ContractError(
            f"candidates are not available for {case.case_id}: {unknown}; "
            f"available={sorted(available)}"
        )
    return tuple(available[candidate_id] for candidate_id in requested_ids)


def _candidate_protocol(
    base: MeasurementProtocol,
    candidate: TuningCandidate,
) -> MeasurementProtocol:
    protocol = replace(
        base,
        compile_baseline=False,
        compile_solution=candidate.compile_solution,
        compile_mode=candidate.compile_mode,
    )
    protocol.validate()
    return protocol


def _observation(
    candidate: TuningCandidate,
    result: dict[str, Any],
    result_path: Path,
) -> dict[str, Any]:
    performance = result.get("performance")
    target = performance.get("target") if isinstance(performance, dict) else None
    baseline = performance.get("baseline") if isinstance(performance, dict) else None
    correctness = result.get("correctness")
    execution_path = result.get("execution_path")
    requested_policy = (
        execution_path.get("requested_policy")
        if isinstance(execution_path, dict)
        else None
    )
    selected_policy = (
        execution_path.get("selected_policy")
        if isinstance(execution_path, dict)
        else None
    )
    selected_matches = selected_policy == candidate.solution_policy
    if candidate.solution_policy == "triton":
        selected_matches = selected_policy in {"triton", "triton_partial"}
    route_matches = True
    if isinstance(execution_path, dict):
        if candidate.solution_policy == "torch":
            route_matches = (
                execution_path.get("resolved_qkv_layout")
                == "torch_three_contiguous_copies"
            )
        elif candidate.solution_policy == "padding":
            route_matches = (
                execution_path.get("block_fusion")
                == "triton_residual_add_padding_when_masked"
            )
        elif candidate.solution_policy == "packed":
            route_matches = (
                execution_path.get("padding_route") == "packed_valid_token_ffn"
            )
    policy_applied = (
        requested_policy == candidate.solution_policy
        and selected_matches
        and route_matches
    )
    return {
        "candidate_id": candidate.candidate_id,
        "solution_policy": candidate.solution_policy,
        "compile_solution": candidate.compile_solution,
        "compile_mode": candidate.compile_mode,
        "outcome": result.get("outcome"),
        "correctness_passed": (
            correctness.get("passed") if isinstance(correctness, dict) else None
        ),
        "failed_elements": (
            correctness.get("failed_elements")
            if isinstance(correctness, dict)
            else None
        ),
        "max_abs_error": (
            correctness.get("max_abs_error")
            if isinstance(correctness, dict)
            else None
        ),
        "baseline_median_ms": (
            baseline.get("median_ms") if isinstance(baseline, dict) else None
        ),
        "target_median_ms": (
            target.get("median_ms") if isinstance(target, dict) else None
        ),
        "target_p90_ms": target.get("p90_ms") if isinstance(target, dict) else None,
        "speedup": (
            performance.get("speedup") if isinstance(performance, dict) else None
        ),
        "policy_applied": policy_applied,
        "execution_path": execution_path,
        "result_path": str(result_path),
    }


def run_tuning_case(
    project_root: Path,
    *,
    workload_set_id: str,
    workload_sha256: str,
    case: WorkloadCase,
    base_protocol: MeasurementProtocol,
    device: str,
    requested_candidates: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run applicable candidates serially and return an in-memory comparison."""

    observations: list[dict[str, Any]] = []
    tuning_id = new_run_id()
    for candidate in select_candidates(case, requested_candidates):
        protocol = _candidate_protocol(base_protocol, candidate)
        with solution_policy(candidate.solution_policy):
            result, result_path = run_managed_benchmark(
                project_root,
                workload_set_id=workload_set_id,
                case=case,
                protocol=protocol,
                device=device,
                target="solution",
                workload_sha256=workload_sha256,
                sweep_id=tuning_id,
            )
        observations.append(_observation(candidate, result, result_path))
        if result.get("outcome") == "cancelled":
            break

    eligible = [
        item
        for item in observations
        if item["outcome"] == "success"
        and item["correctness_passed"] is True
        and item["policy_applied"] is True
        and isinstance(item["speedup"], (int, float))
        and isfinite(float(item["speedup"]))
        and float(item["speedup"]) > 0
        and isinstance(item["target_median_ms"], (int, float))
        and isfinite(float(item["target_median_ms"]))
        and float(item["target_median_ms"]) > 0
    ]
    winner = max(eligible, key=lambda item: float(item["speedup"])) if eligible else None
    return {
        "tuning_id": tuning_id,
        "case_id": case.case_id,
        "observations": observations,
        "winner": winner,
    }
