"""Shared contracts for tuning summaries and deployable winner selection."""

from __future__ import annotations

import copy
import math
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from runner.candidates import candidate_spec, deployable_policy_ids
from runner.contracts import ContractError

TUNING_SCHEMA_VERSION = 4


def _positive_latency(value: Any) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        return None
    return float(value)


def observation_latency_key(
    observation: Mapping[str, Any],
) -> tuple[float, float, str]:
    """Return the deterministic target-latency order for one observation."""

    median = _positive_latency(observation.get("target_median_ms"))
    p90 = _positive_latency(observation.get("target_p90_ms"))
    candidate_id = observation.get("candidate_id")
    if median is None or p90 is None or not isinstance(candidate_id, str):
        raise ContractError("tuning observation has invalid target timing")
    return median, p90, candidate_id


def target_latency_gain(
    reference: Mapping[str, Any],
    challenger: Mapping[str, Any],
) -> float:
    """Return how many times faster the challenger target latency is."""

    reference_median, _, _ = observation_latency_key(reference)
    challenger_median, _, _ = observation_latency_key(challenger)
    return reference_median / challenger_median


def select_deployable_winner(
    summary: Mapping[str, Any],
    *,
    allowed_policies: Collection[str] | None = None,
) -> dict[str, Any]:
    """Select the lowest-latency correct candidate with execution evidence."""

    observations = summary.get("observations")
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise ContractError("tuning summary observations must be a sequence")
    policies = (
        frozenset(deployable_policy_ids())
        if allowed_policies is None
        else frozenset(allowed_policies)
    )

    eligible: list[tuple[tuple[float, float, str], Mapping[str, Any]]] = []
    for observation in observations:
        if not isinstance(observation, Mapping):
            continue
        policy = observation.get("solution_policy")
        candidate_id = observation.get("candidate_id")
        execution_path = observation.get("execution_path")
        spec = candidate_spec(candidate_id) if isinstance(candidate_id, str) else None
        if (
            policy not in policies
            or spec is None
            or not spec.deployable
            or spec.solution_policy != policy
            or observation.get("outcome") != "success"
            or observation.get("correctness_passed") is not True
            or observation.get("failed_elements") != 0
            or not isinstance(execution_path, Mapping)
            or not spec.evidence_matches(execution_path)
        ):
            continue
        try:
            latency_key = observation_latency_key(observation)
        except ContractError:
            continue
        eligible.append((latency_key, observation))

    if not eligible:
        raise ContractError("tuning summary has no correct, applied dispatch candidate")
    return copy.deepcopy(dict(min(eligible, key=lambda item: item[0])[1]))


__all__ = [
    "TUNING_SCHEMA_VERSION",
    "observation_latency_key",
    "select_deployable_winner",
    "target_latency_gain",
]
