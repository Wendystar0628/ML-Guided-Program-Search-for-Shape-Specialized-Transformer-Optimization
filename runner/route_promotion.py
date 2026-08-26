"""Promote one formally measured eager winner into the dispatch table."""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from runner.contracts import ContractError, load_json, solution_implementation_hash
from solution.dispatch import resolve_route, validate_route_table

ROUTE_SCHEMA_VERSION = 1
DEFAULT_ROUTE_POLICY = "auto"
MINIMUM_ROUTE_GAIN = 1.02
_FORMAL_MINIMUM_COUNTS = {
    "accuracy_trials": 5,
    "warmup": 20,
    "repeats": 100,
    "rounds": 3,
}
_FORMAL_MAXIMUM_TOLERANCES = {"rtol": 0.01, "atol": 0.001}
DEPLOYABLE_EAGER_POLICIES = frozenset(
    {
        "auto",
        "reference",
        "torch",
        "triton",
        "preprocess",
        "s512-native-softmax",
        "long-pv",
        "long-tail-online",
        "wide-epilogue",
        "wide-triton-inplace",
        "cuda-graph",
        "balanced-cuda-graph",
    }
)

_SHAPE_FIELD_MAP = {
    "dtype": "dtype",
    "batch_size": "B",
    "seq_len": "S",
    "d_model": "D",
    "num_heads": "heads",
    "ffn_dim": "ffn",
    "num_layers": "layers",
    "causal": "causal",
}


def _positive_number(value: Any) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        return None
    return float(value)


def _paired_conservative_speedup(observation: Mapping[str, Any]) -> float | None:
    baseline = observation.get("baseline_round_medians_ms")
    target = observation.get("target_round_medians_ms")
    if (
        not isinstance(baseline, Sequence)
        or isinstance(baseline, (str, bytes))
        or not isinstance(target, Sequence)
        or isinstance(target, (str, bytes))
        or not baseline
        or len(baseline) != len(target)
    ):
        return None
    baseline_values = [_positive_number(value) for value in baseline]
    target_values = [_positive_number(value) for value in target]
    if any(value is None for value in (*baseline_values, *target_values)):
        return None
    return min(
        float(baseline_value) / float(target_value)
        for baseline_value, target_value in zip(
            baseline_values,
            target_values,
            strict=True,
        )
    )


def select_deployable_winner(
    summary: Mapping[str, Any],
    *,
    allowed_policies: frozenset[str] = DEPLOYABLE_EAGER_POLICIES,
) -> dict[str, Any]:
    """Select the best paired-baseline speedup among correct eager routes."""

    observations = summary.get("observations")
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise ContractError("tuning summary observations must be a sequence")

    eligible: list[tuple[float, float, str, Mapping[str, Any]]] = []
    for observation in observations:
        if not isinstance(observation, Mapping):
            continue
        policy = observation.get("solution_policy")
        candidate_id = observation.get("candidate_id")
        speedup = _positive_number(observation.get("conservative_speedup"))
        measured_speedup = _paired_conservative_speedup(observation)
        median = _positive_number(observation.get("target_median_ms"))
        p90 = _positive_number(observation.get("target_p90_ms"))
        execution_path = observation.get("execution_path")
        if (
            policy not in allowed_policies
            or not isinstance(candidate_id, str)
            or not candidate_id
            or observation.get("compile_solution") is not False
            or observation.get("cuda_graph_solution", False) is not False
            or observation.get("outcome") != "success"
            or observation.get("correctness_passed") is not True
            or observation.get("failed_elements") != 0
            or observation.get("policy_applied") is not True
            or speedup is None
            or measured_speedup is None
            or not math.isclose(speedup, measured_speedup, rel_tol=1e-12, abs_tol=1e-12)
            or median is None
            or p90 is None
            or not isinstance(execution_path, Mapping)
            or not isinstance(execution_path.get("shape_route"), str)
            or not execution_path["shape_route"]
        ):
            continue
        eligible.append(
            (
                -speedup,
                math.inf if p90 is None else p90,
                candidate_id,
                observation,
            )
        )

    if not eligible:
        raise ContractError(
            "tuning summary has no correct, applied, eager dispatch candidate"
        )
    return copy.deepcopy(dict(min(eligible, key=lambda item: item[:3])[3]))


def _route_match(summary: Mapping[str, Any]) -> dict[str, Any]:
    workload = summary.get("workload")
    case = workload.get("case") if isinstance(workload, Mapping) else None
    if not isinstance(case, Mapping):
        raise ContractError("tuning summary workload is missing its case object")

    match: dict[str, Any] = {}
    for source_name, route_name in _SHAPE_FIELD_MAP.items():
        if source_name not in case:
            raise ContractError(f"tuning workload case is missing {source_name}")
        match[route_name] = copy.deepcopy(case[source_name])

    profile = summary.get("device_profile")
    if not isinstance(profile, Mapping):
        raise ContractError("tuning summary is missing device_profile")
    for profile_name, route_name in (
        ("device_type", "device_type"),
        ("device_name", "device_name"),
        ("compute_capability", "compute_capability"),
    ):
        value = profile.get(profile_name)
        if not isinstance(value, str) or not value:
            raise ContractError(f"tuning device_profile is missing {profile_name}")
        match[route_name] = value
    return match


def _match_identity(match: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(match.items()))


def build_promoted_route_document(
    existing: Mapping[str, Any] | None,
    summary: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the updated route document without performing file I/O."""

    if summary.get("schema_version") != ROUTE_SCHEMA_VERSION:
        raise ContractError(
            f"unsupported tuning summary schema: {summary.get('schema_version')!r}"
        )
    if summary.get("complete") is not True:
        raise ContractError("only a complete tuning summary can be promoted")
    protocol = summary.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("preset") != "formal":
        raise ContractError("only a formal tuning summary can be promoted")
    for field, minimum in _FORMAL_MINIMUM_COUNTS.items():
        value = protocol.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ContractError(
                f"formal tuning summary requires {field} >= {minimum}"
            )
    seed = protocol.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ContractError("formal tuning summary requires an integer seed")
    for field, maximum in _FORMAL_MAXIMUM_TOLERANCES.items():
        value = protocol.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            or float(value) > maximum
        ):
            raise ContractError(
                f"formal tuning summary requires {field} <= {maximum}"
            )
    if protocol.get("matmul_precision") not in {"highest", "high", "medium"}:
        raise ContractError("formal tuning summary has invalid matmul_precision")
    if not isinstance(protocol.get("allow_tf32"), bool):
        raise ContractError("formal tuning summary requires allow_tf32")
    if summary.get("source_consistent") is not True:
        raise ContractError("tuning candidates do not share one Solution source")
    if summary.get("implementation_consistent") is not True:
        raise ContractError("Solution implementation changed during tuning")
    expected_implementation = summary.get("source_implementation_sha256")
    if not isinstance(expected_implementation, str) or not expected_implementation:
        raise ContractError("tuning summary is missing its implementation hash")
    expected_source = summary.get("source_solution_sha256")
    if not isinstance(expected_source, str) or not expected_source:
        raise ContractError("tuning summary is missing its Solution source hash")
    observations = summary.get("observations")
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise ContractError("tuning summary observations must be a sequence")
    if not observations or any(
        not isinstance(observation, Mapping)
        or observation.get("solution_sha256") != expected_source
        for observation in observations
    ):
        raise ContractError("tuning observation source hashes are missing or inconsistent")
    expected_rounds = int(protocol["rounds"])
    for observation in observations:
        if (
            isinstance(observation, Mapping)
            and observation.get("outcome") == "success"
            and observation.get("correctness_passed") is True
            and observation.get("policy_applied") is True
        ):
            baseline_rounds = observation.get("baseline_round_medians_ms")
            target_rounds = observation.get("target_round_medians_ms")
            if (
                not isinstance(baseline_rounds, Sequence)
                or isinstance(baseline_rounds, (str, bytes))
                or not isinstance(target_rounds, Sequence)
                or isinstance(target_rounds, (str, bytes))
                or len(baseline_rounds) != expected_rounds
                or len(target_rounds) != expected_rounds
            ):
                raise ContractError(
                    "successful tuning observations must contain every formal round"
                )

    match = _route_match(summary)
    if existing is None:
        document: dict[str, Any] = {
            "schema_version": ROUTE_SCHEMA_VERSION,
            "default_policy": DEFAULT_ROUTE_POLICY,
            "routes": [],
        }
    else:
        document = copy.deepcopy(dict(existing))
        try:
            existing_table = validate_route_table(document)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"invalid dispatch route document: {exc}") from exc

    if existing is None:
        existing_table = validate_route_table(document)

    winner = select_deployable_winner(summary)
    incumbent_policy = resolve_route(existing_table, match)
    policy = winner["solution_policy"]
    if policy != incumbent_policy:
        auto_winner = select_deployable_winner(
            summary,
            allowed_policies=frozenset({DEFAULT_ROUTE_POLICY}),
        )
        if (
            policy != DEFAULT_ROUTE_POLICY
            and float(winner["conservative_speedup"])
            < float(auto_winner["conservative_speedup"]) * MINIMUM_ROUTE_GAIN
        ):
            raise ContractError(
                "specialized winner does not exceed auto by the promotion margin"
            )
        if incumbent_policy != DEFAULT_ROUTE_POLICY:
            try:
                incumbent = select_deployable_winner(
                    summary,
                    allowed_policies=frozenset({incumbent_policy}),
                )
            except ContractError as exc:
                raise ContractError(
                    "tuning summary is missing a correct incumbent observation; "
                    "refusing to replace the existing route"
                ) from exc
            if float(winner["conservative_speedup"]) < (
                float(incumbent["conservative_speedup"]) * MINIMUM_ROUTE_GAIN
            ):
                raise ContractError(
                    "new winner does not exceed the incumbent by the promotion margin"
                )
    else:
        return document, winner

    identity = _match_identity(match)
    routes = [
        copy.deepcopy(dict(route))
        for route in document["routes"]
        if _match_identity(route["match"]) != identity
    ]
    provisional = copy.deepcopy(document)
    provisional["routes"] = routes
    try:
        provisional_table = validate_route_table(provisional)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"promoted dispatch route is invalid: {exc}") from exc
    if resolve_route(provisional_table, match) != policy:
        # A complete-shape exception must precede broad subset routes, including
        # when it intentionally restores the table's default policy.
        routes.insert(0, {"match": match, "policy": policy})
    document["routes"] = routes
    try:
        table = validate_route_table(document)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"promoted dispatch route is invalid: {exc}") from exc
    if resolve_route(table, match) != policy:
        raise ContractError("promoted dispatch route is shadowed by another route")
    return document, winner


def promote_tuning_summary(
    project_root: Path,
    summary: Mapping[str, Any],
    *,
    route_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Atomically publish a formally screened route."""

    destination = (
        route_path
        if route_path is not None
        else project_root.resolve() / "solution" / "dispatch_routes.json"
    )
    expected_implementation = summary.get("source_implementation_sha256")
    current_implementation = solution_implementation_hash(
        project_root.resolve() / "solution"
    )
    if current_implementation != expected_implementation:
        raise ContractError(
            "current Solution implementation does not match the tuning summary; "
            "rerun tuning"
        )
    existing = load_json(destination) if destination.is_file() else None
    document, winner = build_promoted_route_document(existing, summary)
    _atomic_replace_json(destination, document)
    return document, winner, destination


def _atomic_replace_json(path: Path, document: Mapping[str, Any]) -> None:
    """Replace the mutable route table with a same-directory atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        validate_route_table(json.loads(temporary_path.read_text(encoding="utf-8")))
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


__all__ = [
    "DEFAULT_ROUTE_POLICY",
    "DEPLOYABLE_EAGER_POLICIES",
    "MINIMUM_ROUTE_GAIN",
    "ROUTE_SCHEMA_VERSION",
    "build_promoted_route_document",
    "promote_tuning_summary",
    "select_deployable_winner",
]
