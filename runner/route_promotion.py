"""Promote one formally measured eager winner into the dispatch table."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from runner.contracts import ContractError, load_json, solution_implementation_hash
from solution.dispatch import ROUTE_FIELDS, resolve_route, validate_route_table

TUNING_SCHEMA_VERSION = 1
ROUTE_SCHEMA_VERSION = 2
DEFAULT_ROUTE_POLICY = "auto"
MINIMUM_ROUTE_GAIN = 1.02
_FORMAL_MINIMUM_COUNTS = {
    "accuracy_trials": 5,
    "warmup": 20,
    "repeats": 100,
    "rounds": 3,
}
_FORMAL_MAXIMUM_TOLERANCES = {"rtol": 0.01, "atol": 0.001}
_SHARED_ROUTE_CASE_GROUPS = (
    frozenset({"mask_s512_full_fp16", "mask_s512_padding_fp16"}),
)
DEPLOYABLE_EAGER_POLICIES = frozenset(
    {
        "auto",
        "reference",
        "torch",
        "triton",
        "preprocess",
        "padding",
        "packed",
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
_DEVICE_ROUTE_FIELDS = (
    "device_type",
    "device_name",
    "compute_capability",
    "platform_system",
    "torch",
    "cuda_runtime",
    "triton",
)


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
        ("platform_system", "platform_system"),
        ("torch", "torch"),
        ("cuda_runtime", "cuda_runtime"),
        ("triton", "triton"),
    ):
        value = profile.get(profile_name)
        if not isinstance(value, str) or not value:
            raise ContractError(f"tuning device_profile is missing {profile_name}")
        match[route_name] = value
    return match


def _match_identity(match: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(match.items()))


def _upsert_exact_route(
    document: Mapping[str, Any],
    match: Mapping[str, Any],
    policy: str,
) -> dict[str, Any]:
    """Return a validated route document with one exact decision updated."""

    updated = copy.deepcopy(dict(document))
    identity = _match_identity(match)
    routes = updated.get("routes")
    if not isinstance(routes, list):
        raise ContractError("dispatch route document is missing routes")
    exact = [
        route
        for route in routes
        if isinstance(route, Mapping)
        and isinstance(route.get("match"), Mapping)
        and _match_identity(route["match"]) == identity
    ]
    if len(exact) == 1 and exact[0].get("policy") == policy:
        try:
            current_table = validate_route_table(updated)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"invalid dispatch route document: {exc}") from exc
        if resolve_route(current_table, match) == policy:
            return updated

    retained = [
        copy.deepcopy(dict(route))
        for route in routes
        if not (
            isinstance(route, Mapping)
            and isinstance(route.get("match"), Mapping)
            and _match_identity(route["match"]) == identity
        )
    ]
    retained.insert(0, {"match": copy.deepcopy(dict(match)), "policy": policy})
    updated["routes"] = retained
    try:
        table = validate_route_table(updated)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"promoted dispatch route is invalid: {exc}") from exc
    if resolve_route(table, match) != policy:
        raise ContractError("promoted dispatch route is shadowed by another route")
    return updated


def _verified_profile_identity(profile: Mapping[str, Any]) -> dict[str, str]:
    """Extract the exact route identity owned by one verified device package."""

    if profile.get("schema_version") != 1:
        raise ContractError("verified profile must use schema_version 1")
    if profile.get("device_operation_passed") is not True:
        raise ContractError("verified profile did not pass its device operation")
    hardware = profile.get("hardware_profile")
    if not isinstance(hardware, Mapping):
        raise ContractError("verified profile is missing hardware_profile")
    gpu = hardware.get("gpu")
    platform_profile = hardware.get("platform")
    software = hardware.get("software")
    if not all(
        isinstance(section, Mapping)
        for section in (gpu, platform_profile, software)
    ):
        raise ContractError("verified profile has incomplete identity sections")
    assert isinstance(gpu, Mapping)
    assert isinstance(platform_profile, Mapping)
    assert isinstance(software, Mapping)
    raw_identity = {
        "device_type": hardware.get("device_type"),
        "device_name": gpu.get("name"),
        "compute_capability": gpu.get("compute_capability"),
        "platform_system": platform_profile.get("system"),
        "torch": software.get("torch"),
        "cuda_runtime": software.get("cuda_runtime"),
        "triton": software.get("triton"),
    }
    identity: dict[str, str] = {}
    for field, value in raw_identity.items():
        if not isinstance(value, str) or not value:
            raise ContractError(f"verified profile is missing {field}")
        identity[field] = value
    return identity


def _verified_bundle_identity(profile: Mapping[str, Any]) -> dict[str, str]:
    """Validate the package profile and return its route-visible identity."""

    identity = _verified_profile_identity(profile)
    hardware = profile.get("hardware_profile")
    assert isinstance(hardware, Mapping)
    platform_profile = hardware.get("platform")
    assert isinstance(platform_profile, Mapping)
    machine = platform_profile.get("machine")
    if not isinstance(machine, str) or not machine:
        raise ContractError("verified profile is missing platform_machine")
    return identity


def verified_profile_from_probe_result(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one portable verified-package profile from a successful probe."""

    if result.get("outcome") != "success":
        raise ContractError("automatic route publication requires a successful probe")
    probe = result.get("probe")
    if not isinstance(probe, Mapping):
        raise ContractError("successful probe result is missing its probe payload")
    if probe.get("device_operation_passed") is not True:
        raise ContractError("probe device operation did not pass")
    hardware_profile = probe.get("hardware_profile")
    runtime_policy = probe.get("runtime_policy")
    if not isinstance(hardware_profile, Mapping):
        raise ContractError("probe result is missing hardware_profile")
    if not isinstance(runtime_policy, Mapping):
        raise ContractError("probe result is missing runtime_policy")

    source_probe: dict[str, Any] = {}
    for field in ("run_id", "created_at", "requested_device"):
        value = result.get(field)
        if not isinstance(value, str) or not value:
            raise ContractError(f"probe result is missing {field}")
        source_probe[field] = value
    mode = probe.get("mode")
    if isinstance(mode, str) and mode:
        source_probe["mode"] = mode

    document: dict[str, Any] = {
        "schema_version": 1,
        "source_probe": source_probe,
        "device_operation_passed": True,
        "runtime_policy": copy.deepcopy(dict(runtime_policy)),
        "hardware_profile": copy.deepcopy(dict(hardware_profile)),
    }
    for field in ("performance_anchors", "sdpa"):
        value = probe.get(field)
        if isinstance(value, Mapping):
            document[field] = copy.deepcopy(dict(value))
    _verified_bundle_identity(document)
    return document


def find_matching_verified_route(
    project_root: Path,
    profile: Mapping[str, Any],
) -> Path | None:
    """Find the unique verified route table for one exact runtime identity."""

    expected = _verified_bundle_identity(profile)
    catalog_root = project_root.resolve() / "verified_hardware"
    if not catalog_root.is_dir():
        return None

    matches: list[Path] = []
    for profile_path in sorted(catalog_root.glob("*/profile.json")):
        try:
            candidate = load_json(profile_path)
            identity = _verified_bundle_identity(candidate)
        except (ContractError, OSError, ValueError):
            if profile_path.with_name("routes.json").is_file():
                raise ContractError(
                    f"verified package has an invalid profile: {profile_path.parent.name}"
                )
            continue
        if identity != expected:
            continue
        package_root = profile_path.parent
        required_paths = (
            package_root / "routes.json",
            package_root / "README.md",
            package_root / "run_verified.py",
            package_root / "results" / ".gitignore",
        )
        missing = [path.name for path in required_paths if not path.is_file()]
        if missing:
            raise ContractError(
                "matching verified package is incomplete: "
                f"{package_root.name}: {', '.join(missing)}"
            )
        route_path = package_root / "routes.json"
        try:
            validate_route_table(load_json(route_path))
        except (ContractError, TypeError, ValueError, OSError) as exc:
            raise ContractError(
                f"matching verified package has invalid routes: {package_root.name}"
            ) from exc
        matches.append(route_path)
    if len(matches) > 1:
        names = ", ".join(path.parent.name for path in matches)
        raise ContractError(
            f"multiple verified packages match the current runtime identity: {names}"
        )
    return matches[0] if matches else None


def _safe_bundle_id(device_name: str) -> str:
    normalized = unicodedata.normalize("NFKD", device_name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii").lower()
    value = re.sub(r"[^a-z0-9]+", "_", ascii_name).strip("_")
    if not value:
        raise ContractError("GPU name cannot be converted to a safe package id")
    return value[:80].rstrip("_")


def _new_bundle_path(catalog_root: Path, profile: Mapping[str, Any]) -> Path:
    identity = _verified_bundle_identity(profile)
    base = _safe_bundle_id(identity["device_name"])
    destination = catalog_root / base
    if not destination.exists():
        return destination
    payload = json.dumps(
        identity,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    suffix = hashlib.sha256(payload).hexdigest()[:12]
    destination = catalog_root / f"{base}__{suffix}"
    if destination.exists():
        raise ContractError(
            "verified package destination already exists but did not match the "
            f"current runtime identity: {destination.name}"
        )
    return destination


def _bundle_readme(device_name: str, bundle_id: str) -> str:
    return f"""# {device_name} verified package

This package was created by a complete Formal hardware calibration.

- `profile.json` records the probed hardware and software identity.
- `routes.json` contains only correctness-gated, formally measured exact routes.
- `run_verified.py` runs the shared Transformer workload with this route table.
- generated runs and summaries stay below `results/` and are ignored by Git.

From the repository root:

```powershell
python verified_hardware/{bundle_id}/run_verified.py --preset formal
```
"""


def _write_new_bundle_support_files(
    bundle_root: Path,
    profile: Mapping[str, Any],
    *,
    bundle_id: str,
) -> None:
    identity = _verified_bundle_identity(profile)
    bundle_root.mkdir(parents=True, exist_ok=False)
    results_root = bundle_root / "results"
    results_root.mkdir()
    (bundle_root / "README.md").write_text(
        _bundle_readme(identity["device_name"], bundle_id),
        encoding="utf-8",
    )
    (bundle_root / "run_verified.py").write_text(
        '\"\"\"Launch the shared verifier for this hardware bundle.\"\"\"\n\n'
        "from __future__ import annotations\n\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "PROJECT_ROOT = Path(__file__).resolve().parents[2]\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        'if __name__ == "__main__":\n'
        "    from runner.verified_hardware import main_for_bundle\n\n"
        "    raise SystemExit(main_for_bundle(Path(__file__).resolve().parent))\n",
        encoding="utf-8",
    )
    (results_root / ".gitignore").write_text(
        "*\n!.gitignore\n!reference_formal.json\n",
        encoding="utf-8",
    )
    _atomic_replace_json(
        bundle_root / "profile.json",
        profile,
        validate_as_route_table=False,
    )


def validate_promotion_case_set(case_ids: Sequence[str]) -> None:
    """Reject incomplete groups that share one runtime-visible route key."""

    selected = set(case_ids)
    for required_group in _SHARED_ROUTE_CASE_GROUPS:
        if selected & required_group and not required_group <= selected:
            missing = ", ".join(sorted(required_group - selected))
            raise ContractError(
                "shared runtime route requires formal calibration for: "
                f"{', '.join(sorted(required_group))}; missing: {missing}"
            )


def auto_promote_calibration(
    project_root: Path,
    summaries: Sequence[Mapping[str, Any]],
    *,
    probe_result: Mapping[str, Any],
    full_workload_case_ids: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], Path, bool]:
    """Publish one Formal calibration to its exact verified device package."""

    profile = verified_profile_from_probe_result(probe_result)
    existing_route = find_matching_verified_route(project_root, profile)
    case_ids = [_summary_case_id(summary) for summary in summaries]
    if len(case_ids) != len(set(case_ids)):
        raise ContractError("automatic promotion received duplicate workload cases")
    validate_promotion_case_set(case_ids)

    if existing_route is not None:
        document, winners, destination = promote_tuning_summaries(
            project_root,
            summaries,
            route_path=existing_route,
        )
        return document, winners, destination, False

    if set(case_ids) != set(full_workload_case_ids):
        raise ContractError(
            "a new verified hardware package requires one complete Formal workload "
            "calibration"
        )

    catalog_root = project_root.resolve() / "verified_hardware"
    catalog_root.mkdir(parents=True, exist_ok=True)
    destination_root = _new_bundle_path(catalog_root, profile)
    staging_root = catalog_root / ".staging"
    staging_root.mkdir(exist_ok=True)
    staging_bundle = Path(
        tempfile.mkdtemp(prefix=f"{destination_root.name}.", dir=staging_root)
    )
    try:
        staging_bundle.rmdir()
        _write_new_bundle_support_files(
            staging_bundle,
            profile,
            bundle_id=destination_root.name,
        )
        document, winners, _ = promote_tuning_summaries(
            project_root,
            summaries,
            route_path=staging_bundle / "routes.json",
        )
        if destination_root.exists():
            raise ContractError(
                f"verified package destination appeared concurrently: {destination_root}"
            )
        os.replace(staging_bundle, destination_root)
    finally:
        if staging_bundle.exists():
            shutil.rmtree(staging_bundle)
        try:
            staging_root.rmdir()
        except OSError:
            pass
    return document, winners, destination_root / "routes.json", True


def _validate_verified_package_identity(
    destination: Path,
    existing: Mapping[str, Any] | None,
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    """Keep every route in a verified package tied to its sibling profile."""

    resolved = destination.resolve()
    is_verified_target = any(
        part.lower() == "verified_hardware" for part in resolved.parts
    )
    if is_verified_target and resolved.name != "routes.json":
        raise ContractError(
            "verified route-table destination must be named routes.json"
        )
    profile_path = resolved.with_name("profile.json")
    if not is_verified_target and not profile_path.is_file():
        return
    if not profile_path.is_file():
        raise ContractError(
            "verified route table requires a sibling profile.json"
        )
    expected = _verified_profile_identity(load_json(profile_path))

    matches = [_route_match(summary) for summary in summaries]
    if existing is not None:
        try:
            existing_table = validate_route_table(existing)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"invalid dispatch route document: {exc}") from exc
        matches.extend(match for match, _policy in existing_table.routes)

    for match in matches:
        if set(match) != ROUTE_FIELDS:
            raise ContractError("verified device packages require exact routes")
        mismatches = [
            field
            for field in _DEVICE_ROUTE_FIELDS
            if match.get(field) != expected[field]
        ]
        if mismatches:
            raise ContractError(
                "route identity does not match the verified profile: "
                + ", ".join(mismatches)
            )


def build_promoted_route_document(
    existing: Mapping[str, Any] | None,
    summary: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the updated route document without performing file I/O."""

    if summary.get("schema_version") != TUNING_SCHEMA_VERSION:
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
            raise ContractError(f"formal tuning summary requires {field} >= {minimum}")
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
            raise ContractError(f"formal tuning summary requires {field} <= {maximum}")
    if protocol.get("matmul_precision") != "high":
        raise ContractError("formal route promotion requires matmul_precision=high")
    if protocol.get("allow_tf32") is not True:
        raise ContractError("formal route promotion requires allow_tf32=true")
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
        raise ContractError(
            "tuning observation source hashes are missing or inconsistent"
        )
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

    measured_winner = select_deployable_winner(summary)
    auto_observation = select_deployable_winner(
        summary,
        allowed_policies=frozenset({DEFAULT_ROUTE_POLICY}),
    )
    incumbent_policy = resolve_route(existing_table, match)
    if incumbent_policy == DEFAULT_ROUTE_POLICY:
        incumbent_observation = auto_observation
    else:
        try:
            incumbent_observation = select_deployable_winner(
                summary,
                allowed_policies=frozenset({incumbent_policy}),
            )
        except ContractError as exc:
            raise ContractError(
                "tuning summary is missing a correct incumbent observation; "
                "refusing to replace the existing route"
            ) from exc
    measured_policy = str(measured_winner["solution_policy"])
    deployment = measured_winner
    if measured_policy != incumbent_policy:
        measured_speedup = float(measured_winner["conservative_speedup"])
        auto_speedup = float(auto_observation["conservative_speedup"])
        incumbent_speedup = float(incumbent_observation["conservative_speedup"])
        clears_auto_margin = (
            measured_policy == DEFAULT_ROUTE_POLICY
            or measured_speedup >= auto_speedup * MINIMUM_ROUTE_GAIN
        )
        clears_incumbent_margin = (
            measured_speedup >= incumbent_speedup * MINIMUM_ROUTE_GAIN
        )
        if not (clears_auto_margin and clears_incumbent_margin):
            if (
                incumbent_policy != DEFAULT_ROUTE_POLICY
                and auto_speedup >= incumbent_speedup * MINIMUM_ROUTE_GAIN
            ):
                deployment = auto_observation
            else:
                deployment = incumbent_observation

    policy = str(deployment["solution_policy"])
    return _upsert_exact_route(document, match, policy), deployment


def promote_tuning_summary(
    project_root: Path,
    summary: Mapping[str, Any],
    *,
    route_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Atomically publish a formally screened route."""

    document, winners, destination = promote_tuning_summaries(
        project_root,
        [summary],
        route_path=route_path,
    )
    return document, winners[0], destination


def _summary_case_id(summary: Mapping[str, Any]) -> str:
    workload = summary.get("workload")
    case = workload.get("case") if isinstance(workload, Mapping) else None
    case_id = case.get("case_id") if isinstance(case, Mapping) else None
    if not isinstance(case_id, str) or not case_id:
        raise ContractError("tuning summary workload is missing case_id")
    return case_id


def promote_tuning_summaries(
    project_root: Path,
    summaries: Sequence[Mapping[str, Any]],
    *,
    route_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    """Atomically promote one or more formal summaries with shared-key guards."""

    if not summaries:
        raise ContractError("at least one tuning summary is required")
    if route_path is None:
        raise ContractError(
            "route_path is required; promote into an explicit verified device package"
        )

    destination = route_path
    current_implementation = solution_implementation_hash(
        project_root.resolve() / "solution"
    )
    for summary in summaries:
        expected_implementation = summary.get("source_implementation_sha256")
        if current_implementation != expected_implementation:
            raise ContractError(
                "current Solution implementation does not match the tuning summary; "
                "rerun tuning"
            )

    case_ids = {_summary_case_id(summary) for summary in summaries}
    for required_group in _SHARED_ROUTE_CASE_GROUPS:
        if case_ids & required_group and not required_group <= case_ids:
            missing = ", ".join(sorted(required_group - case_ids))
            raise ContractError(
                "shared runtime route requires formal summaries for: "
                f"{', '.join(sorted(required_group))}; missing: {missing}"
            )

    existing = load_json(destination) if destination.is_file() else None
    _validate_verified_package_identity(destination, existing, summaries)
    proposals: dict[
        tuple[tuple[str, Any], ...],
        list[tuple[int, Mapping[str, Any], dict[str, Any]]],
    ] = {}
    for index, summary in enumerate(summaries):
        _, deployment = build_promoted_route_document(existing, summary)
        identity = _match_identity(_route_match(summary))
        proposals.setdefault(identity, []).append((index, summary, deployment))

    document: dict[str, Any] = (
        copy.deepcopy(dict(existing))
        if existing is not None
        else {
            "schema_version": ROUTE_SCHEMA_VERSION,
            "default_policy": DEFAULT_ROUTE_POLICY,
            "routes": [],
        }
    )
    existing_table = validate_route_table(document)
    deployments_by_index: dict[int, dict[str, Any]] = {}
    for grouped in proposals.values():
        match = _route_match(grouped[0][1])
        policies = {str(item[2]["solution_policy"]) for item in grouped}
        if len(policies) == 1:
            policy = next(iter(policies))
            for index, _summary, deployment in grouped:
                deployments_by_index[index] = deployment
        else:
            policy = resolve_route(existing_table, match)
            for index, summary, _deployment in grouped:
                deployments_by_index[index] = select_deployable_winner(
                    summary,
                    allowed_policies=frozenset({policy}),
                )
        document = _upsert_exact_route(document, match, policy)

    winners = [deployments_by_index[index] for index in range(len(summaries))]
    if existing is None or document != existing:
        _atomic_replace_json(destination, document, validate_as_route_table=True)
    return document, winners, destination


def _atomic_replace_json(
    path: Path,
    document: Mapping[str, Any],
    *,
    validate_as_route_table: bool,
) -> None:
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
        loaded = json.loads(temporary_path.read_text(encoding="utf-8"))
        if validate_as_route_table:
            validate_route_table(loaded)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


__all__ = [
    "DEFAULT_ROUTE_POLICY",
    "DEPLOYABLE_EAGER_POLICIES",
    "MINIMUM_ROUTE_GAIN",
    "ROUTE_SCHEMA_VERSION",
    "TUNING_SCHEMA_VERSION",
    "auto_promote_calibration",
    "build_promoted_route_document",
    "find_matching_verified_route",
    "promote_tuning_summaries",
    "promote_tuning_summary",
    "select_deployable_winner",
    "validate_promotion_case_set",
    "verified_profile_from_probe_result",
]
