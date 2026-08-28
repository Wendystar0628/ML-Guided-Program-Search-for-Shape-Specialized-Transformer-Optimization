"""Promote one formally measured runtime policy into the dispatch table."""

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

from project_identity import official_snapshot_hash, solution_implementation_hash
from route_contracts import (
    MANIFEST_SCHEMA_VERSION,
    ROUTE_FIELDS,
    SCHEMA_VERSION,
    WORKLOAD_ROUTE_FIELDS,
    load_verified_bundle,
    resolve_route,
    validate_bundle_manifest,
    validate_route_table,
    validate_verified_route_table,
)
from runner.candidates import exact_route_policy_ids
from runner.contracts import (
    ContractError,
    RunVariant,
    TransformerShape,
    load_json,
    load_workload_set,
)
from runner.locking import (
    bundle_lock_path,
    exclusive_file_lock,
    hardware_bundle_lock_path,
)
from runner.routing_contracts import (
    hardware_identity_from_verified_profile,
    route_match_from_summary,
    validate_selected_route_groups,
    workload_route_identity,
)
from runner.tuning_contracts import (
    TUNING_SCHEMA_VERSION,
    select_deployable_winner,
    target_latency_gain,
)
from runner.workload_execution import (
    all_benchmark_shapes,
    plan_workload_execution,
    route_eligible_shapes,
)

DEFAULT_ROUTE_POLICY = "eager-sdpa"
MINIMUM_ROUTE_GAIN = 1.02
_FORMAL_MINIMUM_COUNTS = {
    "accuracy_trials": 5,
    "warmup": 20,
    "repeats": 100,
    "rounds": 3,
}
_FORMAL_MAXIMUM_TOLERANCES = {"rtol": 0.02, "atol": 0.002}
_BUNDLE_HARDWARE_FIELDS = frozenset(
    {"device_type", "device_name", "compute_capability"}
)


def _route_match(summary: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return route_match_from_summary(summary)
    except (TypeError, ValueError) as exc:
        raise ContractError(str(exc)) from exc


def _match_identity(match: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(match.items()))


def _upsert_exact_route(
    document: Mapping[str, Any],
    match: Mapping[str, Any],
    policy: str,
) -> dict[str, Any]:
    """Return a validated route document with one exact decision updated."""

    if policy not in exact_route_policy_ids():
        raise ContractError(
            f"policy {policy!r} is not eligible for a resident exact route"
        )

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

    try:
        return hardware_identity_from_verified_profile(profile).as_route_fields()
    except (TypeError, ValueError) as exc:
        raise ContractError(str(exc)) from exc


def _verified_bundle_identity(profile: Mapping[str, Any]) -> dict[str, str]:
    """Return the stable physical identity owned by one device directory."""

    identity = _verified_profile_identity(profile)
    hardware = profile.get("hardware_profile")
    assert isinstance(hardware, Mapping)
    platform_profile = hardware.get("platform")
    assert isinstance(platform_profile, Mapping)
    machine = platform_profile.get("machine")
    if not isinstance(machine, str) or not machine:
        raise ContractError("verified profile is missing platform_machine")
    return {field: identity[field] for field in _BUNDLE_HARDWARE_FIELDS}


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
    *,
    workload_set_id: str | None = None,
    workload_sha256: str | None = None,
) -> Path | None:
    """Find the stable bundle for one physical GPU identity."""

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
            package_root / "manifest.json",
            package_root / "routes.json",
            package_root / "README.md",
            package_root / "run_verified.py",
            package_root / "results" / ".gitignore",
        )
        missing = [path.name for path in required_paths if not path.is_file()]
        if missing:
            continue
        route_path = package_root / "routes.json"
        try:
            _table, _digest, manifest = load_verified_bundle(
                route_path,
                project_root=project_root,
            )
            validate_verified_route_table(
                load_json(route_path),
                expected_identity=expected,
            )
        except (ContractError, TypeError, ValueError, OSError):
            continue
        if workload_set_id is not None and manifest.workload_set_id != workload_set_id:
            continue
        if (
            workload_sha256 is not None
            and manifest.workload_set_sha256 != workload_sha256
        ):
            continue
        matches.append(route_path)
    if len(matches) > 1:
        names = ", ".join(path.parent.name for path in matches)
        raise ContractError(
            f"multiple verified packages match the current runtime identity: {names}"
        )
    return matches[0] if matches else None


def _find_recalibration_bundle(
    project_root: Path,
    profile: Mapping[str, Any],
) -> Path | None:
    """Locate the stable hardware directory whose contents need replacement."""

    expected = _verified_bundle_identity(profile)
    matches: list[Path] = []
    catalog_root = project_root.resolve() / "verified_hardware"
    for profile_path in sorted(catalog_root.glob("*/profile.json")):
        try:
            candidate = load_json(profile_path)
            if _verified_bundle_identity(candidate) != expected:
                continue
        except (ContractError, OSError, ValueError):
            continue
        route_path = profile_path.with_name("routes.json")
        required_support = (
            profile_path.with_name("README.md"),
            profile_path.with_name("run_verified.py"),
            profile_path.parent / "results" / ".gitignore",
        )
        if not route_path.is_file() or any(
            not path.is_file() for path in required_support
        ):
            continue
        matches.append(route_path)
    if len(matches) > 1:
        names = ", ".join(path.parent.name for path in matches)
        raise ContractError(
            "multiple stale verified packages match this calibration: " + names
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
- `manifest.json` binds routes to the Workload, Solution, and Formal evidence.
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
        '"""Launch the shared verifier for this hardware bundle."""\n\n'
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
        "*\n!.gitignore\n!reference_formal.json\n!reference_streamed.json\n",
        encoding="utf-8",
    )
    _atomic_replace_json(
        bundle_root / "profile.json",
        profile,
        validate_as_route_table=False,
    )


def validate_promotion_case_set(
    case_ids: Sequence[str],
    all_shapes: Sequence[TransformerShape] | None = None,
    variant: RunVariant | None = None,
) -> None:
    """Validate shared routes from workload shape fields, never case-id names."""

    if len(case_ids) != len(set(case_ids)):
        raise ContractError("automatic promotion received duplicate workload cases")
    if all_shapes is None:
        return
    if variant is None:
        raise ContractError("promotion route validation requires a run variant")
    eligible_case_ids = {shape.case_id for shape in all_shapes}
    ineligible_case_ids = set(case_ids) - eligible_case_ids
    if ineligible_case_ids:
        raise ContractError(
            "formal promotion accepts only shapes with a formal reference path; "
            "currently provisional or unknown: "
            + ", ".join(sorted(ineligible_case_ids))
        )
    try:
        validate_selected_route_groups(case_ids, all_shapes, variant)
    except (TypeError, ValueError) as exc:
        raise ContractError(str(exc)) from exc


def _summary_workload_identity(
    summaries: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    identities: set[tuple[str, str]] = set()
    for summary in summaries:
        workload = summary.get("workload")
        set_id = workload.get("set_id") if isinstance(workload, Mapping) else None
        digest = workload.get("sha256") if isinstance(workload, Mapping) else None
        if not isinstance(set_id, str) or not set_id:
            raise ContractError("tuning summary workload is missing set_id")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ContractError("tuning summary workload is missing its SHA-256")
        identities.add((set_id, digest))
    if len(identities) != 1:
        raise ContractError("formal summaries do not share one workload set")
    return next(iter(identities))


def _summary_variant(summaries: Sequence[Mapping[str, Any]]) -> RunVariant:
    """Return the single run variant shared by a calibration publication."""

    variants: set[str] = set()
    payloads: list[dict[str, Any]] = []
    for summary in summaries:
        workload = summary.get("workload")
        raw_variant = workload.get("variant") if isinstance(workload, Mapping) else None
        if not isinstance(raw_variant, dict):
            raise ContractError("tuning summary workload is missing its run variant")
        payloads.append(raw_variant)
        variants.add(
            json.dumps(
                raw_variant,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    if len(variants) != 1:
        raise ContractError("formal summaries do not share one run variant")
    try:
        return RunVariant.from_dict(payloads[0])
    except (TypeError, ValueError) as exc:
        raise ContractError(str(exc)) from exc


def _auto_promote_calibration_locked(
    project_root: Path,
    summaries: Sequence[Mapping[str, Any]],
    *,
    probe_result: Mapping[str, Any],
    full_workload_shape_ids: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], Path, bool]:
    """Publish one Formal calibration to its exact verified device package."""

    profile = verified_profile_from_probe_result(probe_result)
    case_ids = [_summary_case_id(summary) for summary in summaries]
    workload_set_id, workload_sha256 = _summary_workload_identity(summaries)
    workload_set = load_workload_set(project_root, workload_set_id)
    if workload_set.sha256 != workload_sha256:
        raise ContractError("formal summary workload hash is stale; rerun calibration")
    formal_variant = _summary_variant(summaries)
    formal_shapes = route_eligible_shapes(workload_set.shapes, formal_variant)
    authoritative_case_ids = [shape.case_id for shape in formal_shapes]
    if set(authoritative_case_ids) != set(full_workload_shape_ids):
        raise ContractError(
            "calibration shape list does not match the persisted workload set"
        )
    validate_promotion_case_set(case_ids, formal_shapes, formal_variant)
    existing_route = find_matching_verified_route(
        project_root,
        profile,
        workload_set_id=workload_set_id,
        workload_sha256=workload_sha256,
    )

    if existing_route is not None:
        document, winners, destination = _publish_bundle_tuning_summaries(
            project_root,
            summaries,
            route_path=existing_route,
            verified_profile=profile,
        )
        return document, winners, destination, False

    if set(case_ids) != set(full_workload_shape_ids):
        raise ContractError(
            "a new verified hardware package requires one complete Formal workload "
            "calibration"
        )

    recalibration_route = _find_recalibration_bundle(
        project_root,
        profile,
    )
    if recalibration_route is not None:
        document, winners, destination = _publish_bundle_tuning_summaries(
            project_root,
            summaries,
            route_path=recalibration_route,
            reset_verified_bundle=True,
            verified_profile=profile,
        )
        return document, winners, destination, False

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
        document, winners, _ = _publish_bundle_tuning_summaries(
            project_root,
            summaries,
            route_path=staging_bundle / "routes.json",
            verified_profile=profile,
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


def auto_promote_calibration(
    project_root: Path,
    summaries: Sequence[Mapping[str, Any]],
    *,
    probe_result: Mapping[str, Any],
    full_workload_shape_ids: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], Path, bool]:
    """Discover and update one stable GPU bundle under a catalog lock."""

    profile = verified_profile_from_probe_result(probe_result)
    identity = _verified_bundle_identity(profile)
    hardware_id = _safe_bundle_id(identity["device_name"])
    with exclusive_file_lock(
        hardware_bundle_lock_path(project_root, hardware_id),
        purpose=f"verified bundle selection for {identity['device_name']}",
    ):
        return _auto_promote_calibration_locked(
            project_root,
            summaries,
            probe_result=probe_result,
            full_workload_shape_ids=full_workload_shape_ids,
        )


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
    if not is_verified_target or resolved.name != "routes.json":
        raise ContractError(
            "route publication requires a verified_hardware bundle routes.json"
        )
    profile_path = resolved.with_name("profile.json")
    if not profile_path.is_file():
        raise ContractError("verified route table requires a sibling profile.json")
    expected = _verified_bundle_identity(load_json(profile_path))

    matches = [_route_match(summary) for summary in summaries]
    if existing is not None:
        try:
            existing_table = validate_verified_route_table(
                existing,
                expected_identity=expected,
            )
        except (TypeError, ValueError) as exc:
            raise ContractError(f"invalid dispatch route document: {exc}") from exc
        matches.extend(match for match, _policy in existing_table.routes)

    for match in matches:
        if set(match) != ROUTE_FIELDS:
            raise ContractError("verified device packages require exact routes")
        mismatches = [
            field
            for field in _BUNDLE_HARDWARE_FIELDS
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
    if (
        protocol.get("compile_baseline") is not False
        or protocol.get("compile_solution") is not False
    ):
        raise ContractError("formal route promotion requires uncompiled models")
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
    if summary.get("official_consistent") is not True:
        raise ContractError("official snapshot changed during tuning")
    official_hash = summary.get("official_snapshot_sha256")
    if not isinstance(official_hash, str) or len(official_hash) != 64:
        raise ContractError("tuning summary is missing its official snapshot hash")
    expected_implementation = summary.get("source_implementation_sha256")
    if not isinstance(expected_implementation, str) or not expected_implementation:
        raise ContractError("tuning summary is missing its implementation hash")
    expected_source = summary.get("source_solution_sha256")
    if not isinstance(expected_source, str) or not expected_source:
        raise ContractError("tuning summary is missing its Solution source hash")
    if expected_source != expected_implementation:
        raise ContractError(
            "tuning summary Solution source and implementation hashes disagree"
        )
    observations = summary.get("observations")
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise ContractError("tuning summary observations must be a sequence")
    if not observations or any(
        not isinstance(observation, Mapping)
        or observation.get("solution_sha256") != expected_source
        or observation.get("official_snapshot_sha256") != official_hash
        for observation in observations
    ):
        raise ContractError(
            "tuning observation source identities are missing or inconsistent"
        )
    match = _route_match(summary)
    if existing is None:
        document: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
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
    eager_sdpa_observation = select_deployable_winner(
        summary,
        allowed_policies=frozenset({DEFAULT_ROUTE_POLICY}),
    )
    incumbent_policy = resolve_route(existing_table, match)
    if incumbent_policy == DEFAULT_ROUTE_POLICY:
        incumbent_observation = eager_sdpa_observation
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
        clears_default_margin = (
            measured_policy == DEFAULT_ROUTE_POLICY
            or target_latency_gain(eager_sdpa_observation, measured_winner)
            >= MINIMUM_ROUTE_GAIN
        )
        clears_incumbent_margin = (
            target_latency_gain(incumbent_observation, measured_winner)
            >= MINIMUM_ROUTE_GAIN
        )
        if not (clears_default_margin and clears_incumbent_margin):
            if (
                incumbent_policy != DEFAULT_ROUTE_POLICY
                and target_latency_gain(
                    incumbent_observation,
                    eager_sdpa_observation,
                )
                >= MINIMUM_ROUTE_GAIN
            ):
                deployment = eager_sdpa_observation
            else:
                deployment = incumbent_observation

    policy = str(deployment["solution_policy"])
    return _upsert_exact_route(document, match, policy), deployment


def _summary_case_id(summary: Mapping[str, Any]) -> str:
    workload = summary.get("workload")
    shape = workload.get("shape") if isinstance(workload, Mapping) else None
    case_id = shape.get("case_id") if isinstance(shape, Mapping) else None
    if not isinstance(case_id, str) or not case_id:
        raise ContractError("tuning summary workload is missing case_id")
    return case_id


def _is_verified_route_destination(path: Path) -> bool:
    resolved = path.resolve()
    return (
        resolved.name == "routes.json"
        and any(part.lower() == "verified_hardware" for part in resolved.parts)
        and resolved.with_name("profile.json").is_file()
    )


def _json_payload(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _build_verified_bundle_manifest(
    project_root: Path,
    route_document: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    *,
    previous_manifest: Mapping[str, Any] | None,
    reset_previous: bool = False,
) -> dict[str, Any]:
    """Build the manifest before either member of the bundle is replaced."""

    workload_set_id, workload_sha256 = _summary_workload_identity(summaries)
    protocols = {
        json.dumps(
            summary.get("protocol"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for summary in summaries
    }
    if len(protocols) != 1:
        raise ContractError("formal summaries do not share one measurement protocol")
    protocol = summaries[0].get("protocol")
    if not isinstance(protocol, Mapping):
        raise ContractError("formal summary is missing its measurement protocol")
    formal_variant = _summary_variant(summaries)
    variant = formal_variant.as_dict()
    implementation_hash = solution_implementation_hash(
        project_root.resolve() / "solution"
    )
    try:
        current_official_hash = official_snapshot_hash(project_root)
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise ContractError(str(exc)) from exc
    summary_official_hashes = {
        summary.get("official_snapshot_sha256") for summary in summaries
    }
    if summary_official_hashes != {current_official_hash} or any(
        summary.get("official_consistent") is not True for summary in summaries
    ):
        raise ContractError(
            "formal summaries do not share the current official snapshot"
        )
    workload_set = load_workload_set(project_root, workload_set_id)
    all_shapes = all_benchmark_shapes(workload_set.shapes)
    covered_order = tuple(
        shape.case_id for shape in route_eligible_shapes(all_shapes, formal_variant)
    )
    covered_set = set(covered_order)
    provisional_order = tuple(
        shape.case_id
        for shape in all_shapes
        if not plan_workload_execution(shape, formal_variant).formal_eligible
    )
    excluded_order: tuple[str, ...] = ()
    selected_case_ids = {_summary_case_id(summary) for summary in summaries}
    unknown_case_ids = selected_case_ids - covered_set
    if unknown_case_ids:
        raise ContractError(
            "formal summaries include provisional or unknown cases: "
            + ", ".join(sorted(unknown_case_ids))
        )

    manifested_case_ids = set(selected_case_ids)
    if previous_manifest is not None and not reset_previous:
        try:
            previous = validate_bundle_manifest(previous_manifest)
        except (ContractError, TypeError, ValueError) as exc:
            raise ContractError(
                "verified bundle manifest is invalid; rerun a complete calibration"
            ) from exc
        if (
            previous.workload_set_id != workload_set_id
            or previous.workload_set_sha256 != workload_sha256
            or previous.official_snapshot_sha256 != current_official_hash
            or previous.solution_implementation_sha256 != implementation_hash
        ):
            raise ContractError(
                "verified bundle manifest is stale; rerun a complete calibration"
            )
        previous_formal = previous_manifest.get("formal")
        if not isinstance(previous_formal, Mapping):
            raise ContractError(
                "verified bundle manifest is invalid; rerun a complete calibration"
            )
        previous_protocol = previous_formal.get("protocol")
        previous_variant = previous_formal.get("variant")
        previous_covered = previous_formal.get("covered_case_ids")
        previous_provisional = previous_formal.get("provisional_case_ids")
        previous_excluded = previous_formal.get("excluded_case_ids")
        if (
            previous_protocol != protocol
            or previous_variant != variant
            or not isinstance(previous_covered, Sequence)
            or isinstance(previous_covered, (str, bytes))
            or not isinstance(previous_provisional, Sequence)
            or isinstance(previous_provisional, (str, bytes))
            or not isinstance(previous_excluded, Sequence)
            or isinstance(previous_excluded, (str, bytes))
            or set(previous_provisional) != set(provisional_order)
            or set(previous_excluded) != set(excluded_order)
            or not set(previous_covered).issubset(covered_set)
        ):
            raise ContractError(
                "verified bundle manifest is stale; rerun a complete calibration"
            )
        manifested_case_ids.update(str(case_id) for case_id in previous_covered)
    else:
        if selected_case_ids != covered_set:
            raise ContractError(
                "a verified bundle manifest requires one complete Formal workload "
                "calibration"
            )
    if manifested_case_ids != covered_set:
        raise ContractError(
            "verified bundle Formal coverage does not include every local shape"
        )

    route_digest = hashlib.sha256(_json_payload(route_document)).hexdigest()
    document: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "workload_set": {
            "set_id": workload_set_id,
            "sha256": workload_sha256,
        },
        "official": {"snapshot_sha256": current_official_hash},
        "solution": {"implementation_sha256": implementation_hash},
        "route_table": {"sha256": route_digest},
        "formal": {
            "protocol": copy.deepcopy(dict(protocol)),
            "variant": copy.deepcopy(variant),
            "covered_case_ids": [
                case_id for case_id in covered_order if case_id in manifested_case_ids
            ],
            "provisional_case_ids": list(provisional_order),
            "excluded_case_ids": list(excluded_order),
        },
    }
    table = validate_verified_route_table(route_document)
    expected_workload_routes = {
        workload_route_identity(shape, formal_variant)
        for shape in route_eligible_shapes(workload_set.shapes, formal_variant)
    }
    published_workload_routes = {
        tuple((field, match[field]) for field in WORKLOAD_ROUTE_FIELDS)
        for match, _policy in table.routes
    }
    if published_workload_routes != expected_workload_routes:
        raise ContractError(
            "verified routes do not exactly cover the Formal workload cases"
        )
    try:
        validate_bundle_manifest(document)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"invalid verified bundle manifest: {exc}") from exc
    return document


def _publish_bundle_tuning_summaries_locked(
    project_root: Path,
    summaries: Sequence[Mapping[str, Any]],
    *,
    route_path: Path,
    reset_verified_bundle: bool = False,
    verified_profile: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    """Atomically promote one or more formal summaries with shared-key guards."""

    if not summaries:
        raise ContractError("at least one tuning summary is required")
    destination = route_path
    if not _is_verified_route_destination(destination):
        raise ContractError(
            "route publication requires a complete verified_hardware bundle"
        )
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

    case_ids = [_summary_case_id(summary) for summary in summaries]
    workload_set_id, workload_sha256 = _summary_workload_identity(summaries)
    workload_set = load_workload_set(project_root, workload_set_id)
    if workload_set.sha256 != workload_sha256:
        raise ContractError("formal summary workload hash is stale; rerun calibration")
    formal_variant = _summary_variant(summaries)
    formal_shapes = route_eligible_shapes(workload_set.shapes, formal_variant)
    validate_promotion_case_set(
        case_ids,
        formal_shapes,
        formal_variant,
    )

    if reset_verified_bundle and not _is_verified_route_destination(destination):
        raise ContractError("only a verified hardware bundle can be reset")
    existing = (
        None
        if reset_verified_bundle
        else load_json(destination)
        if destination.is_file()
        else None
    )
    _validate_verified_package_identity(destination, existing, summaries)
    manifest_path = destination.with_name("manifest.json")
    previous_manifest_document: Mapping[str, Any] | None = None
    if (
        _is_verified_route_destination(destination)
        and manifest_path.is_file()
        and not reset_verified_bundle
    ):
        try:
            previous_manifest_document = load_json(manifest_path)
            manifest = validate_bundle_manifest(previous_manifest_document)
        except (ContractError, TypeError, ValueError) as exc:
            raise ContractError("verified bundle manifest is invalid") from exc
        if (
            manifest.workload_set_id != workload_set_id
            or manifest.workload_set_sha256 != workload_sha256
            or manifest.official_snapshot_sha256 != official_snapshot_hash(project_root)
            or manifest.solution_implementation_sha256 != current_implementation
        ):
            raise ContractError(
                "verified bundle manifest is stale; rerun a complete calibration"
            )
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
            "schema_version": SCHEMA_VERSION,
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
    if reset_verified_bundle or not destination.with_name("manifest.json").is_file():
        full_case_ids = {shape.case_id for shape in formal_shapes}
        if set(case_ids) != full_case_ids:
            raise ContractError(
                "a verified bundle manifest requires one complete Formal workload "
                "calibration"
            )
    manifest_document = _build_verified_bundle_manifest(
        project_root,
        document,
        summaries,
        previous_manifest=previous_manifest_document,
        reset_previous=reset_verified_bundle,
    )
    _publish_verified_bundle(
        destination,
        document,
        manifest_document,
        profile_document=verified_profile,
    )
    return document, winners, destination


def _publish_bundle_tuning_summaries(
    project_root: Path,
    summaries: Sequence[Mapping[str, Any]],
    *,
    route_path: Path,
    reset_verified_bundle: bool = False,
    verified_profile: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    """Publish one complete Bundle update under a cross-process lock."""

    destination = route_path.resolve()
    if not _is_verified_route_destination(destination):
        raise ContractError(
            "route publication requires a complete verified_hardware bundle"
        )
    with exclusive_file_lock(
        bundle_lock_path(destination.parent),
        purpose=f"route publication for {destination.parent.name}",
    ):
        return _publish_bundle_tuning_summaries_locked(
            project_root,
            summaries,
            route_path=destination,
            reset_verified_bundle=reset_verified_bundle,
            verified_profile=verified_profile,
        )


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _publish_verified_bundle(
    route_path: Path,
    route_document: Mapping[str, Any],
    manifest_document: Mapping[str, Any],
    *,
    profile_document: Mapping[str, Any] | None = None,
) -> None:
    """Replace routes and manifest together, restoring both after any failure."""

    try:
        validate_verified_route_table(route_document)
        validate_bundle_manifest(manifest_document)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"invalid verified bundle publication: {exc}") from exc
    manifest_path = route_path.with_name("manifest.json")
    profile_path = route_path.with_name("profile.json")
    route_payload = _json_payload(route_document)
    manifest_payload = _json_payload(manifest_document)
    if (
        manifest_document["route_table"]["sha256"]
        != hashlib.sha256(route_payload).hexdigest()
    ):
        raise ContractError("verified bundle manifest does not bind the route payload")

    publications = {
        route_path: route_payload,
        manifest_path: manifest_payload,
    }
    if profile_document is not None:
        _verified_bundle_identity(profile_document)
        publications[profile_path] = _json_payload(profile_document)
    previous = {
        route_path: route_path.read_bytes() if route_path.is_file() else None,
        manifest_path: manifest_path.read_bytes() if manifest_path.is_file() else None,
    }
    if profile_document is not None:
        previous[profile_path] = (
            profile_path.read_bytes() if profile_path.is_file() else None
        )
    try:
        for path, payload in publications.items():
            _atomic_replace_bytes(path, payload)
    except BaseException as publication_error:
        recovery_errors: list[BaseException] = []
        for path, payload in previous.items():
            try:
                if payload is None:
                    path.unlink(missing_ok=True)
                else:
                    _atomic_replace_bytes(path, payload)
            except BaseException as recovery_error:  # noqa: BLE001
                recovery_errors.append(recovery_error)
        if recovery_errors:
            raise ContractError(
                "verified bundle publication failed and rollback was incomplete"
            ) from publication_error
        raise


def _atomic_replace_json(
    path: Path,
    document: Mapping[str, Any],
    *,
    validate_as_route_table: bool,
) -> None:
    """Validate and atomically replace one standalone JSON document."""

    payload = _json_payload(document)
    loaded = json.loads(payload)
    if validate_as_route_table:
        validate_route_table(loaded)
    _atomic_replace_bytes(path, payload)


__all__ = [
    "DEFAULT_ROUTE_POLICY",
    "MINIMUM_ROUTE_GAIN",
    "auto_promote_calibration",
    "build_promoted_route_document",
    "find_matching_verified_route",
    "validate_promotion_case_set",
    "verified_profile_from_probe_result",
]
