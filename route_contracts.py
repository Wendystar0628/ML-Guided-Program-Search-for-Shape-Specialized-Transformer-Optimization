"""Pure contracts for exact runtime routes and verified route bundles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from policy_registry import ROUTABLE_POLICY_IDS
from project_identity import (
    canonical_json_sha256,
    official_snapshot_hash,
    solution_implementation_hash,
)

SCHEMA_VERSION = 6
MANIFEST_SCHEMA_VERSION = 5
ALLOWED_POLICIES = ROUTABLE_POLICY_IDS

HARDWARE_ROUTE_FIELDS = frozenset(
    {
        "device_type",
        "device_name",
        "compute_capability",
        "platform_system",
        "torch",
        "cuda_runtime",
        "driver",
        "matmul_precision",
        "allow_tf32",
    }
)
WORKLOAD_ROUTE_FIELDS = (
    "dtype",
    "B",
    "S",
    "D",
    "heads",
    "ffn",
    "layers",
    "causal",
)
ROUTE_FIELDS = HARDWARE_ROUTE_FIELDS | frozenset(WORKLOAD_ROUTE_FIELDS)

_STRING_FIELDS = frozenset(
    {
        "device_type",
        "device_name",
        "compute_capability",
        "platform_system",
        "torch",
        "cuda_runtime",
        "driver",
        "matmul_precision",
        "dtype",
    }
)
_INTEGER_FIELDS = frozenset({"B", "S", "D", "heads", "ffn", "layers"})
_BOOLEAN_FIELDS = frozenset({"causal", "allow_tf32"})
_TABLE_FIELDS = frozenset({"schema_version", "default_policy", "routes"})
_ROUTE_ENTRY_FIELDS = frozenset({"match", "policy"})
_MANIFEST_FIELDS = frozenset(
    {"schema_version", "workload_set", "official", "solution", "route_table", "formal"}
)
_FORMAL_FIELDS = frozenset(
    {
        "protocol",
        "variant",
        "covered_case_ids",
        "provisional_case_ids",
        "excluded_case_ids",
    }
)
_VARIANT_FIELDS = frozenset({"dtype", "padding_ratio", "input_scale"})


@dataclass(frozen=True, slots=True)
class RouteTable:
    """Validated exact route table."""

    default_policy: str
    routes: tuple[tuple[dict[str, object], str], ...]


@dataclass(frozen=True, slots=True)
class RouteResolution:
    """One deterministic policy decision and its provenance."""

    policy: str
    origin: Literal["calibrated", "fallback"]
    source: str | None = None
    table_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class VerifiedBundleManifest:
    """Minimal provenance required before verified routes are trusted."""

    workload_set_id: str
    workload_set_sha256: str
    official_snapshot_sha256: str
    solution_implementation_sha256: str
    route_table_sha256: str
    formal_protocol: dict[str, object]
    formal_variant: dict[str, object]
    covered_case_ids: tuple[str, ...]
    provisional_case_ids: tuple[str, ...]
    excluded_case_ids: tuple[str, ...]


def _require_policy(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    policy = value.strip()
    if policy not in ALLOWED_POLICIES:
        choices = ", ".join(sorted(ALLOWED_POLICIES))
        raise ValueError(f"{field} must be one of: {choices}")
    return policy


def _validate_match(match: object, *, index: int) -> dict[str, object]:
    if not isinstance(match, dict) or not match:
        raise ValueError(f"routes[{index}].match must be a non-empty object")
    unknown = set(match) - ROUTE_FIELDS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"routes[{index}].match has unknown fields: {names}")

    validated: dict[str, object] = {}
    for name, value in match.items():
        if name in _STRING_FIELDS:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"routes[{index}].match.{name} must be a non-empty string"
                )
            normalized = value.strip()
            if name == "matmul_precision" and normalized not in {
                "highest",
                "high",
                "medium",
            }:
                raise ValueError(
                    f"routes[{index}].match.matmul_precision is unsupported"
                )
            validated[name] = normalized
        elif name in _INTEGER_FIELDS:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"routes[{index}].match.{name} must be a positive integer"
                )
            validated[name] = value
        elif name in _BOOLEAN_FIELDS:
            if not isinstance(value, bool):
                raise ValueError(f"routes[{index}].match.{name} must be boolean")
            validated[name] = value
    return validated


def validate_route_table(payload: object) -> RouteTable:
    """Validate the current exact-route table without legacy compatibility."""

    if not isinstance(payload, dict):
        raise TypeError("route table must be a JSON object")
    missing = _TABLE_FIELDS - set(payload)
    unknown = set(payload) - _TABLE_FIELDS
    if missing:
        raise ValueError(f"route table is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(
            f"route table has unknown fields: {', '.join(sorted(unknown))}"
        )
    if (
        isinstance(payload["schema_version"], bool)
        or not isinstance(payload["schema_version"], int)
        or payload["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")

    default_policy = _require_policy(payload["default_policy"], field="default_policy")
    raw_routes = payload["routes"]
    if not isinstance(raw_routes, list):
        raise TypeError("routes must be a list")

    routes: list[tuple[dict[str, object], str]] = []
    seen_matches: set[tuple[tuple[str, object], ...]] = set()
    for index, entry in enumerate(raw_routes):
        if not isinstance(entry, dict):
            raise TypeError(f"routes[{index}] must be an object")
        missing_entry = _ROUTE_ENTRY_FIELDS - set(entry)
        unknown_entry = set(entry) - _ROUTE_ENTRY_FIELDS
        if missing_entry:
            raise ValueError(
                f"routes[{index}] is missing fields: "
                + ", ".join(sorted(missing_entry))
            )
        if unknown_entry:
            raise ValueError(
                f"routes[{index}] has unknown fields: "
                + ", ".join(sorted(unknown_entry))
            )

        match = _validate_match(entry["match"], index=index)
        missing_match_fields = ROUTE_FIELDS - set(match)
        if missing_match_fields:
            names = ", ".join(sorted(missing_match_fields))
            raise ValueError(
                f"routes[{index}].match must be exact; missing fields: {names}"
            )
        policy = _require_policy(entry["policy"], field=f"routes[{index}].policy")
        fingerprint = tuple(sorted(match.items()))
        if fingerprint in seen_matches:
            raise ValueError(f"routes[{index}].match duplicates an earlier route")
        seen_matches.add(fingerprint)
        routes.append((match, policy))
    return RouteTable(default_policy=default_policy, routes=tuple(routes))


def validate_verified_route_table(
    payload: object,
    *,
    expected_identity: Mapping[str, object] | None = None,
) -> RouteTable:
    """Validate routes published inside a verified hardware bundle."""

    table = validate_route_table(payload)
    if table.default_policy != "eager-sdpa":
        raise ValueError("verified route table must use default_policy=eager-sdpa")
    if expected_identity is not None and (
        not expected_identity or not set(expected_identity) <= HARDWARE_ROUTE_FIELDS
    ):
        raise ValueError("verified hardware identity contains unknown route fields")
    for index, (match, _policy) in enumerate(table.routes):
        if expected_identity is None:
            continue
        mismatches = [
            field
            for field, expected in expected_identity.items()
            if match.get(field) != expected
        ]
        if mismatches:
            raise ValueError(
                f"verified routes[{index}] has a mismatched runtime identity: "
                + ", ".join(sorted(mismatches))
            )
    return table


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value.lower()


def _require_section(
    payload: Mapping[str, object],
    name: str,
    fields: frozenset[str],
) -> Mapping[str, object]:
    section = payload.get(name)
    if not isinstance(section, Mapping):
        raise TypeError(f"verified bundle manifest {name} must be an object")
    if set(section) != fields:
        missing = ", ".join(sorted(fields - set(section)))
        extra = ", ".join(sorted(set(section) - fields))
        raise ValueError(f"invalid {name} fields; missing={missing}; extra={extra}")
    return section


def _case_ids(value: object, *, field: str, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    normalized: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field}[{index}] must be a non-empty string")
        normalized.append(item.strip())
    if not allow_empty and not normalized:
        raise ValueError(f"{field} must not be empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} must not contain duplicates")
    return tuple(normalized)


def _validated_variant(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("formal.variant must be an object")
    if set(value) != _VARIANT_FIELDS:
        missing = ", ".join(sorted(_VARIANT_FIELDS - set(value)))
        extra = ", ".join(sorted(set(value) - _VARIANT_FIELDS))
        raise ValueError(
            f"invalid formal.variant fields; missing={missing}; extra={extra}"
        )
    dtype = value.get("dtype")
    padding_ratio = value.get("padding_ratio")
    input_scale = value.get("input_scale")
    if not isinstance(dtype, str) or not dtype.strip():
        raise ValueError("formal.variant.dtype must be a non-empty string")
    if (
        isinstance(padding_ratio, bool)
        or not isinstance(padding_ratio, (int, float))
        or not 0 <= float(padding_ratio) < 1
    ):
        raise ValueError("formal.variant.padding_ratio must be in [0, 1)")
    if (
        isinstance(input_scale, bool)
        or not isinstance(input_scale, (int, float))
        or float(input_scale) <= 0
    ):
        raise ValueError("formal.variant.input_scale must be positive")
    return {
        "dtype": dtype.strip(),
        "padding_ratio": float(padding_ratio),
        "input_scale": float(input_scale),
    }


def validate_bundle_manifest(payload: object) -> VerifiedBundleManifest:
    """Validate the strict v5 verified-bundle identity document."""

    if not isinstance(payload, dict):
        raise TypeError("verified bundle manifest must be a JSON object")
    if set(payload) != _MANIFEST_FIELDS:
        missing = ", ".join(sorted(_MANIFEST_FIELDS - set(payload)))
        extra = ", ".join(sorted(set(payload) - _MANIFEST_FIELDS))
        raise ValueError(
            f"invalid verified bundle manifest fields; missing={missing}; extra={extra}"
        )
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"verified bundle manifest schema_version must be {MANIFEST_SCHEMA_VERSION}"
        )

    workload = _require_section(
        payload, "workload_set", frozenset({"set_id", "sha256"})
    )
    official = _require_section(payload, "official", frozenset({"snapshot_sha256"}))
    solution = _require_section(
        payload, "solution", frozenset({"implementation_sha256"})
    )
    route_table = _require_section(payload, "route_table", frozenset({"sha256"}))
    formal = _require_section(payload, "formal", _FORMAL_FIELDS)

    set_id = workload.get("set_id")
    if not isinstance(set_id, str) or not set_id.strip() or Path(set_id).name != set_id:
        raise ValueError("workload_set.set_id must be a safe non-empty identifier")
    protocol = formal.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("preset") != "formal":
        raise ValueError("formal.protocol must describe a Formal measurement")
    variant = _validated_variant(formal.get("variant"))
    covered = _case_ids(
        formal.get("covered_case_ids"),
        field="formal.covered_case_ids",
        allow_empty=False,
    )
    provisional = _case_ids(
        formal.get("provisional_case_ids"),
        field="formal.provisional_case_ids",
        allow_empty=True,
    )
    excluded = _case_ids(
        formal.get("excluded_case_ids"),
        field="formal.excluded_case_ids",
        allow_empty=True,
    )
    covered_set = set(covered)
    provisional_set = set(provisional)
    excluded_set = set(excluded)
    overlap = (
        (covered_set & provisional_set)
        | (covered_set & excluded_set)
        | (provisional_set & excluded_set)
    )
    if overlap:
        raise ValueError(
            "formal covered, provisional, and excluded case ids overlap: "
            + ", ".join(sorted(overlap))
        )

    return VerifiedBundleManifest(
        workload_set_id=set_id.strip(),
        workload_set_sha256=_require_sha256(
            workload.get("sha256"), field="workload_set.sha256"
        ),
        official_snapshot_sha256=_require_sha256(
            official.get("snapshot_sha256"), field="official.snapshot_sha256"
        ),
        solution_implementation_sha256=_require_sha256(
            solution.get("implementation_sha256"),
            field="solution.implementation_sha256",
        ),
        route_table_sha256=_require_sha256(
            route_table.get("sha256"), field="route_table.sha256"
        ),
        formal_protocol=dict(protocol),
        formal_variant=variant,
        covered_case_ids=covered,
        provisional_case_ids=provisional,
        excluded_case_ids=excluded,
    )


def _config_value(config: object, name: str) -> object:
    if isinstance(config, Mapping):
        try:
            return config[name]
        except KeyError as exc:
            raise ValueError(f"config is missing {name}") from exc
    try:
        return getattr(config, name)
    except AttributeError as exc:
        raise ValueError(f"config is missing {name}") from exc


def _normalize_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    dimensions = tuple(shape)
    if len(dimensions) != 3:
        raise ValueError("shape must be [B, S, D]")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in dimensions
    ):
        raise ValueError("shape values must be positive integers")
    return dimensions


def _normalize_dtype(dtype: object | None) -> str | None:
    if dtype is None:
        return None
    name = str(dtype).strip()
    if name.startswith("torch."):
        name = name.removeprefix("torch.")
    if not name:
        raise ValueError("dtype must be non-empty")
    return name


def make_workload_route_key(
    config: object,
    *,
    shape: Sequence[int] | None = None,
    dtype: object | None = None,
) -> dict[str, object]:
    """Build the workload-only portion of an exact route key."""

    if shape is None:
        shape = (
            _config_value(config, "batch_size"),
            _config_value(config, "seq_len"),
            _config_value(config, "d_model"),
        )
    batch, sequence, model_dim = _normalize_shape(shape)
    raw_dtype = dtype
    if raw_dtype is None:
        try:
            raw_dtype = _config_value(config, "dtype")
        except ValueError:
            raw_dtype = None
    normalized_dtype = _normalize_dtype(raw_dtype)
    key: dict[str, object] = {
        "B": batch,
        "S": sequence,
        "D": model_dim,
        "heads": _config_value(config, "num_heads"),
        "ffn": _config_value(config, "ffn_dim"),
        "layers": _config_value(config, "num_layers"),
        "causal": _config_value(config, "causal"),
    }
    if normalized_dtype is not None:
        key["dtype"] = normalized_dtype
    _validate_match(key, index=0)
    return key


def make_route_key(
    config: object,
    *,
    device_type: str,
    device_name: str,
    compute_capability: str,
    platform_system: str,
    torch_version: str,
    cuda_runtime: str,
    driver: str,
    matmul_precision: str,
    allow_tf32: bool,
    shape: Sequence[int] | None = None,
    dtype: object | None = None,
) -> dict[str, object]:
    """Build an exact route key entirely from explicit runtime facts."""

    key: dict[str, object] = {
        "device_type": device_type,
        "device_name": device_name,
        "compute_capability": compute_capability,
        "platform_system": platform_system,
        "torch": torch_version,
        "cuda_runtime": cuda_runtime,
        "driver": driver,
        "matmul_precision": matmul_precision,
        "allow_tf32": allow_tf32,
        **make_workload_route_key(config, shape=shape, dtype=dtype),
    }
    validated = _validate_match(key, index=0)
    missing = ROUTE_FIELDS - set(validated)
    if missing:
        raise ValueError("route key is missing fields: " + ", ".join(sorted(missing)))
    return validated


def resolve_route(table: RouteTable, key: Mapping[str, object]) -> str:
    """Return the first exact match or the table default."""

    return resolve_route_result(table, key).policy


def resolve_route_result(
    table: RouteTable,
    key: Mapping[str, object],
) -> RouteResolution:
    """Resolve a route and report calibrated-table versus fallback origin."""

    for match, policy in table.routes:
        if all(key.get(field) == expected for field, expected in match.items()):
            return RouteResolution(policy=policy, origin="calibrated")
    return RouteResolution(policy=table.default_policy, origin="fallback")


def load_route_table_with_digest(path: str | Path) -> tuple[RouteTable, str]:
    """Read, hash, decode, and validate one route table from identical bytes."""

    route_path = Path(path).resolve()
    try:
        content = route_path.read_bytes()
        payload = json.loads(content)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load route table {route_path}: {exc}") from exc
    return validate_route_table(payload), hashlib.sha256(content).hexdigest()


def load_route_table(path: str | Path) -> RouteTable:
    """Read and validate one exact route table."""

    table, _digest = load_route_table_with_digest(path)
    return table


def _workload_shape_key(
    shape: Mapping[str, object], *, dtype: object
) -> tuple[object, ...]:
    names = {
        "B": "batch_size",
        "S": "seq_len",
        "D": "qkv_dim",
        "heads": "heads",
        "ffn": "ffn_dim",
        "layers": "layers",
        "causal": "causal",
    }
    key: dict[str, object] = {"dtype": _normalize_dtype(dtype)}
    for target, source in names.items():
        if source not in shape:
            raise ValueError(f"verified workload shape is missing {source}")
        key[target] = shape[source]
    return tuple(key[field] for field in WORKLOAD_ROUTE_FIELDS)


def _route_workload_key(match: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(match[field] for field in WORKLOAD_ROUTE_FIELDS)


def _validate_manifest_coverage(
    manifest: VerifiedBundleManifest,
    *,
    table: RouteTable,
    workload_document: Mapping[str, object],
) -> None:
    raw_shapes = workload_document.get("ordered_shapes")
    if not isinstance(raw_shapes, Sequence) or isinstance(raw_shapes, (str, bytes)):
        raise TypeError("verified bundle workload set has no ordered shapes")

    shapes_by_id: dict[str, Mapping[str, object]] = {}
    for shape in raw_shapes:
        if not isinstance(shape, Mapping):
            raise TypeError("verified bundle workload set contains an invalid shape")
        case_id = shape.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("verified bundle workload set contains an invalid case id")
        if case_id in shapes_by_id:
            raise ValueError("verified bundle workload set contains duplicate case ids")
        shapes_by_id[case_id] = shape

    covered = set(manifest.covered_case_ids)
    provisional = set(manifest.provisional_case_ids)
    excluded = set(manifest.excluded_case_ids)
    expected = set(shapes_by_id)
    partition = covered | provisional | excluded
    if (covered & provisional) or (covered & excluded) or (provisional & excluded):
        raise ValueError("verified bundle case partitions overlap")
    if partition != expected:
        missing = ", ".join(sorted(expected - partition))
        unknown = ", ".join(sorted(partition - expected))
        raise ValueError(
            "verified bundle case partition is incomplete; "
            f"missing={missing}; unknown={unknown}"
        )

    dtype = manifest.formal_variant["dtype"]
    precision = manifest.formal_protocol.get("matmul_precision")
    allow_tf32 = manifest.formal_protocol.get("allow_tf32")
    if precision not in {"highest", "high", "medium"}:
        raise ValueError("verified Formal protocol has invalid matmul_precision")
    if not isinstance(allow_tf32, bool):
        raise TypeError("verified Formal protocol allow_tf32 must be boolean")
    if manifest.formal_protocol.get("compile_solution") is not False:
        raise ValueError("verified routes require an uncompiled Formal solution")
    covered_keys = {
        _workload_shape_key(shapes_by_id[case_id], dtype=dtype) for case_id in covered
    }
    route_keys = {_route_workload_key(match) for match, _policy in table.routes}
    missing_keys = covered_keys - route_keys
    extra_keys = route_keys - covered_keys
    if missing_keys:
        raise ValueError("verified routes do not cover every Formal workload key")
    if extra_keys:
        raise ValueError(
            "verified routes contain workload keys outside Formal coverage"
        )
    if any(
        match["matmul_precision"] != precision or match["allow_tf32"] is not allow_tf32
        for match, _policy in table.routes
    ):
        raise ValueError("verified routes do not match the Formal runtime policy")


def load_verified_bundle(
    route_path: str | Path,
    *,
    project_root: str | Path,
) -> tuple[RouteTable, str, VerifiedBundleManifest]:
    """Load one verified bundle and reject stale or incomplete provenance."""

    resolved_route = Path(route_path).resolve()
    resolved_project = Path(project_root).resolve()
    table, route_digest = load_route_table_with_digest(resolved_route)
    validate_verified_route_table(
        {
            "schema_version": SCHEMA_VERSION,
            "default_policy": table.default_policy,
            "routes": [
                {"match": match, "policy": policy} for match, policy in table.routes
            ],
        }
    )

    manifest_path = resolved_route.with_name("manifest.json")
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"unable to load bundle manifest {manifest_path}: {exc}"
        ) from exc
    manifest = validate_bundle_manifest(manifest_payload)
    if manifest.route_table_sha256 != route_digest:
        raise ValueError("verified bundle route-table hash is stale")

    current_implementation = solution_implementation_hash(resolved_project / "solution")
    if manifest.solution_implementation_sha256 != current_implementation:
        raise ValueError("verified bundle Solution implementation hash is stale")
    try:
        current_official = official_snapshot_hash(resolved_project)
    except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to verify official snapshot: {exc}") from exc
    if manifest.official_snapshot_sha256 != current_official:
        raise ValueError("verified bundle official snapshot hash is stale")

    workload_path = resolved_project / "official" / "test_shapes.json"
    if not workload_path.is_file():
        raise ValueError("verified bundle workload set is unavailable")
    if manifest.workload_set_sha256 != canonical_json_sha256(workload_path):
        raise ValueError("verified bundle workload-set hash is stale")
    try:
        workload_document = json.loads(workload_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load verified workload set: {exc}") from exc
    if not isinstance(workload_document, Mapping):
        raise TypeError("verified bundle workload set must be a JSON object")
    if workload_document.get("workload_set_id") != manifest.workload_set_id:
        raise ValueError("verified bundle workload-set id is stale")
    _validate_manifest_coverage(
        manifest,
        table=table,
        workload_document=workload_document,
    )
    return table, route_digest, manifest


__all__ = [
    "ALLOWED_POLICIES",
    "HARDWARE_ROUTE_FIELDS",
    "MANIFEST_SCHEMA_VERSION",
    "ROUTE_FIELDS",
    "SCHEMA_VERSION",
    "WORKLOAD_ROUTE_FIELDS",
    "RouteResolution",
    "RouteTable",
    "VerifiedBundleManifest",
    "load_route_table",
    "load_route_table_with_digest",
    "load_verified_bundle",
    "make_route_key",
    "make_workload_route_key",
    "resolve_route",
    "resolve_route_result",
    "validate_bundle_manifest",
    "validate_route_table",
    "validate_verified_route_table",
]
