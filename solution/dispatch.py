"""Deterministic policy routing from an offline-calibrated route table."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from .policies import ROUTABLE_POLICY_IDS

SCHEMA_VERSION = 2
MANIFEST_SCHEMA_VERSION = 1
ALLOWED_POLICIES = ROUTABLE_POLICY_IDS
HARDWARE_ROUTE_FIELDS = frozenset(
    (
        "device_type",
        "device_name",
        "compute_capability",
        "platform_system",
        "torch",
        "cuda_runtime",
        "triton",
    )
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
        "triton",
        "dtype",
    }
)
_INTEGER_FIELDS = frozenset({"B", "S", "D", "heads", "ffn", "layers"})
_TABLE_FIELDS = frozenset({"schema_version", "default_policy", "routes"})
_ROUTE_ENTRY_FIELDS = frozenset({"match", "policy"})


@dataclass(frozen=True)
class RouteTable:
    """Validated route table used by the deterministic resolver."""

    default_policy: str
    routes: tuple[tuple[dict[str, object], str], ...]


@dataclass(frozen=True)
class RouteResolution:
    """One deterministic policy decision and where it came from."""

    policy: str
    origin: Literal["calibrated", "fallback"]
    source: str | None = None
    table_sha256: str | None = None


@dataclass(frozen=True)
class FormalSummaryRef:
    """Compact reference to the Formal runs behind verified routes."""

    summary_id: str
    case_ids: tuple[str, ...]


@dataclass(frozen=True)
class VerifiedBundleManifest:
    """Minimal immutable provenance required before catalog routes are trusted."""

    workload_set_id: str
    workload_set_sha256: str
    solution_implementation_sha256: str
    route_table_sha256: str
    formal_protocol: dict[str, object]
    source_summaries: tuple[FormalSummaryRef, ...]


def _static_runtime_facts() -> dict[str, str]:
    """Return process-static software facts used by schema-version-2 routes."""

    facts = {
        "platform_system": platform.system(),
        "torch": str(torch.__version__),
        "triton": "unavailable",
    }
    if torch.version.cuda is not None:
        facts["cuda_runtime"] = str(torch.version.cuda)
    try:
        import triton
    except Exception:  # noqa: BLE001 - Triton is an optional route capability.
        return facts
    else:
        facts["triton"] = str(getattr(triton, "__version__", "unknown"))
    return facts


_STATIC_RUNTIME_FACTS = _static_runtime_facts()


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
            validated[name] = value.strip()
        elif name in _INTEGER_FIELDS:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"routes[{index}].match.{name} must be a positive integer"
                )
            validated[name] = value
        elif name == "causal":
            if not isinstance(value, bool):
                raise ValueError(f"routes[{index}].match.causal must be boolean")
            validated[name] = value
    return validated


def validate_route_table(payload: object) -> RouteTable:
    """Validate a schema-version-2 decoded route table."""

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
        not isinstance(payload["schema_version"], int)
        or isinstance(payload["schema_version"], bool)
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
            names = ", ".join(sorted(missing_entry))
            raise ValueError(f"routes[{index}] is missing fields: {names}")
        if unknown_entry:
            names = ", ".join(sorted(unknown_entry))
            raise ValueError(f"routes[{index}] has unknown fields: {names}")

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
    """Validate the stricter exact-route contract used by verified bundles.

    The base schema already requires exact keys. This stricter entry point also
    checks that every route belongs to the package's declared hardware identity.
    """

    table = validate_route_table(payload)
    if table.default_policy != "auto":
        raise ValueError("verified route table must use default_policy=auto")
    if (
        expected_identity is not None
        and set(expected_identity) != HARDWARE_ROUTE_FIELDS
    ):
        raise ValueError("verified hardware identity does not match route schema")
    for index, (match, _policy) in enumerate(table.routes):
        if expected_identity is not None:
            mismatches = [
                field
                for field in HARDWARE_ROUTE_FIELDS
                if match.get(field) != expected_identity[field]
            ]
            if mismatches:
                raise ValueError(
                    f"verified routes[{index}] has a mismatched hardware identity: "
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


def _canonical_value_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_bundle_manifest(payload: object) -> VerifiedBundleManifest:
    """Validate the compact identity document beside a verified route table."""

    if not isinstance(payload, dict):
        raise TypeError("verified bundle manifest must be a JSON object")
    required = {
        "schema_version",
        "workload_set",
        "solution",
        "route_table",
        "formal",
    }
    if set(payload) != required:
        missing = ", ".join(sorted(required - set(payload)))
        extra = ", ".join(sorted(set(payload) - required))
        raise ValueError(
            f"invalid verified bundle manifest fields; missing={missing}; extra={extra}"
        )
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"verified bundle manifest schema_version must be {MANIFEST_SCHEMA_VERSION}"
        )

    workload = payload.get("workload_set")
    solution = payload.get("solution")
    route_table = payload.get("route_table")
    formal = payload.get("formal")
    if not all(
        isinstance(value, Mapping)
        for value in (workload, solution, route_table, formal)
    ):
        raise TypeError("verified bundle manifest sections must be objects")
    assert isinstance(workload, Mapping)
    assert isinstance(solution, Mapping)
    assert isinstance(route_table, Mapping)
    assert isinstance(formal, Mapping)

    set_id = workload.get("set_id")
    if not isinstance(set_id, str) or not set_id.strip() or Path(set_id).name != set_id:
        raise ValueError("workload_set.set_id must be a safe non-empty identifier")
    if set(workload) != {"set_id", "sha256"}:
        raise ValueError("workload_set must contain set_id and sha256")
    if set(solution) != {"implementation_sha256"}:
        raise ValueError("solution must contain implementation_sha256")
    if set(route_table) != {"sha256"}:
        raise ValueError("route_table must contain sha256")
    if set(formal) != {"protocol", "source_summaries"}:
        raise ValueError("formal must contain protocol and source_summaries")

    protocol = formal.get("protocol")
    sources = formal.get("source_summaries")
    if not isinstance(protocol, Mapping) or protocol.get("preset") != "formal":
        raise ValueError("formal.protocol must describe a Formal measurement")
    if (
        not isinstance(sources, Sequence)
        or isinstance(sources, (str, bytes))
        or not sources
    ):
        raise ValueError("formal.source_summaries must be a non-empty sequence")
    workload_sha256 = _require_sha256(
        workload.get("sha256"), field="workload_set.sha256"
    )
    solution_sha256 = _require_sha256(
        solution.get("implementation_sha256"),
        field="solution.implementation_sha256",
    )
    normalized_sources: list[FormalSummaryRef] = []
    seen_summary_ids: set[str] = set()
    seen_case_ids: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise TypeError(f"formal.source_summaries[{index}] must be an object")
        if set(source) != {"summary_id", "case_ids"}:
            raise ValueError(
                f"invalid formal.source_summaries[{index}] fields; "
                "expected summary_id and case_ids"
            )
        summary_id = source.get("summary_id")
        case_ids = source.get("case_ids")
        if not isinstance(summary_id, str) or not summary_id.strip():
            raise ValueError(
                f"formal.source_summaries[{index}].summary_id must be non-empty"
            )
        if (
            not isinstance(case_ids, Sequence)
            or isinstance(case_ids, (str, bytes))
            or not case_ids
            or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
        ):
            raise ValueError(
                f"formal.source_summaries[{index}].case_ids must be non-empty strings"
            )
        if len(case_ids) != len(set(case_ids)):
            raise ValueError(
                f"formal.source_summaries[{index}].case_ids contains duplicates"
            )
        normalized_summary_id = summary_id.strip()
        normalized_case_ids = tuple(str(case_id) for case_id in case_ids)
        if normalized_summary_id in seen_summary_ids:
            raise ValueError(
                f"formal.source_summaries[{index}].summary_id is duplicated"
            )
        overlap = seen_case_ids.intersection(normalized_case_ids)
        if overlap:
            raise ValueError(
                "formal.source_summaries assign a case more than once: "
                + ", ".join(sorted(overlap))
            )
        seen_summary_ids.add(normalized_summary_id)
        seen_case_ids.update(normalized_case_ids)
        normalized_sources.append(
            FormalSummaryRef(normalized_summary_id, normalized_case_ids)
        )

    return VerifiedBundleManifest(
        workload_set_id=set_id.strip(),
        workload_set_sha256=workload_sha256,
        solution_implementation_sha256=solution_sha256,
        route_table_sha256=_require_sha256(
            route_table.get("sha256"), field="route_table.sha256"
        ),
        formal_protocol=dict(protocol),
        source_summaries=tuple(normalized_sources),
    )


def _canonical_json_sha256(path: Path) -> str:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to hash JSON document {path}: {exc}") from exc
    return _canonical_value_sha256(document)


def _validate_manifest_case_coverage(
    manifest: VerifiedBundleManifest,
    *,
    workload_document: Mapping[str, object],
) -> None:
    raw_cases = workload_document.get("ordered_cases")
    if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)):
        raise TypeError("verified bundle workload set has no ordered cases")
    expected_case_ids: set[str] = set()
    for case in raw_cases:
        case_id = case.get("case_id") if isinstance(case, Mapping) else None
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("verified bundle workload set contains an invalid case")
        expected_case_ids.add(case_id)
    manifested_case_ids = {
        case_id for source in manifest.source_summaries for case_id in source.case_ids
    }
    if manifested_case_ids != expected_case_ids:
        raise ValueError("verified bundle Formal sources do not cover the workload set")


def _solution_implementation_sha256(solution_root: Path) -> str:
    suffixes = {".py", ".cpp", ".cc", ".c", ".h", ".cu", ".cuh"}
    files = sorted(
        path
        for path in solution_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in suffixes
        and "__pycache__" not in path.parts
    )
    if not files:
        raise ValueError(f"no Solution source files found under {solution_root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(solution_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def load_verified_bundle(
    route_path: str | Path,
    *,
    project_root: str | Path,
) -> tuple[RouteTable, str, VerifiedBundleManifest]:
    """Load one verified bundle and reject stale provenance as one unit."""

    resolved_route = Path(route_path).resolve()
    resolved_project = Path(project_root).resolve()
    table, route_digest = _load_route_table_bytes(resolved_route)
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
    current_implementation = _solution_implementation_sha256(
        resolved_project / "solution"
    )
    if manifest.solution_implementation_sha256 != current_implementation:
        raise ValueError("verified bundle Solution implementation hash is stale")
    workload_path = (
        resolved_project / "runner" / "workloads" / f"{manifest.workload_set_id}.json"
    )
    if not workload_path.is_file():
        raise ValueError("verified bundle workload set is unavailable")
    if manifest.workload_set_sha256 != _canonical_json_sha256(workload_path):
        raise ValueError("verified bundle workload-set hash is stale")
    try:
        workload_document = json.loads(workload_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load verified workload set: {exc}") from exc
    if not isinstance(workload_document, Mapping):
        raise TypeError("verified bundle workload set must be a JSON object")
    _validate_manifest_case_coverage(
        manifest,
        workload_document=workload_document,
    )
    return table, route_digest, manifest


def _load_route_table_bytes(path: str | Path) -> tuple[RouteTable, str]:
    """Read, hash, decode, and validate one route table from identical bytes."""

    route_path = Path(path).resolve()
    try:
        content = route_path.read_bytes()
        payload = json.loads(content)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load route table {route_path}: {exc}") from exc
    return validate_route_table(payload), hashlib.sha256(content).hexdigest()


def load_route_table(path: str | Path) -> RouteTable:
    """Read and validate a route table from disk."""

    table, _ = _load_route_table_bytes(path)
    return table


def _route_source_label(path: Path, project_root: Path) -> str:
    """Return a portable repository path for internal tables."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _catalog_paths(catalog_root: Path) -> tuple[Path, ...]:
    """Discover deterministic device-package route tables."""

    if not catalog_root.is_dir():
        return ()
    return tuple(sorted(catalog_root.glob("*/routes.json")))


def _merge_catalog_tables(
    paths: Sequence[Path],
    *,
    project_root: Path,
) -> tuple[
    RouteTable,
    tuple[str, ...],
    tuple[tuple[Path, str], ...],
    tuple[tuple[Path, str], ...],
]:
    """Merge current verified routes and report stale bundles skipped closed."""

    default_policy = "auto"
    routes: list[tuple[dict[str, object], str]] = []
    digests: list[str] = []
    route_provenance: list[tuple[Path, str]] = []
    ignored: list[tuple[Path, str]] = []
    seen_matches: set[tuple[tuple[str, object], ...]] = set()
    for path in paths:
        try:
            table, digest, _manifest = load_verified_bundle(
                path,
                project_root=project_root,
            )
        except (OSError, TypeError, ValueError) as exc:
            ignored.append((path.resolve(), str(exc)))
            continue
        for match, policy in table.routes:
            fingerprint = tuple(sorted(match.items()))
            if fingerprint in seen_matches:
                raise ValueError(
                    f"verified route table {path} duplicates another exact route"
                )
            seen_matches.add(fingerprint)
            routes.append((match, policy))
            route_provenance.append((path.resolve(), digest))
        digests.append(digest)
    return (
        RouteTable(default_policy=default_policy, routes=tuple(routes)),
        tuple(digests),
        tuple(route_provenance),
        tuple(ignored),
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


def _normalize_shape(shape: Sequence[int] | torch.Size) -> tuple[int, int, int]:
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
    shape: Sequence[int] | torch.Size | None = None,
    dtype: object | None = None,
) -> dict[str, object]:
    """Build the workload-only portion of one deterministic route key."""

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
    # Reuse the route-table validator's exact scalar rules without admitting
    # process-static fields here.
    _validate_match(key, index=0)
    return key


def make_route_key(
    config: object,
    *,
    shape: Sequence[int] | torch.Size | None = None,
    dtype: object | None = None,
    device_type: str | None = None,
    device_name: str | None = None,
    compute_capability: str | None = None,
    platform_system: str | None = None,
    torch_version: str | None = None,
    cuda_runtime: str | None = None,
    triton_version: str | None = None,
) -> dict[str, object]:
    """Build a route key from explicit static facts without touching a device."""

    key: dict[str, object] = {
        **_STATIC_RUNTIME_FACTS,
        **make_workload_route_key(config, shape=shape, dtype=dtype),
    }
    if device_type is not None:
        key["device_type"] = str(device_type).strip()
    if device_name is not None:
        key["device_name"] = str(device_name).strip()
    if compute_capability is not None:
        key["compute_capability"] = str(compute_capability).strip()
    if platform_system is not None:
        key["platform_system"] = str(platform_system).strip()
    if torch_version is not None:
        key["torch"] = str(torch_version).strip()
    if cuda_runtime is not None:
        key["cuda_runtime"] = str(cuda_runtime).strip()
    if triton_version is not None:
        key["triton"] = str(triton_version).strip()
    return key


def resolve_route(table: RouteTable, key: Mapping[str, object]) -> str:
    """Return the first exact subset match or the table's default policy."""

    return resolve_route_result(table, key).policy


def resolve_route_result(
    table: RouteTable,
    key: Mapping[str, object],
) -> RouteResolution:
    """Return a policy plus whether it matched calibrated static facts."""

    for match, policy in table.routes:
        if all(key.get(field) == expected for field, expected in match.items()):
            return RouteResolution(policy=policy, origin="calibrated")
    return RouteResolution(policy=table.default_policy, origin="fallback")


def _device_facts(device: torch.device) -> tuple[str, str | None, str | None]:
    device_type = device.type
    if device_type != "cuda" or not torch.cuda.is_available():
        return device_type, None, None
    index = device.index if device.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    return device_type, torch.cuda.get_device_name(index), f"{major}.{minor}"


class OfflineDispatcher:
    """Load an offline route table once and resolve policies deterministically."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        catalog_root: str | Path | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[1]
        configured_path = Path(path) if path is not None else None

        if configured_path is not None:
            route_path = configured_path.resolve()
            self.table, self.table_sha256 = _load_route_table_bytes(route_path)
            self.path: Path | None = route_path
            self.paths = (route_path,)
            self.sources = (_route_source_label(route_path, project_root),)
            self._route_provenance = tuple(
                (self.sources[0], self.table_sha256) for _ in self.table.routes
            )
        else:
            root = (
                Path(catalog_root).resolve()
                if catalog_root is not None
                else project_root / "verified_hardware"
            )
            discovered_paths = _catalog_paths(root)
            (
                self.table,
                digests,
                raw_provenance,
                raw_ignored,
            ) = _merge_catalog_tables(discovered_paths, project_root=project_root)
            self.ignored_bundles = tuple(
                (_route_source_label(route_path, project_root), reason)
                for route_path, reason in raw_ignored
            )
            self.paths = tuple(
                path
                for path in discovered_paths
                if path.resolve() not in {item[0] for item in raw_ignored}
            )
            self.path = self.paths[0] if len(self.paths) == 1 else None
            self.sources = tuple(
                _route_source_label(route_path, project_root)
                for route_path in self.paths
            )
            self._route_provenance = tuple(
                (_route_source_label(route_path, project_root), digest)
                for route_path, digest in raw_provenance
            )
            if len(digests) == 1:
                self.table_sha256 = digests[0]
            elif digests:
                digest = hashlib.sha256()
                for source, table_digest in zip(self.sources, digests, strict=True):
                    digest.update(source.encode("utf-8"))
                    digest.update(bytes.fromhex(table_digest))
                self.table_sha256 = digest.hexdigest()
            else:
                self.table_sha256 = None

        if configured_path is not None:
            self.ignored_bundles: tuple[tuple[str, str], ...] = ()

        self.source = ",".join(self.sources) if self.sources else None

    def resolve(
        self,
        config: object,
        tensor: torch.Tensor | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: object | None = None,
        shape: Sequence[int] | torch.Size | None = None,
        device_name: str | None = None,
        compute_capability: str | None = None,
        platform_system: str | None = None,
        torch_version: str | None = None,
        cuda_runtime: str | None = None,
        triton_version: str | None = None,
    ) -> str:
        """Resolve a policy while preserving the original string-only API."""

        return self.resolve_result(
            config,
            tensor,
            device=device,
            dtype=dtype,
            shape=shape,
            device_name=device_name,
            compute_capability=compute_capability,
            platform_system=platform_system,
            torch_version=torch_version,
            cuda_runtime=cuda_runtime,
            triton_version=triton_version,
        ).policy

    def resolve_result(
        self,
        config: object,
        tensor: torch.Tensor | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: object | None = None,
        shape: Sequence[int] | torch.Size | None = None,
        device_name: str | None = None,
        compute_capability: str | None = None,
        platform_system: str | None = None,
        torch_version: str | None = None,
        cuda_runtime: str | None = None,
        triton_version: str | None = None,
    ) -> RouteResolution:
        """Resolve a policy and report calibrated-table versus fallback origin."""

        if tensor is not None:
            if shape is None:
                shape = tensor.shape
            if dtype is None:
                dtype = tensor.dtype
            if device is None:
                device = tensor.device

        normalized_device = torch.device(device) if device is not None else None
        device_type: str | None = None
        detected_name: str | None = None
        detected_capability: str | None = None
        if normalized_device is not None:
            device_type, detected_name, detected_capability = _device_facts(
                normalized_device
            )

        key = make_route_key(
            config,
            shape=shape,
            dtype=dtype,
            device_type=device_type,
            device_name=device_name or detected_name,
            compute_capability=compute_capability or detected_capability,
            platform_system=platform_system,
            torch_version=torch_version,
            cuda_runtime=cuda_runtime,
            triton_version=triton_version,
        )
        for (match, policy), (source, digest) in zip(
            self.table.routes,
            self._route_provenance,
            strict=True,
        ):
            if all(key.get(field) == expected for field, expected in match.items()):
                return RouteResolution(
                    policy=policy,
                    origin="calibrated",
                    source=source,
                    table_sha256=digest,
                )
        return RouteResolution(
            policy=self.table.default_policy,
            origin="fallback",
            source=self.source,
            table_sha256=self.table_sha256,
        )
