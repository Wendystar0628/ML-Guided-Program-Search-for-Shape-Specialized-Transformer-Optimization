"""Deterministic policy routing from an offline-calibrated route table."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

SCHEMA_VERSION = 2
ALLOWED_POLICIES = frozenset(
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
        "padding",
        "packed",
    }
)
ROUTE_FIELDS = frozenset(
    {
        "device_type",
        "device_name",
        "compute_capability",
        "platform_system",
        "torch",
        "cuda_runtime",
        "triton",
        "dtype",
        "B",
        "S",
        "D",
        "heads",
        "ffn",
        "layers",
        "causal",
    }
)
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
ROUTE_TABLE_ENV = "TRANSFORMER_ROUTE_TABLE"


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
        policy = _require_policy(entry["policy"], field=f"routes[{index}].policy")
        fingerprint = tuple(sorted(match.items()))
        if fingerprint in seen_matches:
            raise ValueError(f"routes[{index}].match duplicates an earlier route")
        seen_matches.add(fingerprint)
        routes.append((match, policy))

    return RouteTable(default_policy=default_policy, routes=tuple(routes))


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
) -> tuple[
    RouteTable,
    tuple[str, ...],
    tuple[tuple[Path, str], ...],
]:
    """Merge complete device routes and reject ambiguous duplicate matches."""

    default_policy = "auto"
    routes: list[tuple[dict[str, object], str]] = []
    digests: list[str] = []
    route_provenance: list[tuple[Path, str]] = []
    seen_matches: set[tuple[tuple[str, object], ...]] = set()
    for path in paths:
        table, digest = _load_route_table_bytes(path)
        if table.default_policy != default_policy:
            raise ValueError(
                f"verified route table {path} must use default_policy=auto"
            )
        for match, policy in table.routes:
            missing = ROUTE_FIELDS - set(match)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(
                    f"verified route table {path} has a non-exact route; "
                    f"missing fields: {names}"
                )
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

    if shape is None:
        shape = (
            _config_value(config, "batch_size"),
            _config_value(config, "seq_len"),
            _config_value(config, "d_model"),
        )
    batch, sequence, model_dim = _normalize_shape(shape)

    key: dict[str, object] = {
        **_STATIC_RUNTIME_FACTS,
        "B": batch,
        "S": sequence,
        "D": model_dim,
        "heads": _config_value(config, "num_heads"),
        "ffn": _config_value(config, "ffn_dim"),
        "layers": _config_value(config, "num_layers"),
        "causal": _config_value(config, "causal"),
    }
    if dtype is not None:
        key["dtype"] = _normalize_dtype(dtype)
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
        configured_path: Path | None = None
        if path is not None:
            configured_path = Path(path)
        else:
            configured = os.environ.get(ROUTE_TABLE_ENV)
            if configured is not None:
                if not configured.strip():
                    raise ValueError(f"{ROUTE_TABLE_ENV} must not be blank")
                configured_path = Path(configured)

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
            self.paths = _catalog_paths(root)
            self.table, digests, raw_provenance = _merge_catalog_tables(self.paths)
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
