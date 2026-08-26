"""Deterministic policy routing from an offline-calibrated route table."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

SCHEMA_VERSION = 1
ALLOWED_POLICIES = frozenset(
    {
        "auto",
        "reference",
        "torch",
        "triton",
        "preprocess",
        "long-pv",
        "wide-epilogue",
        "cuda-graph",
        "padding",
        "packed",
    }
)
ROUTE_FIELDS = frozenset(
    {
        "device_type",
        "device_name",
        "compute_capability",
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
    {"device_type", "device_name", "compute_capability", "dtype"}
)
_INTEGER_FIELDS = frozenset({"B", "S", "D", "heads", "ffn", "layers"})
_TABLE_FIELDS = frozenset({"schema_version", "default_policy", "routes"})
_ROUTE_ENTRY_FIELDS = frozenset({"match", "policy"})


@dataclass(frozen=True)
class RouteTable:
    """Validated route table used by the deterministic resolver."""

    default_policy: str
    routes: tuple[tuple[dict[str, object], str], ...]


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
    """Validate a decoded schema-version-1 route table."""

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


def load_route_table(path: str | Path) -> RouteTable:
    """Read and validate a route table from disk."""

    route_path = Path(path)
    try:
        payload = json.loads(route_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load route table {route_path}: {exc}") from exc
    return validate_route_table(payload)


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
    return key


def resolve_route(table: RouteTable, key: Mapping[str, object]) -> str:
    """Return the first exact subset match or the table's default policy."""

    for match, policy in table.routes:
        if all(key.get(field) == expected for field, expected in match.items()):
            return policy
    return table.default_policy


def _device_facts(device: torch.device) -> tuple[str, str | None, str | None]:
    device_type = device.type
    if device_type != "cuda" or not torch.cuda.is_available():
        return device_type, None, None
    index = device.index if device.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    return device_type, torch.cuda.get_device_name(index), f"{major}.{minor}"


class OfflineDispatcher:
    """Load an offline route table once and resolve policies deterministically."""

    def __init__(self, path: str | Path | None = None) -> None:
        route_path = (
            Path(path)
            if path is not None
            else Path(__file__).with_name("dispatch_routes.json")
        )
        self.path = route_path
        self.table = load_route_table(route_path)

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
    ) -> str:
        """Resolve using only config, tensor metadata, and optional overrides."""

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
        )
        return resolve_route(self.table, key)
