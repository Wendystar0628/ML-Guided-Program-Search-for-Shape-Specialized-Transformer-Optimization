"""Small repository identities shared by Solution and Runner code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_SOLUTION_SUFFIXES = frozenset({".py", ".cpp", ".cc", ".c", ".h", ".cu", ".cuh"})


def sha256_file(path: Path) -> str:
    """Hash one file without loading it fully into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def solution_implementation_hash(solution_root: Path) -> str:
    """Hash the executable Solution source tree in a path-stable order."""

    resolved_root = solution_root.resolve()
    files = sorted(
        path
        for path in resolved_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in _SOLUTION_SUFFIXES
        and "__pycache__" not in path.parts
    )
    if not files:
        raise ValueError(f"no Solution source files found under {resolved_root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(resolved_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def canonical_json_sha256(path: Path) -> str:
    """Hash a JSON document independently of whitespace and key order."""

    document = json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def official_snapshot_hash(project_root: Path) -> str:
    """Verify and hash the official benchmark together with its shape table."""

    resolved_project = project_root.resolve()
    metadata_path = resolved_project / "official" / "snapshot.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("official snapshot metadata must be a JSON object")
    required = {
        "schema_version",
        "source",
        "benchmark",
        "shapes",
        "combined_sha256",
    }
    if set(metadata) != required or metadata.get("schema_version") != 2:
        raise ValueError("unsupported official snapshot metadata schema")

    expected_paths = {
        "benchmark": "official/torch_transformer_benchmark.py",
        "shapes": "official/test_shapes.json",
    }
    component_hashes: dict[str, str] = {}
    for component, expected_path in expected_paths.items():
        descriptor = metadata.get(component)
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "path",
            "byte_count",
            "sha256",
        }:
            raise ValueError(f"invalid official {component} descriptor")
        if descriptor.get("path") != expected_path:
            raise ValueError(f"official {component} path is not canonical")
        expected_size = descriptor.get("byte_count")
        expected_hash = descriptor.get("sha256")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int):
            raise TypeError(f"official {component} byte_count is invalid")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"official {component} sha256 is invalid")
        component_path = resolved_project / expected_path
        if not component_path.is_file():
            raise ValueError(f"official {component} is missing: {component_path}")
        if component_path.stat().st_size != expected_size:
            raise ValueError(f"official {component} byte count does not match")
        actual_hash = (
            canonical_json_sha256(component_path)
            if component == "shapes"
            else sha256_file(component_path)
        )
        if actual_hash != expected_hash.lower():
            raise ValueError(f"official {component} checksum does not match")
        component_hashes[component] = actual_hash

    combined_payload = {
        "benchmark_sha256": component_hashes["benchmark"],
        "schema_version": 2,
        "shapes_sha256": component_hashes["shapes"],
    }
    combined_hash = hashlib.sha256(
        json.dumps(
            combined_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    expected_combined = metadata.get("combined_sha256")
    if not isinstance(expected_combined, str) or expected_combined.lower() != combined_hash:
        raise ValueError("official combined checksum does not match metadata")
    return combined_hash


__all__ = [
    "canonical_json_sha256",
    "official_snapshot_hash",
    "sha256_file",
    "solution_implementation_hash",
]
