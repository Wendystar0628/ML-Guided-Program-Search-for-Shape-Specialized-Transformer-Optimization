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
    """Return the verified hash of the repository's official code snapshot."""

    metadata_path = project_root.resolve() / "official" / "snapshot.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("official snapshot metadata must be a JSON object")
    snapshot_value = metadata.get("snapshot_path")
    expected_size = metadata.get("byte_count")
    expected_hash = metadata.get("sha256")
    if not isinstance(snapshot_value, str) or not snapshot_value:
        raise ValueError("official snapshot metadata is missing snapshot_path")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int):
        raise TypeError("official snapshot metadata has an invalid byte_count")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("official snapshot metadata has an invalid sha256")
    snapshot_path = project_root.resolve() / snapshot_value
    if not snapshot_path.is_file():
        raise ValueError(f"official snapshot is missing: {snapshot_path}")
    if snapshot_path.stat().st_size != expected_size:
        raise ValueError("official snapshot byte count does not match metadata")
    actual_hash = sha256_file(snapshot_path)
    if actual_hash != expected_hash.lower():
        raise ValueError("official snapshot checksum does not match metadata")
    return actual_hash


__all__ = [
    "canonical_json_sha256",
    "official_snapshot_hash",
    "sha256_file",
    "solution_implementation_hash",
]
