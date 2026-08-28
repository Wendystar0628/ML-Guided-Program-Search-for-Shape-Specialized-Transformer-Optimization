"""Runtime-only deterministic dispatch over verified offline routes."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from collections.abc import Sequence
from pathlib import Path

import torch

from route_contracts import (
    RouteResolution,
    RouteTable,
    load_route_table_with_digest,
    load_verified_bundle,
    make_route_key,
)


def _route_source_label(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _catalog_paths(catalog_root: Path) -> tuple[Path, ...]:
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
        RouteTable(default_policy="eager-sdpa", routes=tuple(routes)),
        tuple(digests),
        tuple(route_provenance),
        tuple(ignored),
    )


def _driver_version() -> str:
    """Collect the process driver lazily; failure is an exact-route fact."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    rows = completed.stdout.splitlines()
    if completed.returncode != 0 or not rows or not rows[0].strip():
        return "unavailable"
    return rows[0].strip()


def _device_facts(device: torch.device) -> tuple[str, str, str]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return device.type, device.type, "unavailable"
    index = device.index if device.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    return device.type, torch.cuda.get_device_name(index), f"{major}.{minor}"


class OfflineDispatcher:
    """Load verified routes once and resolve them from current runtime facts."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        catalog_root: str | Path | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[1]
        configured_path = Path(path) if path is not None else None
        self._driver: str | None = None

        if configured_path is not None:
            route_path = configured_path.resolve()
            self.table, self.table_sha256 = load_route_table_with_digest(route_path)
            self.path: Path | None = route_path
            self.paths = (route_path,)
            self.sources = (_route_source_label(route_path, project_root),)
            self._route_provenance = tuple(
                (self.sources[0], self.table_sha256) for _ in self.table.routes
            )
            self.ignored_bundles: tuple[tuple[str, str], ...] = ()
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
            ignored_paths = {item[0] for item in raw_ignored}
            self.ignored_bundles = tuple(
                (_route_source_label(route_path, project_root), reason)
                for route_path, reason in raw_ignored
            )
            self.paths = tuple(
                route_path
                for route_path in discovered_paths
                if route_path.resolve() not in ignored_paths
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

        self.source = ",".join(self.sources) if self.sources else None

    def _runtime_driver(self) -> str:
        if self._driver is None:
            self._driver = _driver_version()
        return self._driver

    def resolve(
        self,
        config: object,
        tensor: torch.Tensor | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: object | None = None,
        shape: Sequence[int] | None = None,
        device_name: str | None = None,
        compute_capability: str | None = None,
        platform_system: str | None = None,
        torch_version: str | None = None,
        cuda_runtime: str | None = None,
        driver: str | None = None,
        matmul_precision: str | None = None,
        allow_tf32: bool | None = None,
    ) -> str:
        """Resolve a policy while preserving the string-only convenience API."""

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
            driver=driver,
            matmul_precision=matmul_precision,
            allow_tf32=allow_tf32,
        ).policy

    def resolve_result(
        self,
        config: object,
        tensor: torch.Tensor | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: object | None = None,
        shape: Sequence[int] | None = None,
        device_name: str | None = None,
        compute_capability: str | None = None,
        platform_system: str | None = None,
        torch_version: str | None = None,
        cuda_runtime: str | None = None,
        driver: str | None = None,
        matmul_precision: str | None = None,
        allow_tf32: bool | None = None,
    ) -> RouteResolution:
        """Resolve a policy and report calibrated-table versus fallback origin."""

        if tensor is not None:
            if shape is None:
                shape = tensor.shape
            if dtype is None:
                dtype = tensor.dtype
            if device is None:
                device = tensor.device

        normalized_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        device_type, detected_name, detected_capability = _device_facts(
            normalized_device
        )
        resolved_cuda_runtime = (
            str(torch.version.cuda) if torch.version.cuda is not None else "unavailable"
        )
        resolved_allow_tf32 = (
            bool(torch.backends.cuda.matmul.allow_tf32)
            if allow_tf32 is None
            else allow_tf32
        )
        key = make_route_key(
            config,
            shape=shape,
            dtype=dtype,
            device_type=device_type,
            device_name=device_name or detected_name,
            compute_capability=compute_capability or detected_capability,
            platform_system=platform_system or platform.system(),
            torch_version=torch_version or str(torch.__version__),
            cuda_runtime=cuda_runtime or resolved_cuda_runtime,
            driver=(
                driver or self._runtime_driver()
                if device_type == "cuda"
                else "unavailable"
            ),
            matmul_precision=(matmul_precision or torch.get_float32_matmul_precision()),
            allow_tf32=resolved_allow_tf32,
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


__all__ = ["OfflineDispatcher"]
