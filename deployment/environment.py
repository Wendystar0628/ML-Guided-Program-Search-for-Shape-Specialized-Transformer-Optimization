"""Stable identity for GPU-kernel measurements and deployments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_OFFICIAL_DEFINITION_PATHS = (
    Path("official/test_shapes.json"),
    Path("official/torch_transformer_benchmark.py"),
)


class ImplementationScope(StrEnum):
    """Independent resident and Shape-14 implementation lifecycles."""

    RESIDENT = "resident"
    SHAPE14 = "shape14"


def _stable_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def stable_digest(value: object) -> str:
    """Return one canonical SHA-256 identity for JSON-compatible data."""

    return hashlib.sha256(_stable_json(value)).hexdigest()


def _files_digest(project_root: Path, relative_paths: tuple[Path, ...]) -> str:
    records = []
    for relative_path in sorted(relative_paths, key=lambda item: item.as_posix()):
        content = (project_root / relative_path).read_bytes()
        records.append(
            {
                "path": relative_path.as_posix(),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return stable_digest(records)


def official_definitions_digest(project_root: Path = PROJECT_ROOT) -> str:
    """Digest the immutable benchmark program and official workload table."""

    return _files_digest(Path(project_root), _OFFICIAL_DEFINITION_PATHS)


def _solution_source_scope(relative_path: Path) -> ImplementationScope | None:
    if relative_path.is_relative_to(Path("solution/shape14")):
        return ImplementationScope.SHAPE14
    if relative_path.is_relative_to(Path("solution/kernels")) or (
        relative_path.is_relative_to(Path("solution/runtimes"))
    ):
        return ImplementationScope.RESIDENT
    return None


def solution_implementation_digest(
    project_root: Path = PROJECT_ROOT,
    *,
    scope: ImplementationScope = ImplementationScope.RESIDENT,
) -> str:
    """Digest shared sources plus only the requested scope's implementation."""

    root = Path(project_root)
    solution_root = root / "solution"
    normalized_scope = ImplementationScope(scope)
    relative_paths = tuple(
        path.relative_to(root)
        for path in solution_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".py"
        and _solution_source_scope(path.relative_to(root)) in {None, normalized_scope}
    )
    if not relative_paths:
        raise FileNotFoundError(
            f"no Solution implementation files under {solution_root}"
        )
    return _files_digest(root, relative_paths)


def _driver_version(device_index: int, device_uuid: object | None) -> str:
    selector = str(device_index)
    if device_uuid is not None:
        selector = str(device_uuid)
        if not selector.startswith("GPU-"):
            selector = f"GPU-{selector}"
    command = (
        "nvidia-smi",
        f"--id={selector}",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    )
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("cannot determine the NVIDIA driver version") from exc
    versions = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(versions) != 1:
        raise RuntimeError("nvidia-smi did not return one NVIDIA driver version")
    return versions[0]


def installed_triton_version() -> str:
    """Return the installed Triton distribution version on any supported OS."""

    for distribution in ("triton", "triton-windows"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "not-installed"


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"environment fingerprint field {field!r} must be a string")
    return value


def configure_process_math_mode() -> None:
    """Use full-precision accumulation for FP16 matrix reductions."""

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


@dataclass(frozen=True, slots=True)
class EnvironmentFingerprint:
    """All execution facts that may change a GPU-kernel winner."""

    device_name: str
    compute_capability: str
    driver_version: str
    torch_version: str
    cuda_runtime_version: str
    cudnn_version: str
    triton_version: str
    matmul_precision: str
    allow_tf32: bool
    allow_fp16_reduced_precision_reduction: bool
    cudnn_allow_tf32: bool | None
    official_definitions_digest: str
    solution_implementation_digest: str
    scope: ImplementationScope = ImplementationScope.RESIDENT

    @classmethod
    def detect(
        cls,
        device: str | torch.device,
        *,
        project_root: Path = PROJECT_ROOT,
        scope: ImplementationScope = ImplementationScope.RESIDENT,
    ) -> EnvironmentFingerprint:
        resolved = torch.device(device)
        if resolved.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("environment fingerprint detection requires CUDA")
        index = (
            torch.cuda.current_device() if resolved.index is None else resolved.index
        )
        properties = torch.cuda.get_device_properties(index)
        major, minor = torch.cuda.get_device_capability(index)
        cuda_runtime = torch.version.cuda
        if not cuda_runtime:
            raise RuntimeError("CUDA runtime version is unavailable from PyTorch")
        cudnn = torch.backends.cudnn.version()
        normalized_scope = ImplementationScope(scope)
        return cls(
            device_name=torch.cuda.get_device_name(index),
            compute_capability=f"{major}.{minor}",
            driver_version=_driver_version(index, getattr(properties, "uuid", None)),
            torch_version=str(torch.__version__),
            cuda_runtime_version=str(cuda_runtime),
            cudnn_version="not-available" if cudnn is None else str(cudnn),
            triton_version=installed_triton_version(),
            matmul_precision=str(torch.get_float32_matmul_precision()),
            allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            allow_fp16_reduced_precision_reduction=bool(
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            ),
            cudnn_allow_tf32=getattr(torch.backends.cudnn, "allow_tf32", None),
            official_definitions_digest=official_definitions_digest(project_root),
            solution_implementation_digest=solution_implementation_digest(
                project_root,
                scope=normalized_scope,
            ),
            scope=normalized_scope,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EnvironmentFingerprint:
        allow_tf32 = value["allow_tf32"]
        if not isinstance(allow_tf32, bool):
            raise TypeError(
                "environment fingerprint field 'allow_tf32' must be boolean"
            )
        allow_fp16_reduced_precision_reduction = value[
            "allow_fp16_reduced_precision_reduction"
        ]
        if not isinstance(allow_fp16_reduced_precision_reduction, bool):
            raise TypeError(
                "environment fingerprint field "
                "'allow_fp16_reduced_precision_reduction' must be boolean"
            )
        cudnn_allow_tf32 = value["cudnn_allow_tf32"]
        if cudnn_allow_tf32 is not None and not isinstance(cudnn_allow_tf32, bool):
            raise TypeError(
                "environment fingerprint field 'cudnn_allow_tf32' "
                "must be boolean or null"
            )
        return cls(
            device_name=_required_string(value["device_name"], "device_name"),
            compute_capability=_required_string(
                value["compute_capability"], "compute_capability"
            ),
            driver_version=_required_string(value["driver_version"], "driver_version"),
            torch_version=_required_string(value["torch_version"], "torch_version"),
            cuda_runtime_version=_required_string(
                value["cuda_runtime_version"], "cuda_runtime_version"
            ),
            cudnn_version=_required_string(value["cudnn_version"], "cudnn_version"),
            triton_version=_required_string(value["triton_version"], "triton_version"),
            matmul_precision=_required_string(
                value["matmul_precision"], "matmul_precision"
            ),
            allow_tf32=allow_tf32,
            allow_fp16_reduced_precision_reduction=(
                allow_fp16_reduced_precision_reduction
            ),
            cudnn_allow_tf32=cudnn_allow_tf32,
            official_definitions_digest=_required_string(
                value["official_definitions_digest"], "official_definitions_digest"
            ),
            solution_implementation_digest=_required_string(
                value["solution_implementation_digest"],
                "solution_implementation_digest",
            ),
            scope=ImplementationScope(_required_string(value["scope"], "scope")),
        )

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "scope": self.scope.value,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "driver_version": self.driver_version,
            "torch_version": self.torch_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "cudnn_version": self.cudnn_version,
            "triton_version": self.triton_version,
            "matmul_precision": self.matmul_precision,
            "allow_tf32": self.allow_tf32,
            "allow_fp16_reduced_precision_reduction": (
                self.allow_fp16_reduced_precision_reduction
            ),
            "cudnn_allow_tf32": self.cudnn_allow_tf32,
            "official_definitions_digest": self.official_definitions_digest,
            "solution_implementation_digest": self.solution_implementation_digest,
        }

    @property
    def identity(self) -> str:
        """Stable deployment identity for this exact executable environment."""

        return stable_digest(self.to_dict())

    @property
    def measurement_identity(self) -> str:
        """Hardware/runtime identity, independent of the measured source version."""

        value = self.to_dict()
        value.pop("official_definitions_digest")
        value.pop("solution_implementation_digest")
        return stable_digest(value)


__all__ = [
    "EnvironmentFingerprint",
    "ImplementationScope",
    "configure_process_math_mode",
    "installed_triton_version",
    "official_definitions_digest",
    "solution_implementation_digest",
    "stable_digest",
]
