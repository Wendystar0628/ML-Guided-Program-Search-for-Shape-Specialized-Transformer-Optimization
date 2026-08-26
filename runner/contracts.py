"""Small, stable contracts for benchmark inputs and result files."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """Raised when an input or persisted result violates the runner contract."""


@dataclass(frozen=True)
class WorkloadCase:
    case_id: str
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    dtype: str
    causal: bool
    padding_ratio: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorkloadCase:
        required = set(cls.__dataclass_fields__)
        if set(value) != required:
            missing = sorted(required - set(value))
            extra = sorted(set(value) - required)
            raise ContractError(
                f"invalid workload fields; missing={missing}, extra={extra}"
            )
        case = cls(**value)
        case.validate()
        return case

    def validate(self) -> None:
        if not self.case_id:
            raise ContractError("case_id must not be empty")
        positive = (
            self.batch_size,
            self.seq_len,
            self.d_model,
            self.num_heads,
            self.ffn_dim,
            self.num_layers,
        )
        if any(value <= 0 for value in positive):
            raise ContractError("all workload dimensions must be positive")
        if self.d_model % self.num_heads:
            raise ContractError("d_model must be divisible by num_heads")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ContractError(f"unsupported dtype: {self.dtype}")
        if not 0.0 <= self.padding_ratio < 1.0:
            raise ContractError("padding_ratio must be in [0, 1)")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MeasurementProtocol:
    preset: str
    seed: int = 1234
    input_scale: float = 1.0
    accuracy_trials: int = 5
    rtol: float = 0.01
    atol: float = 0.001
    warmup: int = 20
    repeats: int = 100
    rounds: int = 3
    compile_baseline: bool = False
    compile_solution: bool = False
    compile_mode: str = "default"
    matmul_precision: str = "high"
    allow_tf32: bool = True
    timeout_seconds: float = 900.0

    @classmethod
    def for_preset(
        cls,
        preset: str,
        *,
        compile_baseline: bool = False,
        compile_solution: bool = False,
        compile_mode: str = "default",
        matmul_precision: str = "high",
        allow_tf32: bool = True,
        timeout_seconds: float | None = None,
    ) -> MeasurementProtocol:
        if preset == "smoke":
            protocol = cls(
                preset="smoke",
                accuracy_trials=2,
                warmup=2,
                repeats=5,
                rounds=2,
                timeout_seconds=300.0,
            )
        elif preset == "formal":
            protocol = cls(preset="formal")
        else:
            raise ContractError(f"unknown preset: {preset}")
        values = asdict(protocol)
        values.update(
            compile_baseline=compile_baseline,
            compile_solution=compile_solution,
            compile_mode=compile_mode,
            matmul_precision=matmul_precision,
            allow_tf32=allow_tf32,
        )
        if timeout_seconds is not None:
            values["timeout_seconds"] = timeout_seconds
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        if self.accuracy_trials <= 0:
            raise ContractError("accuracy_trials must be positive")
        if self.rtol < 0 or self.atol < 0:
            raise ContractError("rtol and atol must be non-negative")
        if self.input_scale <= 0:
            raise ContractError("input_scale must be positive")
        if self.warmup < 0 or self.repeats <= 0 or self.rounds <= 0:
            raise ContractError("invalid timing iteration counts")
        if self.compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise ContractError(f"unsupported compile mode: {self.compile_mode}")
        if self.matmul_precision not in {"highest", "high", "medium"}:
            raise ContractError(
                f"unsupported matmul precision: {self.matmul_precision}"
            )
        if self.timeout_seconds <= 0:
            raise ContractError("timeout_seconds must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def solution_source_hash(solution_root: Path) -> str:
    """Hash files that can change the measured implementation."""

    suffixes = {".py", ".cpp", ".cc", ".c", ".h", ".cu", ".cuh"}
    files = sorted(
        path
        for path in solution_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in suffixes
        and "__pycache__" not in path.parts
    )
    if not files:
        raise ContractError(f"no solution source files found under {solution_root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(solution_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"expected a JSON object in {path}")
    return value


def validate_official_snapshot(project_root: Path) -> dict[str, Any]:
    metadata = load_json(project_root / "official" / "snapshot.json")
    snapshot = project_root / str(metadata.get("snapshot_path", ""))
    expected_size = metadata.get("byte_count")
    expected_hash = metadata.get("sha256")
    if not snapshot.is_file():
        raise ContractError(f"official snapshot is missing: {snapshot}")
    if snapshot.stat().st_size != expected_size:
        raise ContractError("official snapshot byte count does not match metadata")
    actual_hash = sha256_file(snapshot)
    if actual_hash != expected_hash:
        raise ContractError("official snapshot checksum does not match metadata")
    return metadata


def load_workload_set(project_root: Path, workload_set_id: str) -> dict[str, Any]:
    path = project_root / "runner" / "workloads" / f"{workload_set_id}.json"
    document = load_json(path)
    if document.get("workload_set_id") != workload_set_id:
        raise ContractError("workload_set_id does not match its filename")
    raw_cases = document.get("ordered_cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ContractError("workload set must contain ordered_cases")
    cases = [WorkloadCase.from_dict(value) for value in raw_cases]
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ContractError("workload case_id values must be unique")
    return {"workload_set_id": workload_set_id, "cases": cases, "path": path}


def select_workload_case(workload_set: dict[str, Any], case_id: str) -> WorkloadCase:
    for case in workload_set["cases"]:
        if case.case_id == case_id:
            return case
    available = ", ".join(case.case_id for case in workload_set["cases"])
    raise ContractError(f"unknown case_id {case_id!r}; available: {available}")


def atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    """Publish one immutable, strict JSON result using a same-directory rename."""

    if path.exists():
        raise ContractError(f"refusing to overwrite existing result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        document,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
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
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
