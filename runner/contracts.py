"""Small, stable contracts for benchmark inputs and result files."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from project_identity import official_snapshot_hash


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
    input_scale: float = 1.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorkloadCase:
        # ``input_scale`` is an actual optional field, not merely a constructor
        # default.  Persisted workloads may omit it and receive the documented
        # neutral value of 1.0.
        allowed = set(cls.__dataclass_fields__)
        required = allowed - {"input_scale"}
        if not required.issubset(value) or not set(value).issubset(allowed):
            missing = sorted(required - set(value))
            extra = sorted(set(value) - allowed)
            raise ContractError(
                f"invalid workload fields; missing={missing}, extra={extra}"
            )
        case = cls(**{"input_scale": 1.0, **value})
        case.validate()
        return case

    def validate(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ContractError("case_id must not be empty")
        positive = (
            self.batch_size,
            self.seq_len,
            self.d_model,
            self.num_heads,
            self.ffn_dim,
            self.num_layers,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in positive
        ):
            raise ContractError("all workload dimensions must be integers")
        if any(value <= 0 for value in positive):
            raise ContractError("all workload dimensions must be positive")
        if self.d_model % self.num_heads:
            raise ContractError("d_model must be divisible by num_heads")
        if not isinstance(self.dtype, str) or self.dtype not in {
            "float32",
            "float16",
            "bfloat16",
        }:
            raise ContractError(f"unsupported dtype: {self.dtype}")
        if not isinstance(self.causal, bool):
            raise ContractError("causal must be a boolean")
        if isinstance(self.padding_ratio, bool) or not isinstance(
            self.padding_ratio, (int, float)
        ):
            raise ContractError("padding_ratio must be numeric")
        if not math.isfinite(float(self.padding_ratio)) or not (
            0.0 <= self.padding_ratio < 1.0
        ):
            raise ContractError("padding_ratio must be in [0, 1)")
        if isinstance(self.input_scale, bool) or not isinstance(
            self.input_scale, (int, float)
        ):
            raise ContractError("input_scale must be numeric")
        if not math.isfinite(float(self.input_scale)) or self.input_scale <= 0:
            raise ContractError("input_scale must be a finite positive number")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkloadGroup:
    group_id: str
    display_name: str
    weight: float
    case_ids: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorkloadGroup:
        required = {"group_id", "display_name", "weight", "case_ids"}
        if set(value) != required:
            missing = sorted(required - set(value))
            extra = sorted(set(value) - required)
            raise ContractError(
                f"invalid workload group fields; missing={missing}, extra={extra}"
            )
        raw_case_ids = value["case_ids"]
        if not isinstance(raw_case_ids, list) or not all(
            isinstance(case_id, str) for case_id in raw_case_ids
        ):
            raise ContractError("workload group case_ids must be a list of strings")
        group = cls(
            group_id=value["group_id"],
            display_name=value["display_name"],
            weight=value["weight"],
            case_ids=tuple(raw_case_ids),
        )
        group.validate()
        return group

    def validate(self) -> None:
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ContractError("workload group_id must not be empty")
        if not isinstance(self.display_name, str) or not self.display_name:
            raise ContractError("workload group display_name must not be empty")
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)):
            raise ContractError("workload group weight must be numeric")
        if not math.isfinite(float(self.weight)) or self.weight <= 0:
            raise ContractError("workload group weight must be finite and positive")
        if not self.case_ids or any(not case_id for case_id in self.case_ids):
            raise ContractError("workload group case_ids must not be empty")
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ContractError("workload group case_ids must be unique")

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "display_name": self.display_name,
            "weight": self.weight,
            "case_ids": list(self.case_ids),
        }


@dataclass(frozen=True)
class WorkloadSet:
    """Immutable, typed workload document used by runner services."""

    schema_version: int
    workload_set_id: str
    cases: tuple[WorkloadCase, ...]
    groups: tuple[WorkloadGroup, ...]
    sha256: str
    path: Path


@dataclass(frozen=True)
class MeasurementProtocol:
    preset: str
    seed: int = 1234
    accuracy_trials: int = 5
    rtol: float = 0.01
    atol: float = 0.001
    warmup: int = 20
    repeats: int = 100
    rounds: int = 3
    compile_baseline: bool = False
    compile_solution: bool = False
    cuda_graph_solution: bool = False
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
        cuda_graph_solution: bool = False,
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
            cuda_graph_solution=cuda_graph_solution,
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
        counts = (self.accuracy_trials, self.warmup, self.repeats, self.rounds)
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in counts
        ):
            raise ContractError("measurement iteration counts must be integers")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ContractError("seed must be an integer")
        if self.accuracy_trials <= 0:
            raise ContractError("accuracy_trials must be positive")
        numeric = (self.rtol, self.atol, self.timeout_seconds)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric
        ):
            raise ContractError("measurement numeric values must be finite")
        if self.rtol < 0 or self.atol < 0:
            raise ContractError("rtol and atol must be non-negative")
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
        if not isinstance(self.compile_baseline, bool) or not isinstance(
            self.compile_solution, bool
        ):
            raise ContractError("compile flags must be booleans")
        if not isinstance(self.cuda_graph_solution, bool):
            raise ContractError("cuda_graph_solution must be a boolean")
        if self.cuda_graph_solution and (
            self.compile_baseline or self.compile_solution
        ):
            raise ContractError(
                "CUDA Graph and torch.compile candidates cannot combine"
            )
        if not isinstance(self.allow_tf32, bool):
            raise ContractError("allow_tf32 must be a boolean")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"expected a JSON object in {path}")
    return value


def validate_official_snapshot(project_root: Path) -> dict[str, Any]:
    metadata = load_json(project_root / "official" / "snapshot.json")
    try:
        actual_hash = official_snapshot_hash(project_root)
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise ContractError(str(exc)) from exc
    if metadata.get("sha256") != actual_hash:
        raise ContractError("official snapshot metadata hash is not normalized")
    return metadata


def load_workload_set(project_root: Path, workload_set_id: str) -> WorkloadSet:
    path = project_root / "runner" / "workloads" / f"{workload_set_id}.json"
    document = load_json(path)
    required = {"schema_version", "workload_set_id", "groups", "ordered_cases"}
    if set(document) != required:
        missing = sorted(required - set(document))
        extra = sorted(set(document) - required)
        raise ContractError(
            f"invalid workload document fields; missing={missing}, extra={extra}"
        )
    schema_version = document["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        raise ContractError(f"unsupported workload schema_version: {schema_version!r}")
    if document.get("workload_set_id") != workload_set_id:
        raise ContractError("workload_set_id does not match its filename")
    raw_cases = document.get("ordered_cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ContractError("workload set must contain ordered_cases")
    if not all(isinstance(value, dict) for value in raw_cases):
        raise ContractError("each ordered_cases item must be a JSON object")
    cases = [WorkloadCase.from_dict(value) for value in raw_cases]
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ContractError("workload case_id values must be unique")

    raw_groups = document.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ContractError("workload set must contain groups")
    if not all(isinstance(value, dict) for value in raw_groups):
        raise ContractError("each groups item must be a JSON object")
    groups = [WorkloadGroup.from_dict(value) for value in raw_groups]
    group_ids = [group.group_id for group in groups]
    if len(group_ids) != len(set(group_ids)):
        raise ContractError("workload group_id values must be unique")
    if not math.isclose(
        math.fsum(float(group.weight) for group in groups),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ContractError("workload group weights must sum to 1")
    grouped_case_ids = [case_id for group in groups for case_id in group.case_ids]
    if len(grouped_case_ids) != len(set(grouped_case_ids)):
        raise ContractError("each workload case must belong to exactly one group")
    if set(grouped_case_ids) != set(case_ids):
        missing = sorted(set(case_ids) - set(grouped_case_ids))
        unknown = sorted(set(grouped_case_ids) - set(case_ids))
        raise ContractError(
            f"workload groups do not cover the cases; missing={missing}, unknown={unknown}"
        )

    canonical = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return WorkloadSet(
        schema_version=schema_version,
        workload_set_id=workload_set_id,
        cases=tuple(cases),
        groups=tuple(groups),
        sha256=hashlib.sha256(canonical).hexdigest(),
        path=path,
    )


def select_workload_case(workload_set: WorkloadSet, case_id: str) -> WorkloadCase:
    for case in workload_set.cases:
        if case.case_id == case_id:
            return case
    available = ", ".join(case.case_id for case in workload_set.cases)
    raise ContractError(f"unknown case_id {case_id!r}; available: {available}")


def _atomic_publish_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        document,
        ensure_ascii=False,
        indent=2,
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
        json.loads(temporary_path.read_text(encoding="utf-8"))
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    """Publish one immutable, strict JSON result using a same-directory rename."""

    if path.exists():
        raise ContractError(f"refusing to overwrite existing result: {path}")
    _atomic_publish_json(path, document)


def atomic_replace_json(path: Path, document: dict[str, Any]) -> None:
    """Atomically replace one explicitly mutable JSON reference document."""

    _atomic_publish_json(path, document)
