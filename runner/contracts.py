"""Small, stable contracts for benchmark inputs and result files."""

from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from project_identity import canonical_json_sha256, official_snapshot_hash


class ContractError(ValueError):
    """Raised when an input or persisted result violates the runner contract."""


OFFICIAL_WORKLOAD_SET_ID = "official_transformer_v1"


@dataclass(frozen=True)
class TransformerShape:
    """One official Transformer shape, independent of runtime precision."""

    case_id: str
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TransformerShape:
        external_fields = {
            "case_id",
            "batch_size",
            "qkv_dim",
            "heads",
            "seq_len",
            "layers",
            "causal",
            "ffn_dim",
        }
        if set(value) != external_fields:
            missing = sorted(external_fields - set(value))
            extra = sorted(set(value) - external_fields)
            raise ContractError(
                f"invalid Transformer shape fields; missing={missing}, extra={extra}"
            )
        shape = cls(
            case_id=value["case_id"],
            batch_size=value["batch_size"],
            seq_len=value["seq_len"],
            d_model=value["qkv_dim"],
            num_heads=value["heads"],
            ffn_dim=value["ffn_dim"],
            num_layers=value["layers"],
            causal=value["causal"],
        )
        shape.validate()
        return shape

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
            isinstance(item, bool) or not isinstance(item, int) for item in positive
        ):
            raise ContractError("all Transformer dimensions must be integers")
        if any(item <= 0 for item in positive):
            raise ContractError("all Transformer dimensions must be positive")
        if self.d_model % self.num_heads:
            raise ContractError("qkv_dim must be divisible by heads")
        if not isinstance(self.causal, bool):
            raise ContractError("causal must be a boolean")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "batch_size": self.batch_size,
            "qkv_dim": self.d_model,
            "heads": self.num_heads,
            "seq_len": self.seq_len,
            "layers": self.num_layers,
            "causal": self.causal,
            "ffn_dim": self.ffn_dim,
        }


@dataclass(frozen=True)
class RunVariant:
    """Runtime inputs that are intentionally separate from official shapes."""

    dtype: str = "float32"
    padding_ratio: float = 0.0
    input_scale: float = 1.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunVariant:
        required = set(cls.__dataclass_fields__)
        if set(value) != required:
            missing = sorted(required - set(value))
            extra = sorted(set(value) - required)
            raise ContractError(
                f"invalid run variant fields; missing={missing}, extra={extra}"
            )
        variant = cls(**value)
        variant.validate()
        return variant

    def validate(self) -> None:
        if not isinstance(self.dtype, str) or self.dtype not in {
            "float32",
            "float16",
            "bfloat16",
        }:
            raise ContractError(f"unsupported dtype: {self.dtype}")
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
class WorkloadSet:
    """Immutable view of the official Appendix shape table."""

    schema_version: int
    workload_set_id: str
    shapes: tuple[TransformerShape, ...]
    sha256: str
    path: Path


@dataclass(frozen=True)
class MeasurementProtocol:
    preset: str
    seed: int = 1234
    accuracy_trials: int = 5
    rtol: float = 0.02
    atol: float = 0.002
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
    if metadata.get("combined_sha256") != actual_hash:
        raise ContractError("official snapshot metadata hash is not normalized")
    return metadata


def load_workload_set(project_root: Path, workload_set_id: str) -> WorkloadSet:
    if workload_set_id != OFFICIAL_WORKLOAD_SET_ID:
        raise ContractError(
            f"unknown workload_set_id {workload_set_id!r}; "
            f"available: {OFFICIAL_WORKLOAD_SET_ID}"
        )
    path = project_root / "official" / "test_shapes.json"
    document = load_json(path)
    required = {"schema_version", "workload_set_id", "ordered_shapes"}
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
    raw_shapes = document.get("ordered_shapes")
    if not isinstance(raw_shapes, list) or not raw_shapes:
        raise ContractError("workload set must contain ordered_shapes")
    if not all(isinstance(value, dict) for value in raw_shapes):
        raise ContractError("each ordered_shapes item must be a JSON object")
    shapes = [TransformerShape.from_dict(value) for value in raw_shapes]
    case_ids = [shape.case_id for shape in shapes]
    if len(case_ids) != len(set(case_ids)):
        raise ContractError("Transformer shape case_id values must be unique")
    if len(shapes) != 14:
        raise ContractError("official workload must contain exactly 14 shapes")
    return WorkloadSet(
        schema_version=schema_version,
        workload_set_id=workload_set_id,
        shapes=tuple(shapes),
        sha256=canonical_json_sha256(path),
        path=path,
    )


def select_transformer_shape(
    workload_set: WorkloadSet, case_id: str
) -> TransformerShape:
    for shape in workload_set.shapes:
        if shape.case_id == case_id:
            return shape
    available = ", ".join(shape.case_id for shape in workload_set.shapes)
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
