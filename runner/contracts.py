"""Minimal benchmark inputs shared by the CLI and search engine."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """Raised for invalid user-facing benchmark inputs."""


@dataclass(frozen=True, slots=True)
class TransformerShape:
    case_id: str
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TransformerShape":
        try:
            shape = cls(
                case_id=str(value["case_id"]),
                batch_size=int(value["batch_size"]),
                seq_len=int(value["seq_len"]),
                d_model=int(value["qkv_dim"]),
                num_heads=int(value["heads"]),
                ffn_dim=int(value["ffn_dim"]),
                num_layers=int(value["layers"]),
                causal=bool(value["causal"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid workload shape: {exc}") from exc
        shape.validate()
        return shape

    def validate(self) -> None:
        if not self.case_id:
            raise ContractError("case_id must not be empty")
        dimensions = (
            self.batch_size,
            self.seq_len,
            self.d_model,
            self.num_heads,
            self.ffn_dim,
            self.num_layers,
        )
        if any(value <= 0 for value in dimensions):
            raise ContractError("all workload dimensions must be positive")
        if self.d_model % self.num_heads:
            raise ContractError("qkv_dim must be divisible by heads")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def streamed(self) -> bool:
        return self.case_id == "official_14"

    def to_dict(self) -> dict[str, Any]:
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


@dataclass(frozen=True, slots=True)
class RunVariant:
    dtype: str = "float32"
    padding_ratio: float = 0.0
    input_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ContractError(f"unsupported dtype: {self.dtype}")
        if not math.isfinite(self.padding_ratio) or not 0.0 <= self.padding_ratio < 1.0:
            raise ContractError("padding_ratio must be in [0, 1)")
        if not math.isfinite(self.input_scale) or self.input_scale <= 0.0:
            raise ContractError("input_scale must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MeasurementProtocol:
    accuracy_trials: int
    warmup: int
    repeats: int
    rounds: int
    full_logical_batch: bool = True
    seed: int = 1234
    rtol: float = 0.02
    atol: float = 0.002

    @classmethod
    def for_preset(cls, preset: str) -> "MeasurementProtocol":
        if preset == "smoke":
            return cls(accuracy_trials=2, warmup=2, repeats=5, rounds=2)
        if preset == "formal":
            return cls(accuracy_trials=5, warmup=20, repeats=100, rounds=3)
        raise ContractError(f"unknown preset: {preset}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"expected a JSON object in {path}")
    return value


def load_shapes(project_root: Path) -> tuple[TransformerShape, ...]:
    document = load_json(project_root / "official" / "test_shapes.json")
    raw_shapes = document.get("ordered_shapes")
    if not isinstance(raw_shapes, list):
        raise ContractError("official/test_shapes.json has no ordered_shapes array")
    shapes = tuple(TransformerShape.from_dict(value) for value in raw_shapes)
    if not shapes:
        raise ContractError("official workload is empty")
    return shapes


def load_shape(project_root: Path, case_id: str) -> TransformerShape:
    for shape in load_shapes(project_root):
        if shape.case_id == case_id:
            return shape
    raise ContractError(f"unknown case_id: {case_id}")


def write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


__all__ = [
    "ContractError",
    "MeasurementProtocol",
    "RunVariant",
    "TransformerShape",
    "load_json",
    "load_shape",
    "load_shapes",
    "write_json",
]
