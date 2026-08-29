"""Small exact-match table for formally selected configurations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import ConfigSpec

DEFAULT_DEPLOYED_CONFIGS_PATH = (
    Path(__file__).resolve().parents[1] / "deployments" / "deployed_configs.json"
)


@dataclass(frozen=True, slots=True)
class HardwareFingerprint:
    """Only hardware facts used by the runtime lookup."""

    device_name: str
    compute_capability: str

    @classmethod
    def detect(cls, device: str | torch.device) -> "HardwareFingerprint":
        resolved = torch.device(device)
        if resolved.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("deployed configuration lookup requires CUDA")
        index = (
            torch.cuda.current_device() if resolved.index is None else resolved.index
        )
        major, minor = torch.cuda.get_device_capability(index)
        return cls(
            device_name=torch.cuda.get_device_name(index),
            compute_capability=f"{major}.{minor}",
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HardwareFingerprint":
        return cls(
            device_name=str(value["device_name"]),
            compute_capability=str(value["compute_capability"]),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
        }


@dataclass(frozen=True, slots=True)
class ShapeFingerprint:
    batch_size: int
    qkv_dim: int
    heads: int
    seq_len: int
    layers: int
    causal: bool
    ffn_dim: int
    dtype: str
    padding_ratio: float
    input_scale: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ShapeFingerprint":
        return cls(
            batch_size=int(value["batch_size"]),
            qkv_dim=int(value["qkv_dim"]),
            heads=int(value["heads"]),
            seq_len=int(value["seq_len"]),
            layers=int(value["layers"]),
            causal=bool(value["causal"]),
            ffn_dim=int(value["ffn_dim"]),
            dtype=str(value["dtype"]),
            padding_ratio=float(value["padding_ratio"]),
            input_scale=float(value["input_scale"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "qkv_dim": self.qkv_dim,
            "heads": self.heads,
            "seq_len": self.seq_len,
            "layers": self.layers,
            "causal": self.causal,
            "ffn_dim": self.ffn_dim,
            "dtype": self.dtype,
            "padding_ratio": self.padding_ratio,
            "input_scale": self.input_scale,
        }


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("bundles"), list):
        raise ValueError("deployed config file must contain a bundles array")
    return value


def _keys(
    hardware: HardwareFingerprint | dict[str, Any],
    shape: ShapeFingerprint | dict[str, Any],
) -> tuple[HardwareFingerprint, ShapeFingerprint]:
    return (
        hardware
        if isinstance(hardware, HardwareFingerprint)
        else HardwareFingerprint.from_dict(hardware),
        shape
        if isinstance(shape, ShapeFingerprint)
        else ShapeFingerprint.from_dict(shape),
    )


def resolve_deployed_config(
    *,
    hardware: HardwareFingerprint | dict[str, Any],
    shape: ShapeFingerprint | dict[str, Any],
    path: str | Path = DEFAULT_DEPLOYED_CONFIGS_PATH,
) -> ConfigSpec | None:
    """Return the exact device/shape config, or ``None`` when not measured."""

    hardware_key, shape_key = _keys(hardware, shape)
    try:
        document = _load(path)
    except FileNotFoundError:
        return None
    for bundle in document["bundles"]:
        if not isinstance(bundle, dict) or not isinstance(bundle.get("hardware"), dict):
            continue
        if HardwareFingerprint.from_dict(bundle["hardware"]) != hardware_key:
            continue
        for entry in bundle.get("entries", []):
            if not isinstance(entry, dict) or not isinstance(entry.get("shape"), dict):
                continue
            if ShapeFingerprint.from_dict(entry["shape"]) == shape_key:
                return ConfigSpec.from_dict(entry["config"])
        return None
    return None


def publish_deployed_config(
    *,
    hardware: HardwareFingerprint | dict[str, Any],
    shape: ShapeFingerprint | dict[str, Any],
    config: ConfigSpec,
    path: str | Path = DEFAULT_DEPLOYED_CONFIGS_PATH,
) -> Path:
    """Replace one exact winner in the readable deployment table."""

    hardware_key, shape_key = _keys(hardware, shape)
    target = Path(path)
    try:
        document = _load(target)
    except FileNotFoundError:
        document = {"schema_version": 1, "bundles": []}

    matching_bundle: dict[str, Any] | None = None
    for bundle in document["bundles"]:
        if (
            isinstance(bundle, dict)
            and isinstance(bundle.get("hardware"), dict)
            and HardwareFingerprint.from_dict(bundle["hardware"]) == hardware_key
        ):
            matching_bundle = bundle
            break
    if matching_bundle is None:
        matching_bundle = {"hardware": hardware_key.to_dict(), "entries": []}
        document["bundles"].append(matching_bundle)

    entries = matching_bundle.setdefault("entries", [])
    replacement = {"shape": shape_key.to_dict(), "config": config.to_dict()}
    for index, entry in enumerate(entries):
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("shape"), dict)
            and ShapeFingerprint.from_dict(entry["shape"]) == shape_key
        ):
            entries[index] = replacement
            break
    else:
        entries.append(replacement)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


__all__ = [
    "DEFAULT_DEPLOYED_CONFIGS_PATH",
    "HardwareFingerprint",
    "ShapeFingerprint",
    "publish_deployed_config",
    "resolve_deployed_config",
]
