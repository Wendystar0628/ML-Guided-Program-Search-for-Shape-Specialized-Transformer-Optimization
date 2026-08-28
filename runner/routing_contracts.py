"""Runner adapters for the neutral exact-route contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from route_contracts import (
    ROUTE_FIELDS,
    WORKLOAD_ROUTE_FIELDS,
    make_route_key,
    make_workload_route_key,
)
from runner.contracts import RunVariant, TransformerShape


@dataclass(frozen=True)
class HardwareIdentity:
    """Route-visible hardware and software identity for one runtime."""

    device_type: str
    device_name: str
    compute_capability: str
    platform_system: str
    torch: str
    cuda_runtime: str
    driver: str
    matmul_precision: str
    allow_tf32: bool

    def as_route_fields(self) -> dict[str, object]:
        return {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "platform_system": self.platform_system,
            "torch": self.torch,
            "cuda_runtime": self.cuda_runtime,
            "driver": self.driver,
            "matmul_precision": self.matmul_precision,
            "allow_tf32": self.allow_tf32,
        }

    def as_make_route_key_kwargs(self) -> dict[str, object]:
        return {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "platform_system": self.platform_system,
            "torch_version": self.torch,
            "cuda_runtime": self.cuda_runtime,
            "driver": self.driver,
            "matmul_precision": self.matmul_precision,
            "allow_tf32": self.allow_tf32,
        }


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"route identity is missing {name}")
    return value.strip()


def _runtime_policy(value: object, name: str) -> tuple[str, bool]:
    if not isinstance(value, Mapping):
        raise TypeError(f"route identity is missing {name}")
    precision = _required_string(value.get("matmul_precision"), "matmul_precision")
    if precision not in {"highest", "high", "medium"}:
        raise ValueError("route identity has unsupported matmul_precision")
    allow_tf32 = value.get("allow_tf32")
    if not isinstance(allow_tf32, bool):
        raise TypeError("route identity allow_tf32 must be boolean")
    return precision, allow_tf32


def hardware_identity_from_flat_profile(
    profile: Mapping[str, Any],
) -> HardwareIdentity:
    """Read the compact identity embedded in one tuning summary."""

    precision, allow_tf32 = _runtime_policy(profile, "runtime policy")
    return HardwareIdentity(
        device_type=_required_string(profile.get("device_type"), "device_type"),
        device_name=_required_string(profile.get("device_name"), "device_name"),
        compute_capability=_required_string(
            profile.get("compute_capability"), "compute_capability"
        ),
        platform_system=_required_string(
            profile.get("platform_system"), "platform_system"
        ),
        torch=_required_string(profile.get("torch"), "torch"),
        cuda_runtime=_required_string(profile.get("cuda_runtime"), "cuda_runtime"),
        driver=_required_string(profile.get("driver") or "unavailable", "driver"),
        matmul_precision=precision,
        allow_tf32=allow_tf32,
    )


def hardware_identity_from_hardware_profile(
    hardware_profile: Mapping[str, Any],
    *,
    runtime_policy: Mapping[str, Any],
) -> HardwareIdentity:
    """Read the nested identity emitted by the hardware probe."""

    gpu = hardware_profile.get("gpu")
    platform_profile = hardware_profile.get("platform")
    software = hardware_profile.get("software")
    if not isinstance(gpu, Mapping):
        raise TypeError("route identity is missing gpu")
    if not isinstance(platform_profile, Mapping):
        raise TypeError("route identity is missing platform")
    if not isinstance(software, Mapping):
        raise TypeError("route identity is missing software")
    precision, allow_tf32 = _runtime_policy(runtime_policy, "runtime policy")
    return HardwareIdentity(
        device_type=_required_string(
            hardware_profile.get("device_type"), "device_type"
        ),
        device_name=_required_string(gpu.get("name"), "device_name"),
        compute_capability=_required_string(
            gpu.get("compute_capability"), "compute_capability"
        ),
        platform_system=_required_string(
            platform_profile.get("system"), "platform_system"
        ),
        torch=_required_string(software.get("torch"), "torch"),
        cuda_runtime=_required_string(software.get("cuda_runtime"), "cuda_runtime"),
        driver=_required_string(software.get("driver") or "unavailable", "driver"),
        matmul_precision=precision,
        allow_tf32=allow_tf32,
    )


def hardware_identity_from_verified_profile(
    profile: Mapping[str, Any],
) -> HardwareIdentity:
    """Validate a persisted probe profile and return its route identity."""

    if profile.get("schema_version") != 1:
        raise ValueError("verified profile must use schema_version 1")
    if profile.get("device_operation_passed") is not True:
        raise ValueError("verified profile did not pass its device operation")
    hardware_profile = profile.get("hardware_profile")
    if not isinstance(hardware_profile, Mapping):
        raise TypeError("verified profile is missing hardware_profile")
    runtime_policy = profile.get("runtime_policy")
    if not isinstance(runtime_policy, Mapping):
        raise TypeError("verified profile is missing runtime_policy")
    return hardware_identity_from_hardware_profile(
        hardware_profile,
        runtime_policy=runtime_policy,
    )


def hardware_identity_from_runtime(
    runtime: Mapping[str, Any],
    *,
    device_type: str = "cuda",
) -> HardwareIdentity:
    """Read the nested identity collected immediately before a verified run."""

    gpu = runtime.get("gpu")
    platform_profile = runtime.get("platform")
    software = runtime.get("software")
    runtime_policy = runtime.get("runtime_policy")
    if not isinstance(gpu, Mapping):
        raise TypeError("route identity is missing gpu")
    if not isinstance(platform_profile, Mapping):
        raise TypeError("route identity is missing platform")
    if not isinstance(software, Mapping):
        raise TypeError("route identity is missing software")
    precision, allow_tf32 = _runtime_policy(runtime_policy, "runtime policy")
    return HardwareIdentity(
        device_type=_required_string(device_type, "device_type"),
        device_name=_required_string(gpu.get("name"), "device_name"),
        compute_capability=_required_string(
            gpu.get("compute_capability"), "compute_capability"
        ),
        platform_system=_required_string(
            platform_profile.get("system"), "platform_system"
        ),
        torch=_required_string(software.get("torch"), "torch"),
        cuda_runtime=_required_string(software.get("cuda_runtime"), "cuda_runtime"),
        driver=_required_string(software.get("driver") or "unavailable", "driver"),
        matmul_precision=precision,
        allow_tf32=allow_tf32,
    )


def exact_route_key(
    shape: TransformerShape,
    variant: RunVariant,
    identity: HardwareIdentity,
) -> dict[str, object]:
    """Build the one exact route key shared by promotion and verification."""

    key = make_route_key(
        shape,
        dtype=variant.dtype,
        **identity.as_make_route_key_kwargs(),
    )
    if set(key) != ROUTE_FIELDS:
        missing = ", ".join(sorted(ROUTE_FIELDS - set(key)))
        extra = ", ".join(sorted(set(key) - ROUTE_FIELDS))
        raise ValueError(
            f"exact route key is incomplete; missing={missing}; extra={extra}"
        )
    return key


def workload_route_identity(
    shape: TransformerShape,
    variant: RunVariant,
) -> tuple[tuple[str, object], ...]:
    """Return the runtime-visible workload identity, excluding case metadata."""

    key = make_workload_route_key(shape, dtype=variant.dtype)
    return tuple((field, key[field]) for field in WORKLOAD_ROUTE_FIELDS)


def route_match_from_summary(summary: Mapping[str, Any]) -> dict[str, object]:
    """Build one exact dispatch match from a validated tuning summary."""

    workload = summary.get("workload")
    shape_payload = workload.get("shape") if isinstance(workload, Mapping) else None
    variant_payload = workload.get("variant") if isinstance(workload, Mapping) else None
    if not isinstance(shape_payload, dict):
        raise TypeError("tuning summary workload is missing its shape object")
    if not isinstance(variant_payload, dict):
        raise TypeError("tuning summary workload is missing its run variant")
    shape = TransformerShape.from_dict(shape_payload)
    variant = RunVariant.from_dict(variant_payload)
    profile = summary.get("device_profile")
    if not isinstance(profile, Mapping):
        raise TypeError("tuning summary is missing device_profile")
    return exact_route_key(
        shape,
        variant,
        hardware_identity_from_flat_profile(profile),
    )


def shared_route_groups(
    shapes: Sequence[TransformerShape],
    variant: RunVariant,
) -> tuple[frozenset[str], ...]:
    """Derive shape groups that collapse to the same runtime route key."""

    groups: dict[tuple[tuple[str, object], ...], set[str]] = {}
    for shape in shapes:
        if not isinstance(shape, TransformerShape):
            raise TypeError("shapes must contain TransformerShape values")
        groups.setdefault(workload_route_identity(shape, variant), set()).add(
            shape.case_id
        )
    return tuple(
        frozenset(case_ids) for case_ids in groups.values() if len(case_ids) > 1
    )


def validate_selected_route_groups(
    selected_case_ids: Sequence[str],
    all_shapes: Sequence[TransformerShape],
    variant: RunVariant,
) -> None:
    """Reject a selection that proves only part of one shared runtime route."""

    selected = set(selected_case_ids)
    for required_group in shared_route_groups(all_shapes, variant):
        if selected & required_group and not required_group <= selected:
            missing = ", ".join(sorted(required_group - selected))
            required = ", ".join(sorted(required_group))
            raise ValueError(
                "shared runtime route requires formal calibration for: "
                f"{required}; missing: {missing}"
            )


__all__ = [
    "HardwareIdentity",
    "exact_route_key",
    "hardware_identity_from_flat_profile",
    "hardware_identity_from_hardware_profile",
    "hardware_identity_from_runtime",
    "hardware_identity_from_verified_profile",
    "route_match_from_summary",
    "shared_route_groups",
    "validate_selected_route_groups",
    "workload_route_identity",
]
