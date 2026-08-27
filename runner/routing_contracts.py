"""Shared adapters for runtime route identities and workload route keys.

The Solution owns the flat dispatch schema.  Runner modules use this file to
translate probe, tuning, and persisted bundle documents into that schema.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from solution.dispatch import (
    ROUTE_FIELDS,
    WORKLOAD_ROUTE_FIELDS,
    make_route_key,
    make_workload_route_key,
)


@dataclass(frozen=True)
class HardwareIdentity:
    """Route-visible hardware and software identity for one runtime."""

    device_type: str
    device_name: str
    compute_capability: str
    platform_system: str
    torch: str
    cuda_runtime: str
    triton: str
    driver: str

    def as_route_fields(self) -> dict[str, str]:
        return {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "platform_system": self.platform_system,
            "torch": self.torch,
            "cuda_runtime": self.cuda_runtime,
            "triton": self.triton,
            "driver": self.driver,
        }

    def as_make_route_key_kwargs(self) -> dict[str, str]:
        return {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "platform_system": self.platform_system,
            "torch_version": self.torch,
            "cuda_runtime": self.cuda_runtime,
            "triton_version": self.triton,
            "driver": self.driver,
        }


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"route identity is missing {name}")
    return value.strip()


def hardware_identity_from_flat_profile(
    profile: Mapping[str, Any],
) -> HardwareIdentity:
    """Read the compact identity embedded in one tuning summary."""

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
        triton=_required_string(profile.get("triton"), "triton"),
        driver=_required_string(profile.get("driver") or "unavailable", "driver"),
    )


def hardware_identity_from_hardware_profile(
    hardware_profile: Mapping[str, Any],
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
        triton=_required_string(software.get("triton"), "triton"),
        driver=_required_string(software.get("driver") or "unavailable", "driver"),
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
    return hardware_identity_from_hardware_profile(hardware_profile)


def hardware_identity_from_runtime(
    runtime: Mapping[str, Any],
    *,
    device_type: str = "cuda",
) -> HardwareIdentity:
    """Read the nested identity collected immediately before a verified run."""

    gpu = runtime.get("gpu")
    platform_profile = runtime.get("platform")
    software = runtime.get("software")
    if not isinstance(gpu, Mapping):
        raise TypeError("route identity is missing gpu")
    if not isinstance(platform_profile, Mapping):
        raise TypeError("route identity is missing platform")
    if not isinstance(software, Mapping):
        raise TypeError("route identity is missing software")
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
        triton=_required_string(software.get("triton"), "triton"),
        driver=_required_string(software.get("driver") or "unavailable", "driver"),
    )


def exact_route_key(
    case: object,
    identity: HardwareIdentity,
) -> dict[str, object]:
    """Build the one exact route key shared by promotion and verification."""

    dtype = case.get("dtype") if isinstance(case, Mapping) else case.dtype
    key = make_route_key(
        case,
        dtype=dtype,
        **identity.as_make_route_key_kwargs(),
    )
    if set(key) != ROUTE_FIELDS:
        missing = ", ".join(sorted(ROUTE_FIELDS - set(key)))
        extra = ", ".join(sorted(set(key) - ROUTE_FIELDS))
        raise ValueError(
            f"exact route key is incomplete; missing={missing}; extra={extra}"
        )
    return key


def workload_route_identity(case: object) -> tuple[tuple[str, object], ...]:
    """Return the runtime-visible workload identity, excluding case metadata."""

    key = make_workload_route_key(case)
    return tuple((field, key[field]) for field in WORKLOAD_ROUTE_FIELDS)


def route_match_from_summary(summary: Mapping[str, Any]) -> dict[str, object]:
    """Build one exact dispatch match from a validated tuning summary."""

    workload = summary.get("workload")
    case = workload.get("case") if isinstance(workload, Mapping) else None
    if not isinstance(case, Mapping):
        raise TypeError("tuning summary workload is missing its case object")
    profile = summary.get("device_profile")
    if not isinstance(profile, Mapping):
        raise TypeError("tuning summary is missing device_profile")
    return exact_route_key(case, hardware_identity_from_flat_profile(profile))


def shared_route_groups(cases: Sequence[object]) -> tuple[frozenset[str], ...]:
    """Derive case groups that collapse to the same runtime route key."""

    groups: dict[tuple[tuple[str, object], ...], set[str]] = {}
    for case in cases:
        case_id = (
            case.get("case_id")
            if isinstance(case, Mapping)
            else getattr(case, "case_id", None)
        )
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("workload case is missing case_id")
        groups.setdefault(workload_route_identity(case), set()).add(case_id)
    return tuple(
        frozenset(case_ids) for case_ids in groups.values() if len(case_ids) > 1
    )


def validate_selected_route_groups(
    selected_case_ids: Sequence[str],
    all_cases: Sequence[object],
) -> None:
    """Reject a selection that proves only part of one shared runtime route."""

    selected = set(selected_case_ids)
    for required_group in shared_route_groups(all_cases):
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
