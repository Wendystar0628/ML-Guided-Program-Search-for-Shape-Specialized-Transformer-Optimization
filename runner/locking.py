"""Minimal cross-process locks for exclusive GPU measurements and publication."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from runner.contracts import ContractError
from runner.result_layout import intermediate_results_dir


@dataclass
class _HeldDeviceLease:
    process_id: int
    thread_id: int
    depth: int
    handle: BinaryIO


_DEVICE_LEASES: dict[Path, _HeldDeviceLease] = {}
_DEVICE_LEASES_GUARD = threading.Lock()


def _try_lock(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise BlockingIOError from exc
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def exclusive_file_lock(path: Path, *, purpose: str) -> Iterator[None]:
    """Fail fast when another process already owns the same operation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
        try:
            _try_lock(handle)
        except BlockingIOError as exc:
            raise ContractError(
                f"another process is already running {purpose}"
            ) from exc
        try:
            yield
        finally:
            _unlock(handle)


def _normalized_cuda_device(device: str) -> str | None:
    normalized = device.strip().lower()
    if not normalized.startswith("cuda"):
        return None
    if normalized == "cuda":
        return "cuda:0"
    prefix, separator, raw_index = normalized.partition(":")
    if prefix != "cuda" or separator != ":" or not raw_index.isdigit():
        raise ContractError(f"invalid CUDA device for measurement lease: {device!r}")
    return f"cuda:{int(raw_index)}"


def _device_measurement_lock_path(
    project_root: Path,
    normalized_device: str,
) -> Path:
    safe_device = normalized_device.replace(":", "_")
    return (
        intermediate_results_dir(
            project_root,
            ".locks",
        )
        / f"device_{safe_device}.lock"
    )


@contextmanager
def device_measurement_lease(
    project_root: Path,
    device: str,
    *,
    purpose: str = "GPU measurement",
) -> Iterator[None]:
    """Exclusively own one CUDA device; CPU and non-CUDA devices are no-ops."""

    normalized_device = _normalized_cuda_device(device)
    if normalized_device is None:
        yield
        return

    path = _device_measurement_lock_path(project_root, normalized_device)
    process_id = os.getpid()
    thread_id = threading.get_ident()
    with _DEVICE_LEASES_GUARD:
        held = _DEVICE_LEASES.get(path)
        if held is not None and held.process_id != process_id:
            held.handle.close()
            del _DEVICE_LEASES[path]
            held = None
        if held is not None:
            if held.thread_id != thread_id:
                raise ContractError(
                    f"another thread is already running {purpose} on "
                    f"{normalized_device}"
                )
            held.depth += 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+b")
            try:
                if path.stat().st_size == 0:
                    handle.write(b"\0")
                    handle.flush()
                _try_lock(handle)
            except BlockingIOError as exc:
                handle.close()
                raise ContractError(
                    f"another process is already running {purpose} on "
                    f"{normalized_device}"
                ) from exc
            except BaseException:
                handle.close()
                raise
            _DEVICE_LEASES[path] = _HeldDeviceLease(
                process_id=process_id,
                thread_id=thread_id,
                depth=1,
                handle=handle,
            )

    try:
        yield
    finally:
        with _DEVICE_LEASES_GUARD:
            held = _DEVICE_LEASES.get(path)
            if (
                held is None
                or held.process_id != process_id
                or held.thread_id != thread_id
            ):
                raise RuntimeError("device measurement lease ownership changed")
            held.depth -= 1
            if held.depth == 0:
                try:
                    _unlock(held.handle)
                finally:
                    held.handle.close()
                    del _DEVICE_LEASES[path]


def bundle_lock_path(bundle_root: Path) -> Path:
    """Keep publication locks in the project's intermediate artifact root."""

    resolved_bundle = bundle_root.resolve()
    project_root = resolved_bundle.parent
    for ancestor in resolved_bundle.parents:
        if ancestor.name == "verified_hardware":
            project_root = ancestor.parent
            break
    return (
        intermediate_results_dir(
            project_root,
            ".locks",
        )
        / f"publish_{resolved_bundle.name}.lock"
    )


def hardware_bundle_lock_path(project_root: Path, hardware_id: str) -> Path:
    """Serialize discovery and creation for one stable hardware directory."""

    return (
        intermediate_results_dir(
            project_root,
            ".locks",
        )
        / f"catalog_{hardware_id}.lock"
    )


__all__ = [
    "bundle_lock_path",
    "device_measurement_lease",
    "exclusive_file_lock",
    "hardware_bundle_lock_path",
]
