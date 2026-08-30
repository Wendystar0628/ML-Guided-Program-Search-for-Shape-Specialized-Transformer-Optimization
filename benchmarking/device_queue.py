"""Exclusive GPU scheduling and fresh-process execution helpers."""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Self, TypeVar

from deployment.environment import configure_process_math_mode

T = TypeVar("T")


class DeviceLeaseTimeout(RuntimeError):
    """Raised when the selected device stays occupied past the wait limit."""


class IsolatedProcessError(RuntimeError):
    """Raised when a fresh worker process cannot return a result."""


def _device_key(device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "cuda":
        normalized = "cuda:0"
    return "".join(character if character.isalnum() else "_" for character in normalized)


def _try_lock(stream: BinaryIO) -> bool:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    import fcntl

    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _unlock(stream: BinaryIO) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class DeviceLease:
    """Serialize this project's GPU commands on one logical CUDA device."""

    def __init__(
        self,
        *,
        device: str,
        root: Path,
        timeout_seconds: float | None = None,
        on_wait: Callable[[str], None] | None = None,
    ) -> None:
        self.device = device
        self.root = root
        self.timeout_seconds = timeout_seconds
        self.on_wait = on_wait
        self._stream: BinaryIO | None = None

    @property
    def path(self) -> Path:
        return self.root / f"{_device_key(self.device)}.lock"

    def __enter__(self) -> Self:
        if not self.device.lower().startswith("cuda"):
            return self
        self.root.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        stream = self.path.open("r+b", buffering=0)
        if self.path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        started = time.monotonic()
        announced = False
        try:
            while not _try_lock(stream):
                if not announced and self.on_wait is not None:
                    self.on_wait(
                        f"GPU {self.device} is busy; waiting for the active project run"
                    )
                    announced = True
                if (
                    self.timeout_seconds is not None
                    and time.monotonic() - started >= self.timeout_seconds
                ):
                    raise DeviceLeaseTimeout(
                        f"timed out waiting for GPU {self.device}"
                    )
                time.sleep(0.25)
            self._stream = stream
            metadata = {
                "pid": os.getpid(),
                "device": self.device,
                "acquired_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "command": " ".join(sys.argv),
            }
            stream.seek(0)
            stream.truncate()
            stream.write(
                b"0"
                + json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            )
            stream.flush()
            return self
        except BaseException:
            stream.close()
            raise

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._stream is None:
            return
        stream = self._stream
        self._stream = None
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(b"0")
            stream.flush()
            _unlock(stream)
        finally:
            stream.close()


def _isolated_entry(
    connection: Any,
    target: Callable[..., Any],
    args: tuple[Any, ...],
) -> None:
    try:
        configure_process_math_mode()
        result = target(*args)
        connection.send(("ok", result))
    except BaseException as exc:  # noqa: BLE001 - worker must report all exits
        connection.send(("error", type(exc).__name__, str(exc)[:2000]))
    finally:
        connection.close()


def run_in_fresh_process(target: Callable[..., T], *args: object) -> T:
    """Run one picklable call in a new spawned process and return its result."""

    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_isolated_entry,
        args=(send, target, tuple(args)),
    )
    process.start()
    send.close()
    message: tuple[Any, ...] | None = None
    try:
        while process.is_alive():
            if receive.poll(0.1):
                message = receive.recv()
                break
        if message is None and receive.poll():
            message = receive.recv()
        process.join()
    except KeyboardInterrupt:
        if process.is_alive():
            process.terminate()
        process.join()
        raise
    finally:
        receive.close()

    if message is None:
        raise IsolatedProcessError(
            f"worker exited with code {process.exitcode} without a result"
        )
    if message[0] == "error":
        error_type, error_message = str(message[1]), str(message[2])
        if error_type == "KeyboardInterrupt":
            raise KeyboardInterrupt
        raise IsolatedProcessError(f"{error_type}: {error_message}")
    return message[1]


__all__ = [
    "DeviceLease",
    "DeviceLeaseTimeout",
    "IsolatedProcessError",
    "run_in_fresh_process",
]
