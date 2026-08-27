"""Focused tests for the cross-process device measurement lease."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

from runner import locking
from runner.contracts import ContractError


def test_cuda_aliases_are_reentrant_in_the_same_thread(tmp_path: Path) -> None:
    with (
        locking.device_measurement_lease(tmp_path, "cuda"),
        locking.device_measurement_lease(tmp_path, "CUDA:0"),
    ):
        pass

    with locking.device_measurement_lease(tmp_path, "cuda:00"):
        pass
    assert not hasattr(locking, "formal_device_lock_path")


def test_other_thread_cannot_share_an_active_device_lease(tmp_path: Path) -> None:
    failures: list[BaseException] = []

    def compete() -> None:
        try:
            with locking.device_measurement_lease(tmp_path, "cuda:0"):
                pass
        except ContractError as exc:
            failures.append(exc)

    with locking.device_measurement_lease(tmp_path, "cuda"):
        thread = threading.Thread(target=compete)
        thread.start()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], ContractError)
    assert "another thread" in str(failures[0])


def test_other_process_fails_fast_on_the_same_cuda_alias(tmp_path: Path) -> None:
    script = """
import sys
from pathlib import Path
from runner.contracts import ContractError
from runner.locking import device_measurement_lease

try:
    with device_measurement_lease(Path(sys.argv[1]), sys.argv[2]):
        pass
except ContractError:
    raise SystemExit(23)
"""

    with locking.device_measurement_lease(tmp_path, "cuda"):
        blocked = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), "cuda:0"],
            check=False,
        )

    released = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), "cuda:0"],
        check=False,
    )
    assert blocked.returncode == 23
    assert released.returncode == 0


def test_cpu_lease_is_a_no_op(tmp_path: Path) -> None:
    with locking.device_measurement_lease(tmp_path, "cpu"):
        pass

    assert not (tmp_path / "results").exists()
