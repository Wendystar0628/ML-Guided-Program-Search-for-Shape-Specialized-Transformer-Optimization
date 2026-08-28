"""Lazy full-stack compilation for one explicit execution plan."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import TypeAlias

import torch

ForwardFunction: TypeAlias = Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]

_COMPILE_MODE = "max-autotune-no-cudagraphs"


@dataclass(slots=True)
class _CompiledEntry:
    function: ForwardFunction
    has_run: bool = False


class CompiledForward:
    """Compile and cache fixed-plan forwards without an eager fallback.

    ``plan_key`` is supplied by the caller and must change whenever captured
    execution behavior changes. Input and mask signatures are added here so a
    compiled callable is never reused for a different tensor contract.
    """

    def __init__(self, *, mode: str = _COMPILE_MODE) -> None:
        if not mode:
            raise ValueError("compile mode must not be empty")
        self._mode = mode
        self._entries: dict[tuple[object, ...], _CompiledEntry] = {}

    @property
    def mode(self) -> str:
        """Return the configured ``torch.compile`` mode."""

        return self._mode

    @property
    def cache_size(self) -> int:
        """Return the number of compiled input and plan signatures."""

        return len(self._entries)

    @staticmethod
    def _tensor_signature(value: torch.Tensor) -> tuple[object, ...]:
        return (
            value.device,
            value.dtype,
            tuple(value.shape),
            tuple(value.stride()),
        )

    @classmethod
    def _cache_key(
        cls,
        value: torch.Tensor,
        valid_mask: torch.Tensor | None,
        plan_key: Hashable,
    ) -> tuple[object, ...]:
        try:
            hash(plan_key)
        except TypeError as exc:
            raise TypeError("plan_key must be hashable") from exc
        mask_signature = (
            None if valid_mask is None else cls._tensor_signature(valid_mask)
        )
        return (plan_key, cls._tensor_signature(value), mask_signature)

    def _create(self, function: ForwardFunction) -> _CompiledEntry:
        compile_function = getattr(torch, "compile", None)
        if not callable(compile_function):
            raise TypeError("torch.compile is unavailable")
        try:
            compiled = compile_function(
                function,
                fullgraph=True,
                dynamic=False,
                mode=self._mode,
            )
        except Exception as exc:
            raise RuntimeError("failed to create full-stack compiled forward") from exc
        return _CompiledEntry(function=compiled)

    def run(
        self,
        function: ForwardFunction,
        value: torch.Tensor,
        valid_mask: torch.Tensor | None,
        *,
        plan_key: Hashable,
    ) -> torch.Tensor:
        """Run the compiled callable for one exact tensor and plan signature."""

        key = self._cache_key(value, valid_mask, plan_key)
        entry = self._entries.get(key)
        if entry is None:
            entry = self._create(function)
            self._entries[key] = entry
        first_execution = not entry.has_run
        try:
            output = entry.function(value, valid_mask)
        except Exception as exc:
            self._entries.pop(key, None)
            stage = "compilation" if first_execution else "execution"
            raise RuntimeError(f"full-stack compiled forward {stage} failed") from exc
        if not isinstance(output, torch.Tensor):
            self._entries.pop(key, None)
            raise TypeError("full-stack compiled forward must return a Tensor")
        if (
            output.shape != value.shape
            or output.device != value.device
            or output.dtype != value.dtype
        ):
            self._entries.pop(key, None)
            raise RuntimeError(
                "full-stack compiled forward must preserve input shape, device, and dtype"
            )
        entry.has_run = True
        return output

    def rebuild(
        self,
        function: ForwardFunction,
        value: torch.Tensor,
        valid_mask: torch.Tensor | None,
        *,
        plan_key: Hashable,
    ) -> torch.Tensor:
        """Discard and rebuild one exact tensor and plan signature."""

        key = self._cache_key(value, valid_mask, plan_key)
        self._entries.pop(key, None)
        return self.run(
            function,
            value,
            valid_mask,
            plan_key=plan_key,
        )

    def clear(self) -> None:
        """Discard every cached compiled callable."""

        self._entries.clear()


__all__ = ["CompiledForward", "ForwardFunction"]
