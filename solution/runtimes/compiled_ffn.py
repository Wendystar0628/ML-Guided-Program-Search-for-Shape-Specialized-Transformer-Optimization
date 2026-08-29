"""Lazy full-graph compilation for one explicit FFN plan."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import TypeAlias

import torch

COMPILED_FFN_MODE = "max-autotune-no-cudagraphs"
FFNFunction: TypeAlias = Callable[[torch.Tensor], torch.Tensor]


@dataclass(slots=True)
class _CompiledEntry:
    function: FFNFunction
    has_run: bool = False


class CompiledFFN:
    """Compile and cache fixed-plan FFNs without an eager fallback.

    ``plan_key`` must change whenever behavior captured by ``function`` changes,
    including when a different layer or set of weights is closed over. The input
    signature is added here so a compiled callable is never reused for another
    tensor contract.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[object, ...], _CompiledEntry] = {}

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
        plan_key: Hashable,
    ) -> tuple[object, ...]:
        try:
            hash(plan_key)
        except TypeError as exc:
            raise TypeError("plan_key must be hashable") from exc
        return (plan_key, cls._tensor_signature(value))

    @staticmethod
    def _create(function: FFNFunction) -> _CompiledEntry:
        compile_function = getattr(torch, "compile", None)
        if not callable(compile_function):
            raise TypeError("torch.compile is unavailable")
        try:
            compiled = compile_function(
                function,
                fullgraph=True,
                dynamic=False,
                mode=COMPILED_FFN_MODE,
            )
        except Exception as exc:
            raise RuntimeError("failed to create compiled FFN") from exc
        return _CompiledEntry(function=compiled)

    def run(
        self,
        function: FFNFunction,
        value: torch.Tensor,
        *,
        plan_key: Hashable,
    ) -> torch.Tensor:
        """Run the compiled FFN for one exact tensor and plan signature."""

        key = self._cache_key(value, plan_key)
        entry = self._entries.get(key)
        if entry is None:
            entry = self._create(function)
            self._entries[key] = entry
        first_execution = not entry.has_run
        try:
            output = entry.function(value)
        except Exception as exc:
            self._entries.pop(key, None)
            stage = "compilation" if first_execution else "execution"
            raise RuntimeError(f"compiled FFN {stage} failed") from exc
        if not isinstance(output, torch.Tensor):
            self._entries.pop(key, None)
            raise TypeError("compiled FFN must return a Tensor")
        if output.shape != value.shape or output.device != value.device:
            self._entries.pop(key, None)
            raise RuntimeError("compiled FFN must preserve input shape and device")
        entry.has_run = True
        return output

    def clear(self) -> None:
        """Discard every cached compiled callable."""

        self._entries.clear()


__all__ = ["COMPILED_FFN_MODE", "CompiledFFN", "FFNFunction"]
