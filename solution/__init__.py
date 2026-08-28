"""Lazy public solution contract used by the official benchmark."""

from __future__ import annotations

from typing import Any

__all__ = ["UserOptimizedTransformer", "copy_model_weights"]


def __getattr__(name: str) -> Any:
    """Load the submitted Transformer only when the public entry is requested."""

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .transformer import UserOptimizedTransformer, copy_model_weights

    exports = {
        "UserOptimizedTransformer": UserOptimizedTransformer,
        "copy_model_weights": copy_model_weights,
    }
    globals().update(exports)
    return exports[name]
