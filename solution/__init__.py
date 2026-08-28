"""Public solution contract used by the official benchmark."""

from .transformer import UserOptimizedTransformer, copy_model_weights

__all__ = ["UserOptimizedTransformer", "copy_model_weights"]
