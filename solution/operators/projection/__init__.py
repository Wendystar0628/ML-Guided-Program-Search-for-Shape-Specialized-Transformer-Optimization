"""PyTorch projection and layout compositions."""

from .qkv_layout import can_split_qkv, split_qkv

__all__ = ["can_split_qkv", "split_qkv"]
