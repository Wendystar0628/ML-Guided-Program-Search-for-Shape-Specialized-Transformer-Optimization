"""Minimal CUDA facts useful to program search and human diagnosis."""

from __future__ import annotations

import torch

from deployment.environment import installed_triton_version


def execute_probe(device: str = "cuda:0") -> dict[str, object]:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("hardware probe requires a CUDA device")

    index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(index)
    major, minor = torch.cuda.get_device_capability(index)
    return {
        "device": f"cuda:{index}",
        "name": torch.cuda.get_device_name(index),
        "compute_capability": f"{major}.{minor}",
        "total_memory_bytes": int(properties.total_memory),
        "multiprocessor_count": int(properties.multi_processor_count),
        "shared_memory_per_block_bytes": int(properties.shared_memory_per_block),
        "warp_size": int(properties.warp_size),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_version": installed_triton_version(),
    }


__all__ = ["execute_probe"]
