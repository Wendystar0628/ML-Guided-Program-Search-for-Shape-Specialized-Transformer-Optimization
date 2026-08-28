"""Build official-compatible models without coupling execution modes."""

from __future__ import annotations

import importlib
import sys
import tempfile
import types
import uuid
from pathlib import Path
from types import ModuleType

import torch
from torch import nn

from official import torch_transformer_benchmark as official
from project_identity import solution_implementation_hash
from runner.contracts import ContractError, MeasurementProtocol, TransformerShape


def load_solution_module(project_root: Path) -> ModuleType:
    """Load the current Solution without depending on the caller's cwd."""

    solution_root = (project_root / "solution").resolve()
    source_path = solution_root / "transformer.py"
    if not source_path.is_file():
        raise ContractError(f"Solution entry file is missing: {source_path}")

    package_name = f"_benchmark_solution_{uuid.uuid4().hex}"
    bytecode_cache = tempfile.TemporaryDirectory(prefix="solution-bytecode-")
    previous_pycache_prefix = sys.pycache_prefix
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.pycache_prefix = bytecode_cache.name
    sys.dont_write_bytecode = True
    package = types.ModuleType(package_name)
    package.__path__ = [str(solution_root)]
    package.__bytecode_cache__ = bytecode_cache
    sys.modules[package_name] = package
    module_name = f"{package_name}.transformer"
    try:
        try:
            module = importlib.import_module(module_name)
        finally:
            sys.pycache_prefix = previous_pycache_prefix
            sys.dont_write_bytecode = previous_dont_write_bytecode
    except BaseException:
        unload_solution_module(package_name)
        bytecode_cache.cleanup()
        raise

    solution_class = getattr(module, "UserOptimizedTransformer", None)
    if not isinstance(solution_class, type) or not issubclass(
        solution_class, nn.Module
    ):
        unload_solution_module(package_name)
        bytecode_cache.cleanup()
        raise ContractError(
            "Solution must export an nn.Module named UserOptimizedTransformer"
        )
    return module


def unload_solution_module(package_name: str) -> None:
    """Remove one dynamically loaded Solution package from ``sys.modules``."""

    for loaded_name in tuple(sys.modules):
        if loaded_name == package_name or loaded_name.startswith(f"{package_name}."):
            sys.modules.pop(loaded_name, None)


def config_for_shape(shape: TransformerShape) -> official.TransformerConfig:
    config = official.TransformerConfig(
        batch_size=shape.batch_size,
        seq_len=shape.seq_len,
        d_model=shape.d_model,
        num_heads=shape.num_heads,
        ffn_dim=shape.ffn_dim,
        num_layers=shape.num_layers,
        causal=shape.causal,
    )
    config.validate()
    return config


def configure_runtime(
    protocol: MeasurementProtocol,
    device: torch.device,
) -> None:
    torch.manual_seed(protocol.seed)
    torch.set_float32_matmul_precision(protocol.matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(protocol.seed)
        torch.backends.cuda.matmul.allow_tf32 = protocol.allow_tf32
        torch.backends.cudnn.allow_tf32 = protocol.allow_tf32


def load_solution_source(project_root: Path) -> tuple[ModuleType, str]:
    source_hash_before_load = solution_implementation_hash(project_root / "solution")
    solution_module = load_solution_module(project_root)
    measured_source_hash = solution_implementation_hash(project_root / "solution")
    if measured_source_hash != source_hash_before_load:
        raise ContractError("Solution source changed while it was being loaded")
    return solution_module, measured_source_hash


def build_solution(
    solution_module: ModuleType,
    config: official.TransformerConfig,
    solution_policy: str | None,
) -> nn.Module:
    solution = solution_module.UserOptimizedTransformer(config)
    if solution_policy is None:
        return solution
    configure = getattr(solution, "configure_runtime_policy", None)
    if not callable(configure):
        raise ContractError(
            "Solution does not expose configure_runtime_policy() for an explicit "
            "solution_policy request"
        )
    configure(policy=solution_policy)
    return solution


__all__ = [
    "build_solution",
    "config_for_shape",
    "configure_runtime",
    "load_solution_module",
    "load_solution_source",
]
