#!/usr/bin/env python3
"""Run the immutable official benchmark against the current Solution."""

from pathlib import Path

from official import torch_transformer_benchmark as official
from runner.execution import load_solution_module

solution = load_solution_module(Path(__file__).resolve().parent)
official.UserOptimizedTransformer = solution.UserOptimizedTransformer
if hasattr(solution, "copy_model_weights"):
    official.copy_model_weights = solution.copy_model_weights

if __name__ == "__main__":
    raise SystemExit(official.main())
