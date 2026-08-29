#!/usr/bin/env python3
"""Run the immutable official benchmark against the current Solution."""

from deployment.environment import configure_process_math_mode
from official import torch_transformer_benchmark as official
from solution import UserOptimizedTransformer, copy_model_weights

official.UserOptimizedTransformer = UserOptimizedTransformer
official.copy_model_weights = copy_model_weights

if __name__ == "__main__":
    configure_process_math_mode()
    raise SystemExit(official.main())
