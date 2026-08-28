"""Execution evidence must describe branches that actually ran."""

from __future__ import annotations

from copy import deepcopy

import torch

from official import torch_transformer_benchmark as official
from runner.candidates import CANDIDATE_SPECS
from solution import transformer as transformer_module


def _config() -> official.TransformerConfig:
    return official.TransformerConfig(
        batch_size=1,
        seq_len=4,
        d_model=8,
        num_heads=2,
        ffn_dim=8,
        num_layers=2,
        causal=True,
    )


def test_eager_observation_covers_every_execution_dimension() -> None:
    model = transformer_module.UserOptimizedTransformer(_config()).eval()
    model.configure_runtime_policy(policy="auto")
    model.set_execution_observation(True)

    with torch.inference_mode():
        model(torch.randn(1, 4, 8), torch.ones(1, 4, dtype=torch.bool))

    path = model.describe_execution_path()
    observed = path["observed_execution"]
    assert observed["complete"] is True
    assert observed["attention_backends"] == ["causal_sdpa", "causal_sdpa"]
    assert observed["residual_norm_backends"] == ["torch", "torch"]


def test_graph_fused_norm_evidence_rejects_an_observed_fallback() -> None:
    path = {
        "requested_policy": "graph-fused-norm",
        "selected_policy": "graph-fused-norm",
        "attention_backend": "causal_sdpa",
        "runtime_wrapper": "cuda_graph",
        "residual_norm_backend": "compiled_residual_layer_norm",
        "observed_execution": {
            "complete": True,
            "attention_backends": ["causal_sdpa"],
            "residual_norm_backends": ["torch"],
            "runtime_wrappers": ["cuda_graph"],
        },
    }

    assert not CANDIDATE_SPECS["graph-fused-norm"].evidence_matches(path)


def test_mixed_attention_evidence_rejects_a_native_sdpa_fallback() -> None:
    path = {
        "requested_policy": "mixed-fp16-efficient",
        "selected_policy": "mixed-fp16-efficient",
        "attention_backend": "mixed_fp16_efficient",
        "runtime_wrapper": "eager",
        "residual_norm_backend": "torch",
        "observed_execution": {
            "complete": True,
            "attention_backends": ["mixed_fp16_efficient"],
            "residual_norm_backends": ["torch"],
        },
    }
    fallback = deepcopy(path)
    fallback["observed_execution"]["attention_backends"] = ["causal_sdpa"]

    candidate = CANDIDATE_SPECS["mixed-fp16-efficient"]
    assert candidate.evidence_matches(path)
    assert not candidate.evidence_matches(fallback)
