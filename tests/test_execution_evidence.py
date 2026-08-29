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
    model.configure_runtime_policy(policy="eager-sdpa")
    model.set_execution_observation(True)

    with torch.inference_mode():
        model(torch.randn(1, 4, 8), torch.ones(1, 4, dtype=torch.bool))

    path = model.describe_execution_path()
    observed = path["observed_execution"]
    assert observed["complete"] is True
    assert observed["attention_backends"] == ["causal_sdpa", "causal_sdpa"]
    assert observed["residual_norm_backends"] == [
        "torch",
        "torch",
        "torch",
        "torch",
    ]


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


def test_mixed_core_evidence_rejects_linear_or_dtype_fallback() -> None:
    path = {
        "requested_policy": "mixed-fp16-core-efficient",
        "selected_policy": "mixed-fp16-core-efficient",
        "attention_backend": "mixed_fp16_efficient",
        "attention_compute_dtype": "float16",
        "linear_backend": "autocast_fp16",
        "linear_compute_dtype": "float16",
        "runtime_wrapper": "eager",
        "residual_norm_backend": "torch",
        "observed_execution": {
            "complete": True,
            "attention_backends": ["mixed_fp16_efficient"],
            "attention_compute_dtypes": ["float16"],
            "linear_backends": ["autocast_fp16"],
            "linear_compute_dtypes": ["float16"],
            "residual_norm_backends": ["torch"],
        },
    }
    linear_fallback = deepcopy(path)
    linear_fallback["observed_execution"]["linear_backends"] = ["torch"]
    dtype_fallback = deepcopy(path)
    dtype_fallback["observed_execution"]["linear_compute_dtypes"] = ["float32"]

    candidate = CANDIDATE_SPECS["mixed-fp16-core-efficient"]
    assert candidate.evidence_matches(path)
    assert not candidate.evidence_matches(linear_fallback)
    assert not candidate.evidence_matches(dtype_fallback)


def test_composite_evidence_rejects_residual_norm_fallbacks() -> None:
    triton_path = {
        "requested_policy": "mixed-fp16-core-efficient-triton-norm",
        "selected_policy": "mixed-fp16-core-efficient-triton-norm",
        "attention_backend": "mixed_fp16_efficient",
        "attention_compute_dtype": "float16",
        "linear_backend": "autocast_fp16",
        "linear_compute_dtype": "float16",
        "runtime_wrapper": "eager",
        "residual_norm_backend": "triton_residual_layer_norm",
        "observed_execution": {
            "complete": True,
            "attention_backends": ["mixed_fp16_efficient"],
            "attention_compute_dtypes": ["float16"],
            "linear_backends": ["autocast_fp16"],
            "linear_compute_dtypes": ["float16"],
            "residual_norm_backends": ["triton_residual_layer_norm"],
        },
    }
    compiled_graph_path = {
        "requested_policy": "graph-mixed-fp16-efficient-compiled-norm",
        "selected_policy": "graph-mixed-fp16-efficient-compiled-norm",
        "attention_backend": "mixed_fp16_efficient",
        "runtime_wrapper": "cuda_graph",
        "residual_norm_backend": "compiled_residual_layer_norm",
        "observed_execution": {
            "complete": True,
            "attention_backends": ["mixed_fp16_efficient"],
            "residual_norm_backends": ["compiled_residual_layer_norm"],
            "runtime_wrappers": ["cuda_graph"],
        },
    }
    mixed_core_compiled_graph_path = {
        "requested_policy": "graph-mixed-fp16-core-efficient-compiled-norm",
        "selected_policy": "graph-mixed-fp16-core-efficient-compiled-norm",
        "attention_backend": "mixed_fp16_efficient",
        "attention_compute_dtype": "float16",
        "linear_backend": "autocast_fp16",
        "linear_compute_dtype": "float16",
        "runtime_wrapper": "cuda_graph",
        "residual_norm_backend": "compiled_residual_layer_norm",
        "observed_execution": {
            "complete": True,
            "attention_backends": ["mixed_fp16_efficient"],
            "attention_compute_dtypes": ["float16"],
            "linear_backends": ["autocast_fp16"],
            "linear_compute_dtypes": ["float16"],
            "residual_norm_backends": ["compiled_residual_layer_norm"],
            "runtime_wrappers": ["cuda_graph"],
        },
    }
    batch_tiled_path = deepcopy(mixed_core_compiled_graph_path)
    batch_tiled_path.update(
        {
            "requested_policy": ("batch-tiled-mixed-fp16-core-efficient-compiled-norm"),
            "selected_policy": ("batch-tiled-mixed-fp16-core-efficient-compiled-norm"),
            "runtime_wrapper": "batch_tiled_cuda_graph",
            "batch_tile_size": 128,
            "use_triton_initial_fp16_norm": False,
        }
    )
    batch_tiled_path["observed_execution"]["runtime_wrappers"] = [
        "batch_tiled_cuda_graph"
    ]
    mixed_residual_path = deepcopy(batch_tiled_path)
    mixed_residual_path.update(
        {
            "requested_policy": "batch-tiled-shape06-triton-mixed-norm-fp16-shadow",
            "selected_policy": "batch-tiled-shape06-triton-mixed-norm-fp16-shadow",
            "linear_backend": "fp16_shadow",
            "residual_norm_backend": "triton_mixed_residual_layer_norm",
            "use_triton_initial_fp16_norm": True,
        }
    )
    mixed_residual_path["observed_execution"]["linear_backends"] = ["fp16_shadow"]
    mixed_residual_path["observed_execution"]["residual_norm_backends"] = [
        "triton_mixed_residual_layer_norm"
    ]

    for candidate_id, path in (
        ("mixed-fp16-core-efficient-triton-norm", triton_path),
        ("graph-mixed-fp16-efficient-compiled-norm", compiled_graph_path),
        (
            "graph-mixed-fp16-core-efficient-compiled-norm",
            mixed_core_compiled_graph_path,
        ),
        (
            "batch-tiled-mixed-fp16-core-efficient-compiled-norm",
            batch_tiled_path,
        ),
        (
            "batch-tiled-shape06-triton-mixed-norm-fp16-shadow",
            mixed_residual_path,
        ),
    ):
        candidate = CANDIDATE_SPECS[candidate_id]
        assert candidate.evidence_matches(path)
        fallback = deepcopy(path)
        fallback["observed_execution"]["residual_norm_backends"] = ["torch"]
        assert not candidate.evidence_matches(fallback)

    runtime_fallback = deepcopy(batch_tiled_path)
    runtime_fallback["observed_execution"]["runtime_wrappers"] = ["cuda_graph"]
    assert not CANDIDATE_SPECS[
        "batch-tiled-mixed-fp16-core-efficient-compiled-norm"
    ].evidence_matches(runtime_fallback)

    initial_norm_fallback = deepcopy(mixed_residual_path)
    initial_norm_fallback["use_triton_initial_fp16_norm"] = False
    assert not CANDIDATE_SPECS[
        "batch-tiled-shape06-triton-mixed-norm-fp16-shadow"
    ].evidence_matches(initial_norm_fallback)


def test_compiled_forward_evidence_requires_inner_and_outer_execution() -> None:
    path = {
        "requested_policy": "compiled-mixed-fp16-core-efficient",
        "selected_policy": "compiled-mixed-fp16-core-efficient",
        "attention_backend": "mixed_fp16_efficient",
        "attention_compute_dtype": "float16",
        "linear_backend": "autocast_fp16",
        "linear_compute_dtype": "float16",
        "runtime_wrapper": "compiled_forward",
        "compile_mode": "max-autotune",
        "residual_norm_backend": "torch",
        "observed_execution": {
            "complete": True,
            "attention_backends": ["mixed_fp16_efficient"],
            "attention_compute_dtypes": ["float16"],
            "linear_backends": ["autocast_fp16"],
            "linear_compute_dtypes": ["float16"],
            "residual_norm_backends": ["torch"],
            "runtime_wrappers": ["compiled_forward"],
        },
    }
    candidate = CANDIDATE_SPECS["compiled-mixed-fp16-core-efficient"]

    assert candidate.evidence_matches(path)
    missing_wrapper = deepcopy(path)
    del missing_wrapper["observed_execution"]["runtime_wrappers"]
    assert not candidate.evidence_matches(missing_wrapper)
    wrong_inner = deepcopy(path)
    wrong_inner["observed_execution"]["attention_backends"] = ["causal_sdpa"]
    assert not candidate.evidence_matches(wrong_inner)
    wrong_mode = deepcopy(path)
    wrong_mode["compile_mode"] = "max-autotune-no-cudagraphs"
    assert not candidate.evidence_matches(wrong_mode)


def test_shape13_triton_evidence_rejects_an_efficient_attention_fallback() -> None:
    policy = "compiled-shape13-triton-attention-fp16-shadow"
    path = {
        "requested_policy": policy,
        "selected_policy": policy,
        "attention_backend": "triton_shape13_causal_attention",
        "attention_compute_dtype": "float16",
        "linear_backend": "fp16_shadow",
        "linear_compute_dtype": "float16",
        "runtime_wrapper": "compiled_forward",
        "compile_mode": "max-autotune-no-cudagraphs",
        "residual_norm_backend": "torch",
        "observed_execution": {
            "complete": True,
            "attention_backends": ["triton_shape13_causal_attention"],
            "attention_compute_dtypes": ["float16"],
            "linear_backends": ["fp16_shadow"],
            "linear_compute_dtypes": ["float16"],
            "residual_norm_backends": ["torch"],
            "runtime_wrappers": ["compiled_forward"],
        },
    }
    candidate = CANDIDATE_SPECS[policy]

    assert candidate.evidence_matches(path)
    fallback = deepcopy(path)
    fallback["observed_execution"]["attention_backends"] = ["mixed_fp16_efficient"]
    assert not candidate.evidence_matches(fallback)
    shadow_fallback = deepcopy(path)
    shadow_fallback["observed_execution"]["linear_backends"] = ["autocast_fp16"]
    assert not candidate.evidence_matches(shadow_fallback)
    wrong_mode = deepcopy(path)
    wrong_mode["compile_mode"] = "max-autotune"
    assert not candidate.evidence_matches(wrong_mode)


def test_shape11_dh8_evidence_rejects_attention_or_shadow_fallbacks() -> None:
    policy = "compiled-shape11-dh8-triton-fp16-shadow"
    path = {
        "requested_policy": policy,
        "selected_policy": policy,
        "attention_backend": "triton_dh8_causal_attention_bsd",
        "attention_compute_dtype": "float16",
        "attention_output_layout": "bsd",
        "linear_backend": "fp16_shadow",
        "linear_compute_dtype": "float16",
        "runtime_wrapper": "compiled_forward",
        "compile_mode": "max-autotune",
        "residual_norm_backend": "torch",
        "observed_execution": {
            "complete": True,
            "attention_backends": ["triton_dh8_causal_attention_bsd"],
            "attention_compute_dtypes": ["float16"],
            "linear_backends": ["fp16_shadow"],
            "linear_compute_dtypes": ["float16"],
            "residual_norm_backends": ["torch"],
            "runtime_wrappers": ["compiled_forward"],
        },
    }
    candidate = CANDIDATE_SPECS[policy]

    assert candidate.evidence_matches(path)
    attention_fallback = deepcopy(path)
    attention_fallback["observed_execution"]["attention_backends"] = [
        "mixed_fp16_efficient"
    ]
    assert not candidate.evidence_matches(attention_fallback)
    shadow_fallback = deepcopy(path)
    shadow_fallback["observed_execution"]["linear_backends"] = ["autocast_fp16"]
    assert not candidate.evidence_matches(shadow_fallback)


def test_graph_shadow_evidence_rejects_autocast_weight_fallbacks() -> None:
    paths = {
        "graph-fp16-shadow-efficient-compiled-norm": {
            "residual_norm_backend": "compiled_residual_layer_norm",
        },
        "graph-fp16-shadow-efficient-triton-mixed-norm-reuse-input": {
            "residual_norm_backend": "triton_mixed_residual_layer_norm",
            "reuse_unchanged_input": True,
        },
    }

    for policy, additions in paths.items():
        path = {
            "requested_policy": policy,
            "selected_policy": policy,
            "attention_backend": "mixed_fp16_efficient",
            "attention_compute_dtype": "float16",
            "linear_backend": "fp16_shadow",
            "linear_compute_dtype": "float16",
            "runtime_wrapper": "cuda_graph",
            **additions,
            "observed_execution": {
                "complete": True,
                "attention_backends": ["mixed_fp16_efficient"],
                "attention_compute_dtypes": ["float16"],
                "linear_backends": ["fp16_shadow"],
                "linear_compute_dtypes": ["float16"],
                "residual_norm_backends": [additions["residual_norm_backend"]],
                "runtime_wrappers": ["cuda_graph"],
            },
        }
        candidate = CANDIDATE_SPECS[policy]
        assert candidate.evidence_matches(path)

        fallback = deepcopy(path)
        fallback["observed_execution"]["linear_backends"] = ["autocast_fp16"]
        assert not candidate.evidence_matches(fallback)
