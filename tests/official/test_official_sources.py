"""Regression tests for the immutable official inputs supplied with the task."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from official import torch_transformer_benchmark as benchmark

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_official_benchmark_defaults_match_the_updated_contract(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["torch_transformer_benchmark.py"])

    args = benchmark.parse_args()

    assert args.dtype == "float32"
    assert args.padding_ratio == 0.0
    assert args.input_scale == 1.0
    assert args.rtol == 0.02
    assert args.atol == 0.002


def test_shape_rows_contain_only_official_shape_data() -> None:
    document = json.loads(
        (PROJECT_ROOT / "official" / "test_shapes.json").read_text(encoding="utf-8")
    )
    expected_keys = {
        "case_id",
        "batch_size",
        "qkv_dim",
        "heads",
        "seq_len",
        "layers",
        "causal",
        "ffn_dim",
    }

    assert len(document["ordered_shapes"]) == 14
    assert all(set(shape) == expected_keys for shape in document["ordered_shapes"])
    assert document["ordered_shapes"][-1] == {
        "case_id": "official_14",
        "batch_size": 32,
        "qkv_dim": 1024,
        "heads": 16,
        "seq_len": 100000,
        "layers": 2,
        "causal": True,
        "ffn_dim": 1024,
    }
