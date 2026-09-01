"""Refresh compact report CSVs without drawing figures.

The technical-report plots are rendered exclusively by R and Graphviz.  This
script only extracts the latest completed result, official Shapes, and the
checked-in exact-device deployment registry into small, reviewable CSV tables.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
RESULT_ROOT = PROJECT_ROOT / "result"
SHAPES_PATH = PROJECT_ROOT / "official" / "test_shapes.json"
DEPLOYMENT_PATH = PROJECT_ROOT / "deployment" / "deployed_configs.json"
SOURCE_DATA = HERE / "source_data"


def _latest_result() -> dict[str, Any]:
    for path in sorted(RESULT_ROOT.glob("*/final_performance.json"), reverse=True):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "completed":
            return value
    raise FileNotFoundError("no completed final performance report was found")


def _load_shapes() -> list[dict[str, Any]]:
    return list(json.loads(SHAPES_PATH.read_text(encoding="utf-8"))["ordered_shapes"])


def _load_deployments(shapes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    registry = json.loads(DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    entries = [entry for bundle in registry["bundles"] for entry in bundle["entries"]]
    shape_fields = (
        "batch_size",
        "qkv_dim",
        "heads",
        "seq_len",
        "layers",
        "causal",
        "ffn_dim",
    )
    resolved: dict[str, dict[str, Any]] = {}
    for shape in shapes:
        matches = [
            entry
            for entry in entries
            if all(entry["shape"][field] == shape[field] for field in shape_fields)
            and entry["shape"].get("dtype") == "float32"
            and entry["shape"].get("padding_ratio") == 0.0
            and entry["shape"].get("input_scale") == 1.0
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one deployment for {shape['case_id']}, found {len(matches)}")
        resolved[shape["case_id"]] = matches[0]
    return resolved


def _estimated_model_flops(shape: dict[str, Any]) -> tuple[int, float]:
    batch = int(shape["batch_size"])
    sequence = int(shape["seq_len"])
    width = int(shape["qkv_dim"])
    layers = int(shape["layers"])
    ffn_dim = int(shape["ffn_dim"])
    attention_factor = 2 if shape["causal"] else 4
    attention = attention_factor * batch * sequence * sequence * width
    projections = 8 * batch * sequence * width * width
    ffn = 4 * batch * sequence * width * ffn_dim
    total = layers * (attention + projections + ffn)
    return total, (layers * attention) / total


def _display_program(entry: dict[str, Any]) -> dict[str, str]:
    program = entry["config"]["program"]
    schedule = entry["config"]["schedule"]
    runtime = {
        "cuda_graph": "CUDA graph",
        "compiled_forward": "Compiled",
        "batch_tiled_cuda_graph": f"Tile graph {schedule['batch_tile_size']}",
        "eager": "Eager",
        "streamed": f"Streamed mb{schedule['microbatch_size']}",
    }[schedule["runtime"]]
    attention = {
        "fp16_efficient_sdpa": "Efficient SDPA",
        "causal_sdpa": "Causal SDPA",
        "cudnn_sdpa": "cuDNN SDPA",
        "fp16_cudnn_sdpa": "cuDNN SDPA",
        "triton_dh8": "Triton Dh8",
        "triton_shape13": "Triton S1024",
        "triton_streaming_dh64": "Triton stream Dh64",
    }.get(program["attention"], program["attention"])
    layout = {
        ("triton_native_bhsd", "triton_bhsd_projection"): "Native → Triton O",
        ("triton_native_bhsd", "attention_direct_bsd"): "Native → direct BSD",
        ("triton_native_bhsd", "torch_bhsd_to_bsd"): "Native → Torch O",
        ("view", "torch_bhsd_to_bsd"): "View → Torch O",
        ("view", "attention_direct_bsd"): "View → direct BSD",
    }.get(
        (program["qkv_materialization"], program["attention_output_bridge"]),
        f"{program['qkv_materialization']} → {program['attention_output_bridge']}",
    )
    projection_code = {"fp16_shadow": "S", "autocast_fp16": "A", "input_dtype": "I"}
    projections = "".join(
        projection_code.get(program[field], "?")
        for field in (
            "qkv_projection",
            "attention_output_projection",
            "ffn_input_projection",
            "ffn_output_projection",
        )
    )
    ffn = {
        "triton_linear_exact_gelu": "Linear + GELU",
        "triton_fused_boundary": "Fused boundary",
        "triton_fused_mlp_boundary": "Fused boundary",
        "compiled": "Compiled",
        "torch": "Torch",
    }.get(program["ffn"], program["ffn"])
    norm_name = {
        "triton_fp16": "T16",
        "triton_mixed": "T-mixed",
        "triton_linear_mixed": "Lin-mixed",
        "triton_fused_qkv": "Fused QKV",
        "torch": "Torch",
    }
    norms = (
        f"{norm_name.get(program['initial_norm'], program['initial_norm'])} / "
        f"{norm_name.get(program['residual_norm'], program['residual_norm'])}"
    )
    precision = {
        "fp16_core": "FP16 core",
        "fp16_attention_and_ffn_input": "Attn + FFN-in FP16",
    }.get(program["precision_plan"], program["precision_plan"])
    return {
        "schedule": runtime,
        "attention": attention,
        "layout_bridge": layout,
        "projections": projections,
        "ffn": ffn,
        "norms": norms,
        "precision": precision,
    }


def _write_csv(name: str, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with (SOURCE_DATA / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fieldnames} for row in rows)


def main() -> None:
    result = _latest_result()
    shapes = _load_shapes()
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    shapes_by_id = {shape["case_id"]: shape for shape in shapes}

    performance_rows: list[dict[str, Any]] = []
    for row in result["shapes"]:
        estimated_flops, attention_share = _estimated_model_flops(shapes_by_id[row["case_id"]])
        performance_rows.append(
            {
                **row,
                "estimated_model_flops": estimated_flops,
                "estimated_achieved_tflops": estimated_flops / (float(row["deployed_median_ms"]) * 1e9),
                "attention_flop_share": attention_share,
            }
        )
    _write_csv(
        "performance.csv",
        (
            "case_id", "baseline_median_ms", "deployed_median_ms", "deployed_p90_ms",
            "speedup", "peak_memory_bytes", "correctness_passed",
            "local_b1_semantic_pass", "full_logical_execution_completed",
            "official_b32_io_pass", "official_b32_io_status", "estimated_model_flops",
            "estimated_achieved_tflops", "attention_flop_share",
        ),
        performance_rows,
    )

    workload_rows = [
        {
            **shape,
            "total_tokens": shape["batch_size"] * shape["seq_len"],
            "attention_elements": shape["batch_size"] * shape["heads"] * shape["seq_len"] ** 2 * shape["layers"],
        }
        for shape in shapes
    ]
    _write_csv(
        "workloads.csv",
        (
            "case_id", "batch_size", "qkv_dim", "heads", "seq_len", "layers",
            "causal", "ffn_dim", "total_tokens", "attention_elements",
        ),
        workload_rows,
    )

    deployments = _load_deployments(shapes)
    config_ids = {row["case_id"]: row["config_id"] for row in result["shapes"]}
    program_rows: list[dict[str, Any]] = []
    for shape in shapes:
        case_id = shape["case_id"]
        program_rows.append(
            {
                "case_id": case_id,
                "shape_label": f"{case_id[-2:]} · B{shape['batch_size']} / S{shape['seq_len']} / D{shape['qkv_dim']} / H{shape['heads']}",
                "config_id": config_ids[case_id],
                **_display_program(deployments[case_id]),
            }
        )
    _write_csv(
        "deployed_programs.csv",
        (
            "case_id", "shape_label", "config_id", "schedule", "attention", "layout_bridge",
            "projections", "ffn", "norms", "precision",
        ),
        program_rows,
    )

    shape14 = shapes_by_id["official_14"]
    sequence = int(shape14["seq_len"])
    heads = int(shape14["heads"])
    shape14_result = next(row for row in result["shapes"] if row["case_id"] == "official_14")
    memory_rows = [
        {"label": "Measured streamed B2 peak", "gib": shape14_result["peak_memory_bytes"] / 2**30, "evidence_kind": "measured"},
        {"label": "RTX 4080 device capacity", "gib": result["hardware"]["total_memory_bytes"] / 2**30, "evidence_kind": "capacity"},
        {"label": "Dense B2 score tensor", "gib": 2 * heads * sequence**2 * 4 / 2**30, "evidence_kind": "analytical lower bound"},
        {"label": "Dense B32 score tensor", "gib": int(shape14["batch_size"]) * heads * sequence**2 * 4 / 2**30, "evidence_kind": "analytical context"},
    ]
    _write_csv("shape14_memory.csv", ("label", "gib", "evidence_kind"), memory_rows)


if __name__ == "__main__":
    main()
