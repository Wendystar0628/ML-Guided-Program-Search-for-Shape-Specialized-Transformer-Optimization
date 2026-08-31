"""Generate the technical report's publication-style figures from repository data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: I001
from matplotlib.patches import (
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        # Log-axis superscripts render at roughly 0.7x the parent tick size.
        # 7.5 pt keeps every exported glyph above the 5 pt floor.
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 6.3,
        "legend.frameon": False,
        "savefig.facecolor": "white",
    }
)


BLUE = "#155FA0"
TEAL = "#2A9D8F"
ORANGE = "#E07A2D"
INK = "#263238"
MID = "#6B7780"
LIGHT = "#E9EEF2"
PALE_BLUE = "#DDEAF5"
PALE_TEAL = "#DDF1ED"
PALE_ORANGE = "#F8E7D9"

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
RESULT_ROOT = PROJECT_ROOT / "result"
SHAPES_PATH = PROJECT_ROOT / "official" / "test_shapes.json"
DEPLOYMENT_PATH = PROJECT_ROOT / "deployment" / "deployed_configs.json"
SOURCE_DATA = HERE / "source_data"


def _latest_result() -> tuple[Path, dict[str, Any]]:
    candidates = sorted(RESULT_ROOT.glob("*/final_performance.json"), reverse=True)
    for path in candidates:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") == "completed":
            return path, value
    raise FileNotFoundError("no completed final performance report was found")


def _load_shapes() -> list[dict[str, Any]]:
    value = json.loads(SHAPES_PATH.read_text(encoding="utf-8"))
    return list(value["ordered_shapes"])


def _load_deployments() -> dict[str, dict[str, Any]]:
    """Resolve each official Shape to its exact checked-in deployment entry."""

    registry = json.loads(DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    entries = [
        entry
        for bundle in registry["bundles"]
        for entry in bundle["entries"]
    ]
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
    for official_shape in _load_shapes():
        matches = [
            entry
            for entry in entries
            if all(
                entry["shape"][field] == official_shape[field]
                for field in shape_fields
            )
            and entry["shape"].get("dtype") == "float32"
            and entry["shape"].get("padding_ratio") == 0.0
            and entry["shape"].get("input_scale") == 1.0
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one deployment for {official_shape['case_id']}, "
                f"found {len(matches)}"
            )
        resolved[official_shape["case_id"]] = matches[0]
    return resolved


def _estimated_model_flops(shape: dict[str, Any]) -> tuple[int, float]:
    """Return the project's dominant-matmul FLOP estimate and attention share."""

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
    projection_code = {
        "fp16_shadow": "S",
        "autocast_fp16": "A",
        "input_dtype": "I",
    }
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


def _read_csv(name: str) -> list[dict[str, str]]:
    with (SOURCE_DATA / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_source_data(result: dict[str, Any], shapes: list[dict[str, Any]]) -> None:
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    shapes_by_id = {shape["case_id"]: shape for shape in shapes}
    performance_fields = (
        "case_id",
        "baseline_median_ms",
        "deployed_median_ms",
        "deployed_p90_ms",
        "speedup",
        "peak_memory_bytes",
        "correctness_passed",
        "estimated_model_flops",
        "estimated_achieved_tflops",
        "attention_flop_share",
    )
    with (SOURCE_DATA / "performance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=performance_fields)
        writer.writeheader()
        for row in result["shapes"]:
            estimated_flops, attention_share = _estimated_model_flops(
                shapes_by_id[row["case_id"]]
            )
            output = {field: row.get(field) for field in performance_fields}
            output["estimated_model_flops"] = estimated_flops
            output["estimated_achieved_tflops"] = (
                estimated_flops / (float(row["deployed_median_ms"]) * 1e9)
            )
            output["attention_flop_share"] = attention_share
            writer.writerow(output)

    shape_fields = (
        "case_id",
        "batch_size",
        "qkv_dim",
        "heads",
        "seq_len",
        "layers",
        "causal",
        "ffn_dim",
        "total_tokens",
        "attention_elements",
    )
    with (SOURCE_DATA / "workloads.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=shape_fields)
        writer.writeheader()
        for shape in shapes:
            row = dict(shape)
            row["total_tokens"] = shape["batch_size"] * shape["seq_len"]
            row["attention_elements"] = (
                shape["batch_size"]
                * shape["heads"]
                * shape["seq_len"] ** 2
                * shape["layers"]
            )
            writer.writerow({field: row[field] for field in shape_fields})

    deployments = _load_deployments()
    result_config_ids = {
        row["case_id"]: row["config_id"] for row in result["shapes"]
    }
    deployment_fields = (
        "case_id",
        "shape_label",
        "config_id",
        "schedule",
        "attention",
        "layout_bridge",
        "projections",
        "ffn",
        "norms",
        "precision",
    )
    with (SOURCE_DATA / "deployed_programs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=deployment_fields)
        writer.writeheader()
        for shape in shapes:
            case_id = shape["case_id"]
            entry = deployments[case_id]
            row = _display_program(entry)
            row.update(
                {
                    "case_id": case_id,
                    "shape_label": (
                        f"{case_id[-2:]} · B{shape['batch_size']} / S{shape['seq_len']} / "
                        f"D{shape['qkv_dim']} / H{shape['heads']}"
                    ),
                    "config_id": result_config_ids[case_id],
                }
            )
            writer.writerow({field: row[field] for field in deployment_fields})

    shape14 = shapes_by_id["official_14"]
    batch = int(shape14["batch_size"])
    microbatch = 2
    heads = int(shape14["heads"])
    sequence = int(shape14["seq_len"])
    bytes_per_fp32 = 4
    dense_b2 = microbatch * heads * sequence**2 * bytes_per_fp32 / 2**30
    dense_b32 = batch * heads * sequence**2 * bytes_per_fp32 / 2**30
    shape14_result = next(
        row for row in result["shapes"] if row["case_id"] == "official_14"
    )
    memory_rows = (
        ("Measured streamed B2 peak", shape14_result["peak_memory_bytes"] / 2**30, "measured"),
        ("RTX 4080 device capacity", result["hardware"]["total_memory_bytes"] / 2**30, "capacity"),
        ("Dense B2 score tensor", dense_b2, "analytical lower bound"),
        ("Dense B32 score tensor", dense_b32, "analytical context"),
    )
    with (SOURCE_DATA / "shape14_memory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("label", "gib", "evidence_kind"))
        writer.writerows(memory_rows)


def _require_strictly_positive(values: list[float], label: str) -> None:
    """Reject non-positive values before a logarithmic axis is applied."""

    if any(value <= 0 for value in values):
        raise ValueError(f"{label} must contain strictly positive values")


def _save(fig: plt.Figure, stem: str) -> None:
    svg_path = HERE / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.04)
    # Matplotlib writes insignificant trailing spaces inside multiline SVG path
    # data. Normalize them so generated vector assets remain diff-clean.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        HERE / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


def _panel_label(ax: plt.Axes, label: str, *, x: float = -0.075) -> None:
    ax.text(x, 1.02, label, transform=ax.transAxes, fontsize=8, fontweight="bold")


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 6.5,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.15,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MID,
    dashed: bool = False,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.0,
            color=color,
            linestyle="--" if dashed else "-",
            connectionstyle="arc3,rad=0.0",
        )
    )


def make_architecture() -> None:
    fig = plt.figure(figsize=(7.2, 4.0))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.08, 0.92), hspace=0.18, wspace=0.18)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    for ax in (ax_a, ax_b, ax_c):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    _panel_label(ax_a, "a")
    ax_a.set_title("Typed program construction", loc="left", pad=3)
    _box(ax_a, (0.01, 0.59), 0.23, 0.24, "Official shape\n+ run variant", facecolor=LIGHT, edgecolor=MID)
    _box(ax_a, (0.01, 0.20), 0.23, 0.24, "GPU fingerprint\n+ incumbent", facecolor=LIGHT, edgecolor=MID)
    _box(ax_a, (0.34, 0.40), 0.25, 0.29, "Conditional\nprogram space", facecolor=PALE_BLUE, edgecolor=BLUE)
    _box(ax_a, (0.68, 0.40), 0.27, 0.29, "ConfigSpec\n→ PlanBuilder\n→ ExecutionPlan", facecolor=PALE_BLUE, edgecolor=BLUE)
    _arrow(ax_a, (0.24, 0.71), (0.34, 0.58), color=BLUE)
    _arrow(ax_a, (0.24, 0.32), (0.34, 0.50), color=BLUE)
    _arrow(ax_a, (0.59, 0.545), (0.68, 0.545), color=BLUE)
    ax_a.text(0.49, 0.22, "Reject illegal combinations\nbefore GPU execution", ha="center", va="center", fontsize=6.3, color=MID)

    _panel_label(ax_b, "b")
    ax_b.set_title("Evidence-efficient search and promotion", loc="left", pad=3)
    _box(ax_b, (0.01, 0.62), 0.20, 0.20, "Screen", facecolor=PALE_TEAL, edgecolor=TEAL)
    _box(ax_b, (0.29, 0.62), 0.20, 0.20, "Enhanced", facecolor=PALE_TEAL, edgecolor=TEAL)
    _box(ax_b, (0.57, 0.62), 0.20, 0.20, "Formal\npaired blocks", facecolor=PALE_TEAL, edgecolor=TEAL)
    _arrow(ax_b, (0.21, 0.72), (0.29, 0.72), color=TEAL)
    _arrow(ax_b, (0.49, 0.72), (0.57, 0.72), color=TEAL)
    diamond = Polygon(
        ((0.88, 0.82), (0.98, 0.72), (0.88, 0.62), (0.78, 0.72)),
        closed=True,
        facecolor=PALE_ORANGE,
        edgecolor=ORANGE,
        linewidth=1.0,
    )
    ax_b.add_patch(diamond)
    ax_b.text(0.88, 0.72, "≥2%?", ha="center", va="center", fontsize=6.2, color=INK)
    _arrow(ax_b, (0.77, 0.72), (0.78, 0.72), color=ORANGE)
    ax_b.text(0.88, 0.54, "approved → deploy", ha="center", va="center", fontsize=6.2, color=ORANGE)
    _box(ax_b, (0.12, 0.18), 0.32, 0.22, "Single-GPU lease\n+ fresh process", facecolor=LIGHT, edgecolor=MID)
    _box(ax_b, (0.55, 0.18), 0.32, 0.22, "Accuracy + path\n+ latency + memory", facecolor=LIGHT, edgecolor=MID)
    _arrow(ax_b, (0.28, 0.40), (0.18, 0.61), color=MID, dashed=True)
    _arrow(ax_b, (0.71, 0.40), (0.67, 0.61), color=MID, dashed=True)

    _panel_label(ax_c, "c")
    ax_c.set_title("Exact-device deployment closes the loop", loc="left", pad=3)
    _box(ax_c, (0.01, 0.34), 0.18, 0.30, "Formal winner", facecolor=PALE_ORANGE, edgecolor=ORANGE)
    _box(ax_c, (0.27, 0.34), 0.20, 0.30, "Deployment registry\nexact GPU + shape", facecolor=PALE_ORANGE, edgecolor=ORANGE)
    _arrow(ax_c, (0.19, 0.49), (0.27, 0.49), color=ORANGE)
    ax_c.add_patch(
        FancyArrowPatch(
            (0.37, 0.65),
            (0.10, 0.65),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.9,
            color=ORANGE,
            linestyle="--",
            connectionstyle="arc3,rad=0.48",
        )
    )
    ax_c.text(0.235, 0.84, "incumbent for next cycle", ha="center", va="center", fontsize=6.2, color=ORANGE)
    _box(ax_c, (0.56, 0.58), 0.20, 0.25, "Shapes 01–13\nresident runtime", facecolor=PALE_TEAL, edgecolor=TEAL)
    _box(ax_c, (0.56, 0.14), 0.20, 0.25, "Shape 14\nstreamed runtime", facecolor=PALE_ORANGE, edgecolor=ORANGE)
    _arrow(ax_c, (0.47, 0.49), (0.56, 0.70), color=TEAL)
    _arrow(ax_c, (0.47, 0.49), (0.56, 0.26), color=ORANGE)
    _box(ax_c, (0.83, 0.34), 0.16, 0.30, "Official-compatible\nTransformer output", facecolor=LIGHT, edgecolor=MID)
    _arrow(ax_c, (0.76, 0.70), (0.83, 0.53), color=TEAL)
    _arrow(ax_c, (0.76, 0.26), (0.83, 0.45), color=ORANGE)
    ax_c.text(0.34, 0.09, "Resident TPE and Shape-14 finite search retain separate evidence stores", ha="center", va="center", fontsize=6.2, color=MID)

    fig.suptitle(
        "Closed-loop shape-specialized Transformer optimization",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )
    _save(fig, "architecture_overview")


def make_performance(result: dict[str, Any]) -> None:
    resident = [row for row in result["shapes"] if row["speedup"] is not None]
    all_rows = list(result["shapes"])
    labels = [row["case_id"].split("_")[-1] for row in resident]
    geomean = float(result["resident_geomean_speedup"])

    fig = plt.figure(figsize=(7.2, 5.8))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.25), hspace=0.42, wspace=0.28)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    y = list(range(len(resident)))
    speedups = [float(row["speedup"]) for row in resident]
    ax_a.barh(y, speedups, color=BLUE, edgecolor="white", linewidth=0.5)
    ax_a.axvline(geomean, color=ORANGE, linestyle="--", linewidth=1.2)
    ax_a.set_yticks(y, labels)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Speedup (baseline median / deployed median)")
    ax_a.set_xlim(0, 40)
    ax_a.grid(axis="x", color=LIGHT, linewidth=0.6)
    ax_a.set_axisbelow(True)
    for yi, value in zip(y, speedups, strict=True):
        ax_a.text(value + 0.55, yi, f"{value:.1f}×", va="center", fontsize=6.2, color=INK)
    ax_a.text(
        geomean + 0.6,
        12.65,
        f"Geomean {geomean:.2f}×",
        fontsize=6.3,
        color=ORANGE,
        va="bottom",
    )
    ax_a.set_title("Resident-shape speedup", loc="left")
    _panel_label(ax_a, "a")

    x = list(range(len(resident)))
    baseline = [float(row["baseline_median_ms"]) for row in resident]
    deployed = [float(row["deployed_median_ms"]) for row in resident]
    p90 = [float(row["deployed_p90_ms"]) for row in resident]
    _require_strictly_positive(baseline + deployed + p90, "latency")
    for xi, base, dep in zip(x, baseline, deployed, strict=True):
        ax_b.plot((xi, xi), (dep, base), color="#B7BEC4", linewidth=0.8, zorder=1)
    ax_b.scatter(x, baseline, s=14, color=MID, label="Baseline median", zorder=2)
    ax_b.scatter(x, deployed, s=18, color=BLUE, label="Deployed median", zorder=3)
    ax_b.scatter(x, p90, s=18, marker="D", facecolors="white", edgecolors=ORANGE, linewidths=0.9, label="Deployed P90", zorder=4)
    ax_b.set_yscale("log")
    ax_b.set_xticks(x, labels)
    ax_b.set_ylabel("Latency (ms, log scale)")
    ax_b.grid(axis="y", which="major", color=LIGHT, linewidth=0.6)
    ax_b.set_axisbelow(True)
    ax_b.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax_b.set_title("Absolute latency", loc="left")
    _panel_label(ax_b, "b")

    memory_gib = [float(row["peak_memory_bytes"]) / 2**30 for row in all_rows]
    _require_strictly_positive(memory_gib, "peak memory")
    all_labels = [row["case_id"].split("_")[-1] for row in all_rows]
    memory_colors = [BLUE] * 13 + [ORANGE]
    ax_c.scatter(
        range(14),
        memory_gib,
        s=24,
        color=memory_colors,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax_c.set_yscale("log")
    ax_c.set_xticks(range(14), all_labels)
    ax_c.set_ylabel("Peak allocated memory (GiB, log scale)")
    ax_c.grid(axis="y", which="major", color=LIGHT, linewidth=0.6)
    ax_c.set_axisbelow(True)
    shape14 = all_rows[-1]
    ax_c.text(
        12.65,
        memory_gib[-1] / 1.45,
        f"14 · {memory_gib[-1]:.2f} GiB\n{shape14['deployed_median_ms'] / 1000:.2f} s · PASS",
        ha="right",
        va="top",
        fontsize=6.2,
        color=ORANGE,
    )
    ax_c.set_title("Peak memory and streamed feasibility", loc="left")
    _panel_label(ax_c, "c")

    fig.suptitle(
        "RTX 4080 final performance snapshot",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.99,
        0.995,
        "14/14 correctness PASS · Shape 14 excluded from geomean",
        ha="right",
        va="top",
        fontsize=6.3,
        color=MID,
    )
    _save(fig, "performance_summary")


def _slice_panel(
    ax: plt.Axes,
    shapes_by_id: dict[str, dict[str, Any]],
    performance_by_id: dict[str, dict[str, Any]],
    ids: list[str],
    x_field: str,
    title: str,
    *,
    log_x: bool = False,
) -> None:
    x = [float(shapes_by_id[case_id][x_field]) for case_id in ids]
    y = [float(performance_by_id[case_id]["speedup"]) for case_id in ids]
    ax.scatter(x, y, color=BLUE, s=24, edgecolor="white", linewidth=0.6, zorder=3)
    anchor_index = ids.index("official_01")
    ax.scatter(
        [x[anchor_index]],
        [y[anchor_index]],
        s=32,
        facecolor="white",
        edgecolor=ORANGE,
        linewidth=1.1,
        zorder=4,
    )
    if log_x:
        ax.set_xscale("log")
    ax.set_ylim(0, 40)
    ax.set_title(title, loc="left")
    label_by_field = {
        "batch_size": "Batch size",
        "qkv_dim": "QKV dimension",
        "heads": "Heads",
        "seq_len": "Sequence length",
    }
    ax.set_xlabel(label_by_field[x_field])
    ax.grid(axis="y", which="major", color=LIGHT, linewidth=0.55)
    ax.set_axisbelow(True)
    label_indices = {y.index(min(y)), y.index(max(y))}
    for index in sorted(label_indices):
        xi, yi = x[index], y[index]
        ax.annotate(
            f"{yi:.1f}×",
            (xi, yi),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.8,
            color=INK,
        )
    if title == "Batch sweep":
        ax.annotate(
            "Shape 01 reference",
            (x[anchor_index], y[anchor_index]),
            xytext=(-8, -12),
            textcoords="offset points",
            ha="right",
            fontsize=5.8,
            color=ORANGE,
        )


def make_workload_landscape(result: dict[str, Any], shapes: list[dict[str, Any]]) -> None:
    performance_by_id = {row["case_id"]: row for row in result["shapes"]}
    shapes_by_id = {row["case_id"]: row for row in shapes}
    _require_strictly_positive(
        [float(row["seq_len"]) for row in shapes]
        + [float(row["batch_size"] * row["seq_len"]) for row in shapes],
        "workload coordinates",
    )
    fig = plt.figure(figsize=(7.2, 5.0))
    mosaic = [["a", "a", "b", "c"], ["a", "a", "d", "e"]]
    axes = fig.subplot_mosaic(mosaic, gridspec_kw={"wspace": 0.46, "hspace": 0.48})
    ax_a = axes["a"]

    marker_by_heads = {1: "o", 2: "s", 4: "D", 16: "^"}
    for heads, marker in marker_by_heads.items():
        subset = [shape for shape in shapes if shape["heads"] == heads]
        if not subset:
            continue
        ax_a.scatter(
            [shape["seq_len"] for shape in subset],
            [shape["batch_size"] * shape["seq_len"] for shape in subset],
            s=40,
            marker=marker,
            color=[ORANGE if shape["case_id"] == "official_14" else BLUE for shape in subset],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
            label=f"{heads} head{'s' if heads != 1 else ''}",
        )
        for shape in subset:
            if shape["case_id"] in {
                "official_01",
                "official_07",
                "official_08",
                "official_09",
                "official_10",
                "official_11",
            }:
                continue
            ax_a.annotate(
                shape["case_id"].split("_")[-1],
                (shape["seq_len"], shape["batch_size"] * shape["seq_len"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=5.6,
                color=INK,
            )
    ax_a.annotate(
        "01, 07–11\n(shared B and S; D/H vary)",
        (128, 64 * 128),
        xytext=(12, 4),
        textcoords="offset points",
        fontsize=6.0,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": MID, "linewidth": 0.6},
    )
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Sequence length S (log scale)")
    ax_a.set_ylabel("Logical tokens B × S (log scale)")
    ax_a.grid(which="major", color=LIGHT, linewidth=0.55)
    ax_a.set_axisbelow(True)
    ax_a.legend(loc="upper left", ncols=2)
    ax_a.set_title("Official workload regime map", loc="left")
    _panel_label(ax_a, "a")
    ax_a.text(
        0.98,
        0.02,
        "Marker shape = head count\nOrange = streamed Shape 14",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.1,
        color=MID,
    )

    _slice_panel(
        axes["b"],
        shapes_by_id,
        performance_by_id,
        ["official_02", "official_03", "official_04", "official_01", "official_05", "official_06"],
        "batch_size",
        "Batch sweep",
        log_x=True,
    )
    _slice_panel(
        axes["c"],
        shapes_by_id,
        performance_by_id,
        ["official_07", "official_01", "official_08"],
        "qkv_dim",
        "Width sweep",
    )
    _slice_panel(
        axes["d"],
        shapes_by_id,
        performance_by_id,
        ["official_09", "official_10", "official_01", "official_11"],
        "heads",
        "Head-count sweep",
    )
    _slice_panel(
        axes["e"],
        shapes_by_id,
        performance_by_id,
        ["official_12", "official_01", "official_13"],
        "seq_len",
        "Sequence sweep",
        log_x=True,
    )
    for label in ("b", "c", "d", "e"):
        _panel_label(axes[label], label, x=-0.14)
    axes["b"].set_ylabel("Speedup (×)")
    axes["d"].set_ylabel("Speedup (×)")
    axes["c"].set_yticklabels([])
    axes["e"].set_yticklabels([])

    fig.suptitle(
        "Workload diversity motivates shape-specialized search",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        0.955,
        "Sensitivity panels compare independently deployed plans; they are not causal kernel ablations.",
        ha="left",
        va="top",
        fontsize=6.3,
        color=MID,
    )
    _save(fig, "workload_landscape")


def make_useful_throughput() -> None:
    rows = _read_csv("performance.csv")
    labels = [row["case_id"][-2:] for row in rows]
    throughput = [float(row["estimated_achieved_tflops"]) for row in rows]
    work_gflop = [float(row["estimated_model_flops"]) / 1e9 for row in rows]
    attention_share = [float(row["attention_flop_share"]) for row in rows]

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.2, 4.45),
        gridspec_kw={"width_ratios": (0.92, 1.08), "wspace": 0.34},
    )
    y = list(range(len(rows)))
    colors = [BLUE] * 13 + [ORANGE]
    ax_a.barh(y, throughput, color=colors, edgecolor="white", linewidth=0.5)
    ax_a.set_yticks(y, labels)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 90)
    ax_a.set_xlabel("Project-estimated useful throughput (TFLOP/s)")
    ax_a.grid(axis="x", color=LIGHT, linewidth=0.55)
    ax_a.set_axisbelow(True)
    for yi, value in zip(y, throughput, strict=True):
        ax_a.text(value + 1.2, yi, f"{value:.1f}", va="center", fontsize=6.0, color=INK)
    ax_a.axhline(12.5, color=ORANGE, linewidth=0.8, linestyle="--")
    ax_a.set_title("Useful compute rate", loc="left")
    _panel_label(ax_a, "a")

    norm = mcolors.Normalize(vmin=min(attention_share[:-1]), vmax=max(attention_share[:-1]))
    resident_colors = plt.cm.Blues(0.32 + 0.60 * norm(attention_share[:-1]))
    ax_b.scatter(
        work_gflop[:-1],
        throughput[:-1],
        c=resident_colors,
        s=31,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax_b.scatter(
        [work_gflop[-1]],
        [throughput[-1]],
        marker="D",
        s=42,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    ax_b.set_xscale("log")
    ax_b.set_xlabel("Estimated work per call (GFLOP, log scale)")
    ax_b.set_ylabel("Project-estimated useful throughput (TFLOP/s)")
    ax_b.grid(which="major", color=LIGHT, linewidth=0.55)
    ax_b.set_axisbelow(True)
    for label in ("02", "07", "08", "13", "14"):
        idx = labels.index(label)
        offset = {
            "02": (4, 4),
            "07": (4, 4),
            "08": (4, -11),
            "13": (4, 4),
            "14": (-18, -13),
        }[label]
        ax_b.annotate(
            label,
            (work_gflop[idx], throughput[idx]),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.2,
            color=ORANGE if label == "14" else INK,
        )
    scalar = plt.cm.ScalarMappable(norm=norm, cmap="Blues")
    cbar = fig.colorbar(scalar, ax=ax_b, fraction=0.055, pad=0.03)
    cbar.set_label("Attention share of estimated FLOPs", fontsize=6.2)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_ticks((0.0, 0.3, 0.6))
    cbar.set_ticklabels(("0%", "30%", "60%"))
    ax_b.set_title("Scale, mix, and achieved rate", loc="left")
    _panel_label(ax_b, "b", x=-0.13)

    fig.suptitle(
        "Estimated useful compute complements speedup",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        0.955,
        "Dominant matrix-multiply FLOPs ÷ deployed median latency; not executed-instruction FLOPs or official MFU.",
        ha="left",
        va="top",
        fontsize=6.3,
        color=MID,
    )
    _save(fig, "useful_throughput")


def _matrix_fill(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("stream", "tile graph")):
        return PALE_ORANGE
    if any(token in lowered for token in ("triton", "fused", "native", "direct bsd")):
        return PALE_TEAL
    if "torch" in lowered:
        return LIGHT
    return PALE_BLUE


def make_deployed_program_matrix() -> None:
    rows = _read_csv("deployed_programs.csv")
    columns = (
        ("shape_label", "Official Shape", 0.22),
        ("schedule", "Schedule", 0.11),
        ("attention", "Attention", 0.15),
        ("dataflow", "Dataflow + projections¹", 0.20),
        ("ffn_norm", "FFN + initial/residual norm", 0.20),
        ("precision", "Precision", 0.12),
    )
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    title_y = 0.955
    top = 0.89
    bottom = 0.13
    row_h = (top - bottom) / len(rows)
    x_edges = [0.0]
    for _, _, width in columns:
        x_edges.append(x_edges[-1] + width)

    for column_index, (_, heading, _) in enumerate(columns):
        x0, x1 = x_edges[column_index], x_edges[column_index + 1]
        ax.text(
            (x0 + x1) / 2,
            top + 0.025,
            heading,
            ha="center",
            va="bottom",
            fontsize=6.2,
            fontweight="bold",
            color=INK,
        )
    ax.plot((0, 1), (top + 0.012, top + 0.012), color=MID, linewidth=0.8)

    for row_index, row in enumerate(rows):
        y1 = top - row_index * row_h
        y0 = y1 - row_h
        if row["case_id"] == "official_14":
            ax.plot((0, 1), (y1 + 0.004, y1 + 0.004), color=ORANGE, linewidth=1.2)
        for column_index, (field, _, _) in enumerate(columns):
            x0, x1 = x_edges[column_index], x_edges[column_index + 1]
            if field == "dataflow":
                value = f"{row['layout_bridge']}\nProj {row['projections']}"
            elif field == "ffn_norm":
                value = f"{row['ffn']}\n{row['norms']}"
            else:
                value = row[field]
            facecolor = "#F7F9FA" if field == "shape_label" else _matrix_fill(value)
            patch = Rectangle(
                (x0 + 0.002, y0 + 0.002),
                x1 - x0 - 0.004,
                row_h - 0.004,
                facecolor=facecolor,
                edgecolor="white",
                linewidth=0.45,
            )
            ax.add_patch(patch)
            cell_text = ax.text(
                x0 + 0.006 if field == "shape_label" else (x0 + x1) / 2,
                (y0 + y1) / 2,
                value,
                ha="left" if field == "shape_label" else "center",
                va="center",
                fontsize=5.35,
                color=INK,
                linespacing=1.05,
            )
            cell_text.set_clip_path(patch)

    exact_count = len({row["config_id"] for row in rows})
    structural_fields = (
        "schedule",
        "attention",
        "layout_bridge",
        "projections",
        "ffn",
        "norms",
        "precision",
    )
    structural_count = len(
        {tuple(row[field] for field in structural_fields) for row in rows}
    )
    ax.text(
        0,
        0.095,
        f"{len(rows)} Shapes resolve to {exact_count} exact ConfigSpecs and {structural_count} displayed structural signatures.",
        fontsize=6.3,
        color=INK,
        ha="left",
    )
    ax.text(
        0,
        0.055,
        "¹ Projection order: QKV / attention-out / FFN-in / FFN-out; S = FP16 shadow, A = autocast FP16, I = input dtype.",
        fontsize=6.0,
        color=MID,
        ha="left",
    )
    ax.text(
        1,
        0.025,
        "Orange separator = independent Shape-14 streamed regime",
        fontsize=6.0,
        color=ORANGE,
        ha="right",
    )
    ax.text(
        0,
        title_y,
        "Shape-specialized deployed programs on RTX 4080",
        fontsize=10,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
    )
    _save(fig, "deployed_program_matrix")


def make_shape14_streaming() -> None:
    memory_rows = _read_csv("shape14_memory.csv")
    memory = {row["label"]: float(row["gib"]) for row in memory_rows}

    fig = plt.figure(figsize=(7.2, 3.75))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.05, 0.95), wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")
    _panel_label(ax_a, "a")
    ax_a.set_title("Dense materialization versus online streaming", loc="left", pad=3)

    _box(
        ax_a,
        (0.02, 0.59),
        0.39,
        0.23,
        "Supplied dense path\nfull [B, H, S, S] scores",
        facecolor=PALE_ORANGE,
        edgecolor=ORANGE,
        fontsize=6.3,
    )
    _box(
        ax_a,
        (0.59, 0.59),
        0.39,
        0.23,
        "Global softmax + probabilities\nsecond S × S materialization",
        facecolor=PALE_ORANGE,
        edgecolor=ORANGE,
        fontsize=6.3,
    )
    _arrow(ax_a, (0.41, 0.705), (0.59, 0.705), color=ORANGE)
    ax_a.text(0.50, 0.87, "Does not fit a 16 GB-class GPU", ha="center", fontsize=6.3, color=ORANGE)

    _box(
        ax_a,
        (0.02, 0.18),
        0.25,
        0.23,
        "B=2 microbatch\n16 ordered chunks",
        facecolor=PALE_TEAL,
        edgecolor=TEAL,
        fontsize=6.3,
    )
    _box(
        ax_a,
        (0.37, 0.18),
        0.25,
        0.23,
        "64 × 64 Q/KV tiles\nonline row max + sum",
        facecolor=PALE_TEAL,
        edgecolor=TEAL,
        fontsize=6.3,
    )
    _box(
        ax_a,
        (0.72, 0.18),
        0.26,
        0.23,
        "Emit [B, S, D]\ndiscard score tiles",
        facecolor=PALE_TEAL,
        edgecolor=TEAL,
        fontsize=6.3,
    )
    _arrow(ax_a, (0.27, 0.295), (0.37, 0.295), color=TEAL)
    _arrow(ax_a, (0.62, 0.295), (0.72, 0.295), color=TEAL)
    ax_a.text(0.50, 0.08, "No global S × S tensor is stored", ha="center", fontsize=6.3, color=TEAL)

    labels = [
        "Measured streamed B2 peak",
        "Dense B2 score tensor",
        "Dense B32 score tensor",
    ]
    values = [memory[label] for label in labels]
    y = [2, 1, 0]
    ax_b.scatter(
        [values[0]],
        [y[0]],
        s=38,
        color=TEAL,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label="Measured",
    )
    ax_b.scatter(
        values[1:],
        y[1:],
        s=42,
        marker="D",
        facecolors="white",
        edgecolors=ORANGE,
        linewidths=1.1,
        zorder=4,
        label="Analytical score-tensor bound",
    )
    capacity = memory["RTX 4080 device capacity"]
    ax_b.axvline(capacity, color=MID, linewidth=1.0, linestyle="--")
    ax_b.text(capacity * 1.12, 2.35, f"Device capacity\n{capacity:.2f} GiB", fontsize=6.0, color=MID, va="top")
    for yi, value in zip(y, values, strict=True):
        ax_b.text(value * 1.15, yi, f"{value:,.2f} GiB", va="center", fontsize=6.1, color=INK)
    ax_b.set_xscale("log")
    ax_b.set_xlim(3, 40000)
    ax_b.set_ylim(-0.55, 2.55)
    ax_b.set_yticks(y, ["Streamed B2 peak", "Dense B2 score", "Dense B32 score"])
    ax_b.set_xlabel("Allocated or analytical storage (GiB, log scale)")
    ax_b.grid(axis="x", which="major", color=LIGHT, linewidth=0.55)
    ax_b.set_axisbelow(True)
    ax_b.text(
        0.98,
        0.08,
        f"Same-B lower bound: {values[1] / values[0]:.0f}× the measured streamed peak",
        transform=ax_b.transAxes,
        fontsize=6.2,
        color=ORANGE,
        ha="right",
    )
    ax_b.set_title("Capacity evidence", loc="left")
    _panel_label(ax_b, "b", x=-0.13)

    fig.suptitle(
        "Shape 14 requires a separate streamed execution regime",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        0.955,
        "Dense points are one FP32 score-tensor lower bounds; no dense baseline was executed.",
        ha="left",
        va="top",
        fontsize=6.3,
        color=MID,
    )
    _save(fig, "shape14_streaming")


def make_search_evidence() -> None:
    flow_rows = _read_csv("search_flow.csv")
    timing_rows = _read_csv("search_cycle_timing.csv")
    totals = {
        "Screen": sum(int(row["screen_new_trials"]) for row in flow_rows),
        "Enhanced": sum(int(row["enhanced_entries"]) for row in flow_rows),
        "Formal": sum(int(row["formal_comparisons"]) for row in flow_rows),
        "Deployment": sum(int(row["deployment_updates"]) for row in flow_rows),
    }

    fig = plt.figure(figsize=(7.2, 5.1))
    gs = fig.add_gridspec(2, 1, height_ratios=(0.72, 1.28), hspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")
    _panel_label(ax_a, "a", x=-0.025)
    ax_a.set_title("Candidate flow across four Survivor-TPE resident cycles", loc="left", pad=3)

    stages = list(totals)
    stage_colors = (PALE_BLUE, PALE_TEAL, PALE_TEAL, PALE_ORANGE)
    edge_colors = (BLUE, TEAL, TEAL, ORANGE)
    x_positions = (0.015, 0.27, 0.525, 0.78)
    box_w = 0.19
    for index, (stage, x0) in enumerate(zip(stages, x_positions, strict=True)):
        _box(
            ax_a,
            (x0, 0.30),
            box_w,
            0.40,
            f"{stage}\n{totals[stage]:,}",
            facecolor=stage_colors[index],
            edgecolor=edge_colors[index],
            fontsize=8.0,
        )
        if index < len(stages) - 1:
            next_stage = stages[index + 1]
            retention = totals[next_stage] / totals[stage]
            _arrow(
                ax_a,
                (x0 + box_w, 0.50),
                (x_positions[index + 1], 0.50),
                color=MID,
            )
            ax_a.text(
                (x0 + box_w + x_positions[index + 1]) / 2,
                0.62,
                f"{retention:.1%}",
                ha="center",
                fontsize=6.3,
                color=MID,
            )
    ax_a.text(
        0.985,
        0.12,
        f"Screen → deployment: {totals['Deployment'] / totals['Screen']:.2%}",
        ha="right",
        fontsize=6.3,
        color=ORANGE,
    )

    labels = [row["case_id"][-2:] for row in timing_rows]
    y = list(range(len(timing_rows)))
    segments = (
        ("Planning", "planning_seconds", "#C8D0D6"),
        ("Screen", "screen_seconds", BLUE),
        ("Enhanced", "enhanced_seconds", TEAL),
        ("Formal", "formal_seconds", ORANGE),
    )
    left = [0.0] * len(timing_rows)
    for label, field, color in segments:
        values = [float(row[field]) for row in timing_rows]
        ax_b.barh(
            y,
            values,
            left=left,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            label=label,
        )
        left = [old + value for old, value in zip(left, values, strict=True)]
    for yi, total, row in zip(y, left, timing_rows, strict=True):
        if row["decision"] == "published":
            ax_b.scatter(total + 3, yi, marker="*", s=34, color=ORANGE, zorder=4)
            ax_b.text(total + 7, yi, "deploy", va="center", fontsize=6.0, color=ORANGE)
    ax_b.set_yticks(y, labels)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Stage time (s) in one complete resident cycle")
    ax_b.set_ylabel("Official Shape")
    ax_b.grid(axis="x", color=LIGHT, linewidth=0.55)
    ax_b.set_axisbelow(True)
    ax_b.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax_b.set_title("Measurement time is concentrated in Screen", loc="left")
    _panel_label(ax_b, "b", x=-0.025)

    fig.suptitle(
        "Search evidence narrows into measured deployment decisions",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        0.012,
        "Counts are stage entries, not globally unique candidates; Enhanced may reuse compatible cached evidence.",
        ha="left",
        fontsize=6.1,
        color=MID,
    )
    _save(fig, "search_evidence")


def main() -> None:
    result_path, result = _latest_result()
    shapes = _load_shapes()
    _write_source_data(result, shapes)
    make_architecture()
    make_performance(result)
    make_workload_landscape(result, shapes)
    make_useful_throughput()
    make_deployed_program_matrix()
    make_shape14_streaming()
    make_search_evidence()
    print(f"Generated figures from {result_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
