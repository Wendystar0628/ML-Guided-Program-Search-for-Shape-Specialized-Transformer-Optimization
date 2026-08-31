"""Generate the technical report's publication-style figures from repository data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: I001
from matplotlib.patches import (
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
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
        "legend.fontsize": 6,
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


def _write_source_data(result: dict[str, Any], shapes: list[dict[str, Any]]) -> None:
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    performance_fields = (
        "case_id",
        "baseline_median_ms",
        "deployed_median_ms",
        "deployed_p90_ms",
        "speedup",
        "peak_memory_bytes",
        "correctness_passed",
    )
    with (SOURCE_DATA / "performance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=performance_fields)
        writer.writeheader()
        for row in result["shapes"]:
            writer.writerow({field: row.get(field) for field in performance_fields})

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


def _require_strictly_positive(values: list[float], label: str) -> None:
    """Reject non-positive values before a logarithmic axis is applied."""

    if any(value <= 0 for value in values):
        raise ValueError(f"{label} must contain strictly positive values")


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        HERE / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.075, 1.02, label, transform=ax.transAxes, fontsize=8, fontweight="bold")


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
    fig = plt.figure(figsize=(7.2, 4.25))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.12, 0.88), hspace=0.25, wspace=0.18)
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
    ax_a.text(0.47, 0.24, "Illegal combinations are rejected\nbefore GPU execution", ha="center", va="center", fontsize=6, color=MID)

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
    _box(ax_b, (0.12, 0.18), 0.32, 0.22, "Single-GPU lease\n+ fresh process", facecolor=LIGHT, edgecolor=MID)
    _box(ax_b, (0.55, 0.18), 0.32, 0.22, "Accuracy + path\n+ latency + memory", facecolor=LIGHT, edgecolor=MID)
    _arrow(ax_b, (0.28, 0.40), (0.18, 0.61), color=MID, dashed=True)
    _arrow(ax_b, (0.71, 0.40), (0.67, 0.61), color=MID, dashed=True)

    _panel_label(ax_c, "c")
    ax_c.set_title("Exact-device deployment closes the loop", loc="left", pad=3)
    _box(ax_c, (0.01, 0.34), 0.18, 0.30, "Formal winner", facecolor=PALE_ORANGE, edgecolor=ORANGE)
    _box(ax_c, (0.27, 0.34), 0.20, 0.30, "Deployment registry\nexact GPU + shape", facecolor=PALE_ORANGE, edgecolor=ORANGE)
    _arrow(ax_c, (0.19, 0.49), (0.27, 0.49), color=ORANGE)
    _box(ax_c, (0.56, 0.58), 0.20, 0.25, "Shapes 01–13\nresident runtime", facecolor=PALE_TEAL, edgecolor=TEAL)
    _box(ax_c, (0.56, 0.14), 0.20, 0.25, "Shape 14\nstreamed runtime", facecolor=PALE_ORANGE, edgecolor=ORANGE)
    _arrow(ax_c, (0.47, 0.49), (0.56, 0.70), color=TEAL)
    _arrow(ax_c, (0.47, 0.49), (0.56, 0.26), color=ORANGE)
    _box(ax_c, (0.83, 0.34), 0.16, 0.30, "Official-compatible\nTransformer output", facecolor=LIGHT, edgecolor=MID)
    _arrow(ax_c, (0.76, 0.70), (0.83, 0.53), color=TEAL)
    _arrow(ax_c, (0.76, 0.26), (0.83, 0.45), color=ORANGE)
    ax_c.text(0.34, 0.11, "Resident TPE studies and Shape-14 finite search retain separate evidence stores", ha="center", va="center", fontsize=6, color=MID)

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
    colors = [TEAL if value >= geomean else BLUE for value in speedups]
    ax_a.barh(y, speedups, color=colors, edgecolor="white", linewidth=0.5)
    ax_a.axvline(geomean, color=ORANGE, linestyle="--", linewidth=1.2, label=f"Geomean {geomean:.2f}×")
    ax_a.set_yticks(y, labels)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Speedup (baseline median / deployed median)")
    ax_a.set_xlim(0, 40)
    ax_a.grid(axis="x", color=LIGHT, linewidth=0.7)
    ax_a.set_axisbelow(True)
    for yi, value in zip(y, speedups, strict=True):
        ax_a.text(value + 0.55, yi, f"{value:.1f}×", va="center", fontsize=5.8, color=INK)
    ax_a.legend(loc="lower right")
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
    ax_b.scatter(x, p90, s=17, facecolors="white", edgecolors=ORANGE, linewidths=0.9, label="Deployed P90", zorder=4)
    ax_b.set_yscale("log")
    ax_b.set_xticks(x, labels)
    ax_b.set_ylabel("Latency (ms, log scale)")
    ax_b.grid(axis="y", which="both", color=LIGHT, linewidth=0.7)
    ax_b.set_axisbelow(True)
    ax_b.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax_b.set_title("Absolute latency", loc="left")
    _panel_label(ax_b, "b")

    memory_gib = [float(row["peak_memory_bytes"]) / 2**30 for row in all_rows]
    _require_strictly_positive(memory_gib, "peak memory")
    all_labels = [row["case_id"].split("_")[-1] for row in all_rows]
    memory_colors = [BLUE] * 13 + [ORANGE]
    ax_c.bar(range(14), memory_gib, color=memory_colors, width=0.72)
    ax_c.set_yscale("log")
    ax_c.set_xticks(range(14), all_labels)
    ax_c.set_ylabel("Peak allocated memory (GiB, log scale)")
    ax_c.grid(axis="y", which="both", color=LIGHT, linewidth=0.7)
    ax_c.set_axisbelow(True)
    shape14 = all_rows[-1]
    ax_c.annotate(
        f"Shape 14: {shape14['deployed_median_ms'] / 1000:.2f} s\n6.56 GiB, correctness PASS",
        xy=(13, memory_gib[-1]),
        xytext=(8.2, 1.9),
        arrowprops={"arrowstyle": "->", "color": ORANGE, "linewidth": 0.9},
        fontsize=6,
        color=INK,
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
        "All 14 Shapes pass correctness; Shape 14 has no dense baseline and is excluded from the geomean.",
        ha="right",
        va="top",
        fontsize=6,
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
    ax.plot(x, y, color=BLUE, linewidth=1.2, marker="o", markersize=3.8)
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
    ax.set_xlabel(x_field.replace("_", " ").title())
    ax.grid(color=LIGHT, linewidth=0.6)
    ax.set_axisbelow(True)


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
            s=[24 + 0.09 * shape["qkv_dim"] for shape in subset],
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
        "01, 07–11\n(shared B and S)",
        (128, 64 * 128),
        xytext=(12, 4),
        textcoords="offset points",
        fontsize=5.6,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": MID, "linewidth": 0.6},
    )
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Sequence length S (log scale)")
    ax_a.set_ylabel("Logical tokens B × S (log scale)")
    ax_a.grid(which="both", color=LIGHT, linewidth=0.6)
    ax_a.set_axisbelow(True)
    ax_a.legend(loc="upper left", ncols=2)
    ax_a.set_title("Official workload regime map", loc="left")
    _panel_label(ax_a, "a")
    ax_a.text(
        0.98,
        0.02,
        "Marker area scales with QKV dimension\nOrange = streamed Shape 14",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
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
        _panel_label(axes[label], label)
    axes["b"].set_ylabel("Speedup")
    axes["d"].set_ylabel("Speedup")
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
        0.99,
        0.955,
        "Sensitivity panels compare independently deployed plans; they are not causal kernel ablations.",
        ha="right",
        va="top",
        fontsize=6,
        color=MID,
    )
    _save(fig, "workload_landscape")


def main() -> None:
    result_path, result = _latest_result()
    shapes = _load_shapes()
    _write_source_data(result, shapes)
    make_architecture()
    make_performance(result)
    make_workload_landscape(result, shapes)
    print(f"Generated figures from {result_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
