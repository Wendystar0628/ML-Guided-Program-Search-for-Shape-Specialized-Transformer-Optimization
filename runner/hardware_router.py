"""Explainable cold-start ranking for official Transformer candidates.

The model is a theory-backed prior, not a latency predictor. It limits the
first measurement sweep on unseen hardware; measured Formal results remain the
only source that may update the exact dispatch table.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from runner.candidates import CapabilityTag, RoutingTag, candidate_spec
from runner.contracts import RunVariant, TransformerShape

_DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}


@dataclass(frozen=True)
class WorkloadAnalysis:
    """Hardware-independent shape and cost features used for cold routing."""

    case_id: str
    dtype: str
    batch_size: int
    seq_len: int
    sequence_squared: int
    d_model: int
    num_heads: int
    head_dim: int
    ffn_dim: int
    num_layers: int
    tokens: int
    projection_ffn_flops: int
    attention_flops: int
    total_flops: int
    estimated_bytes: int
    arithmetic_intensity_flops_per_byte: float
    attention_matrix_elements: int
    dense_attention_peak_bytes: int
    estimated_peak_bytes: int
    estimated_parallel_blocks: int
    estimated_kernel_launches: int
    dense_gemm_fraction: float
    attention_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_workload(
    shape: TransformerShape,
    variant: RunVariant,
) -> WorkloadAnalysis:
    """Estimate compute, traffic, and explicit-attention memory for one run."""

    if not isinstance(shape, TransformerShape):
        raise TypeError("shape must be a TransformerShape")
    if not isinstance(variant, RunVariant):
        raise TypeError("variant must be a RunVariant")
    shape.validate()
    variant.validate()

    batch = shape.batch_size
    sequence = shape.seq_len
    model_dim = shape.d_model
    heads = shape.num_heads
    ffn_dim = shape.ffn_dim
    layers = shape.num_layers
    tokens = batch * sequence
    dtype_bytes = _DTYPE_BYTES[variant.dtype]

    dense_flops_per_layer = (
        8 * tokens * model_dim * model_dim + 4 * tokens * model_dim * ffn_dim
    )
    attention_flops_per_layer = 4 * batch * sequence * sequence * model_dim
    projection_ffn_flops = dense_flops_per_layer * layers
    attention_flops = attention_flops_per_layer * layers
    total_flops = projection_ffn_flops + attention_flops

    sequence_squared = sequence * sequence
    attention_elements = batch * heads * sequence_squared
    # The official reference promotes scores to FP32 before softmax and retains
    # probabilities in the requested dtype. A causal bool mask adds S^2 bytes.
    dense_attention_peak_bytes = attention_elements * (4 + dtype_bytes)
    if shape.causal:
        dense_attention_peak_bytes += sequence_squared

    weight_elements_per_layer = 4 * model_dim * model_dim + 2 * model_dim * ffn_dim
    weight_bytes = layers * weight_elements_per_layer * dtype_bytes
    activation_bytes_per_layer = (
        12 * tokens * model_dim + 2 * tokens * ffn_dim
    ) * dtype_bytes
    attention_traffic_per_layer = attention_elements * (4 + 2 * dtype_bytes)
    estimated_bytes = layers * (
        weight_elements_per_layer * dtype_bytes
        + activation_bytes_per_layer
        + attention_traffic_per_layer
    )
    persistent_io_bytes = 2 * tokens * model_dim * dtype_bytes
    estimated_peak_bytes = (
        weight_bytes
        + persistent_io_bytes
        + activation_bytes_per_layer
        + dense_attention_peak_bytes
    )

    dense_blocks = math.ceil(tokens / 128) * math.ceil(max(model_dim, ffn_dim) / 128)
    attention_blocks = (
        batch * heads * math.ceil(sequence / 128) * math.ceil(sequence / 128)
    )
    parallel_blocks = max(dense_blocks, attention_blocks)
    estimated_launches = 2 + 18 * layers
    intensity = total_flops / max(estimated_bytes, 1)
    return WorkloadAnalysis(
        case_id=shape.case_id,
        dtype=variant.dtype,
        batch_size=batch,
        seq_len=sequence,
        sequence_squared=sequence_squared,
        d_model=model_dim,
        num_heads=heads,
        head_dim=model_dim // heads,
        ffn_dim=ffn_dim,
        num_layers=layers,
        tokens=tokens,
        projection_ffn_flops=projection_ffn_flops,
        attention_flops=attention_flops,
        total_flops=total_flops,
        estimated_bytes=estimated_bytes,
        arithmetic_intensity_flops_per_byte=round(intensity, 6),
        attention_matrix_elements=attention_elements,
        dense_attention_peak_bytes=dense_attention_peak_bytes,
        estimated_peak_bytes=estimated_peak_bytes,
        estimated_parallel_blocks=parallel_blocks,
        estimated_kernel_launches=estimated_launches,
        dense_gemm_fraction=round(projection_ffn_flops / total_flops, 6),
        attention_fraction=round(attention_flops / total_flops, 6),
    )


def _positive_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def _positive_int(value: object) -> int | None:
    number = _positive_float(value)
    return int(number) if number is not None and number.is_integer() else None


def _version_pair(value: object) -> tuple[int, int] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parts = value.strip().split(".")
    if not parts[0].isdigit():
        return None
    return int(parts[0]), int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _lookup_gemm_tflops(anchors: Mapping[str, Any], dtype: str) -> float | None:
    values = _mapping(anchors.get("gemm_tflops"))
    aliases = {
        "float16": ("float16", "fp16"),
        "bfloat16": ("bfloat16", "bf16"),
        "float32": ("float32", "fp32", "tf32"),
    }
    for name in aliases[dtype]:
        result = _positive_float(values.get(name))
        if result is not None:
            return result
    return None


def _effective_bandwidth_gbps(
    profile: Mapping[str, Any], anchors: Mapping[str, Any]
) -> float | None:
    measured = _positive_float(anchors.get("memory_bandwidth_gbps"))
    theoretical = _positive_float(profile.get("theoretical_memory_bandwidth_gbps"))
    if measured is not None and theoretical is not None:
        return min(measured, theoretical)
    return measured if measured is not None else theoretical


def _capability_rejection(
    candidate_id: str,
    shape: TransformerShape,
    variant: RunVariant,
    profile: Mapping[str, Any],
) -> str | None:
    spec = candidate_spec(candidate_id)
    if spec is None:
        return "candidate is not registered"
    if not spec.applies(shape, variant):
        return f"candidate applies only to {spec.applicability_description}"
    if CapabilityTag.CUDA in spec.capability_tags:
        if str(profile.get("device_type", "")).lower() != "cuda":
            return "CUDA device capability was not established"
        cuda = _version_pair(profile.get("cuda_runtime"))
        if cuda is None or cuda < (11, 0):
            return "candidate requires CUDA 11 or newer"
    capability = _version_pair(profile.get("compute_capability"))
    if (
        spec.minimum_compute_capability is not None
        and capability is not None
        and capability < spec.minimum_compute_capability
    ):
        required = ".".join(map(str, spec.minimum_compute_capability))
        return f"candidate requires compute capability {required} or newer"
    if variant.dtype == "bfloat16" and profile.get("bf16_supported") is False:
        return "native BF16 support is unavailable"
    if (
        CapabilityTag.CUDA_GRAPH in spec.capability_tags
        and profile.get("cuda_graph_available") is False
    ):
        return "CUDA Graph runtime capability is unavailable"
    if not spec.supports_on_hardware(shape, variant):
        description = (
            spec.hardware_support_description or spec.applicability_description
        )
        return f"candidate backend supports only {description}"
    return None


@dataclass(frozen=True)
class _HardwareSignals:
    bottleneck_class: str
    effective_gemm_tflops: float | None
    effective_bandwidth_gbps: float | None
    machine_ridge_flops_per_byte: float | None
    intensity_to_ridge: float | None
    dense_attention_to_memory: float | None
    estimated_peak_to_memory: float | None
    attention_to_l2: float | None
    blocks_per_sm: float | None
    launch_dominant: bool
    graph_replay_is_cheaper: bool


def _hardware_signals(
    analysis: WorkloadAnalysis,
    profile: Mapping[str, Any],
) -> _HardwareSignals:
    anchors = _mapping(profile.get("performance_anchors"))
    gemm_tflops = _lookup_gemm_tflops(anchors, analysis.dtype)
    bandwidth = _effective_bandwidth_gbps(profile, anchors)
    ridge = (
        gemm_tflops * 1000.0 / bandwidth
        if gemm_tflops is not None and bandwidth is not None
        else None
    )
    intensity_to_ridge = (
        analysis.arithmetic_intensity_flops_per_byte / ridge if ridge else None
    )

    total_memory = _positive_float(profile.get("total_memory_bytes"))
    attention_to_memory = (
        analysis.dense_attention_peak_bytes / total_memory if total_memory else None
    )
    peak_to_memory = (
        analysis.estimated_peak_bytes / total_memory if total_memory else None
    )
    l2_bytes = _positive_float(profile.get("l2_cache_bytes"))
    attention_to_l2 = (
        analysis.dense_attention_peak_bytes / l2_bytes if l2_bytes else None
    )
    sm_count = _positive_int(profile.get("sm_count"))
    blocks_per_sm = analysis.estimated_parallel_blocks / sm_count if sm_count else None

    launch_latency_us = _positive_float(anchors.get("launch_latency_us"))
    graph_node_us = _positive_float(anchors.get("graph_replay_per_node_us"))
    graph_cheaper = (
        launch_latency_us is not None
        and graph_node_us is not None
        and graph_node_us < launch_latency_us
    )
    compute_seconds = (
        analysis.total_flops / (gemm_tflops * 1e12) if gemm_tflops else None
    )
    memory_seconds = analysis.estimated_bytes / (bandwidth * 1e9) if bandwidth else None
    launch_seconds = (
        analysis.estimated_kernel_launches * launch_latency_us * 1e-6
        if launch_latency_us
        else None
    )
    launch_dominant = analysis.tokens <= 512
    if blocks_per_sm is not None and blocks_per_sm < 1.0:
        launch_dominant = True
    if launch_seconds is not None and (
        compute_seconds is not None or memory_seconds is not None
    ):
        launch_dominant = launch_seconds > max(
            compute_seconds or 0.0,
            memory_seconds or 0.0,
        )

    if peak_to_memory is not None and peak_to_memory >= 0.8:
        bottleneck = "memory_capacity"
    elif launch_dominant:
        bottleneck = "launch_underfill"
    elif analysis.attention_fraction >= 0.5 and (
        analysis.seq_len >= 1024 or attention_to_l2 is None or attention_to_l2 >= 1.0
    ):
        bottleneck = "attention_memory"
    elif (
        intensity_to_ridge is not None
        and intensity_to_ridge >= 1.2
        and analysis.dense_gemm_fraction >= 0.7
    ) or (
        intensity_to_ridge is None
        and analysis.dense_gemm_fraction >= 0.9
        and analysis.tokens >= 1024
    ):
        bottleneck = "tensor_compute"
    elif intensity_to_ridge is not None and intensity_to_ridge <= 0.8:
        bottleneck = "memory_bandwidth"
    else:
        bottleneck = "balanced"

    return _HardwareSignals(
        bottleneck_class=bottleneck,
        effective_gemm_tflops=gemm_tflops,
        effective_bandwidth_gbps=bandwidth,
        machine_ridge_flops_per_byte=ridge,
        intensity_to_ridge=intensity_to_ridge,
        dense_attention_to_memory=attention_to_memory,
        estimated_peak_to_memory=peak_to_memory,
        attention_to_l2=attention_to_l2,
        blocks_per_sm=blocks_per_sm,
        launch_dominant=launch_dominant,
        graph_replay_is_cheaper=graph_cheaper,
    )


def _candidate_score(
    candidate_id: str,
    analysis: WorkloadAnalysis,
    signals: _HardwareSignals,
) -> tuple[int, list[str]]:
    spec = candidate_spec(candidate_id)
    tags = spec.routing_tags if spec is not None else frozenset()
    score = 0
    reasons: list[str] = []

    if RoutingTag.AUTO_CONTROL in tags:
        score += 45
        reasons.append("retained as the measured automatic control")
    if RoutingTag.SAFE_FALLBACK in tags:
        score += 25
        reasons.append("portable causal fallback")
    if RoutingTag.CAUSAL_SDPA in tags:
        score += 60
        reasons.append("avoids the explicit causal mask and score/probability path")
        if signals.bottleneck_class in {"attention_memory", "memory_capacity"}:
            score += 45
            reasons.append("attention materialization is a primary shape cost")
        if analysis.head_dim in {32, 64, 128}:
            score += 10
            reasons.append("head dimension is friendly to fused SDPA kernels")
        elif analysis.head_dim in {8, 256}:
            reasons.append("edge head dimension requires measured backend selection")
    if RoutingTag.GRAPH in tags:
        score += 90 if signals.launch_dominant else 15
        reasons.append("amortizes repeated static launch submission")
        if signals.graph_replay_is_cheaper:
            score += 10
            reasons.append("probe measured graph replay below launch cost")
    if RoutingTag.BATCH_TILED in tags:
        score += 100 if analysis.batch_size >= 1024 else 0
        reasons.append("bounds peak memory for the extreme-batch family")
        if (
            signals.estimated_peak_to_memory is not None
            and signals.estimated_peak_to_memory >= 0.35
        ):
            score += 35
            reasons.append("estimated working set consumes a large memory share")
    if RoutingTag.INPLACE_BLOCK in tags:
        score += 25
        reasons.append("removes an exact-GELU allocation without replacing GEMM")
        if signals.bottleneck_class == "tensor_compute" or analysis.d_model >= 1024:
            score += 50
            reasons.append("wide dense work can amortize the in-place block path")
        elif signals.launch_dominant:
            score += 15
            reasons.append("small shapes benefit from fewer block-level operations")
    return score, reasons


def build_routing_plan(
    shape: TransformerShape,
    variant: RunVariant,
    hardware_profile: Mapping[str, Any],
    candidate_ids: Sequence[str],
    *,
    limit: int = 3,
    required_candidate_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Rank a bounded Smoke shortlist; deployment still requires measurement."""

    if not isinstance(shape, TransformerShape):
        raise TypeError("shape must be a TransformerShape")
    if not isinstance(variant, RunVariant):
        raise TypeError("variant must be a RunVariant")
    if not isinstance(hardware_profile, Mapping):
        raise TypeError("hardware_profile must be a mapping")
    if isinstance(candidate_ids, (str, bytes)) or not isinstance(
        candidate_ids, Sequence
    ):
        raise TypeError("candidate_ids must be a sequence of strings")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")

    unique: list[str] = []
    seen: set[str] = set()
    for candidate_id in candidate_ids:
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate_ids must contain non-empty strings")
        if candidate_id not in seen:
            unique.append(candidate_id)
            seen.add(candidate_id)

    analysis = analyze_workload(shape, variant)
    signals = _hardware_signals(analysis, hardware_profile)
    rejections: dict[str, str] = {}
    ranked: list[tuple[int, int, str, list[str]]] = []
    for index, candidate_id in enumerate(unique):
        rejection = _capability_rejection(
            candidate_id,
            shape,
            variant,
            hardware_profile,
        )
        if rejection is not None:
            rejections[candidate_id] = rejection
            continue
        score, reasons = _candidate_score(candidate_id, analysis, signals)
        ranked.append((score, index, candidate_id, reasons))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    required: list[str] = []
    if any(item[2] == "eager-auto" for item in ranked):
        required.append("eager-auto")
    for candidate_id in required_candidate_ids:
        if candidate_id not in seen:
            raise ValueError(f"required candidate is unavailable: {candidate_id}")
        if candidate_id in rejections:
            raise ValueError(
                f"required candidate is not eligible: {candidate_id}: "
                f"{rejections[candidate_id]}"
            )
        if candidate_id not in required:
            required.append(candidate_id)
    if len(required) > limit:
        raise ValueError("candidate limit cannot retain the control and incumbent")

    selected_ids = {item[2] for item in ranked[:limit]}
    selected_ids.update(required)
    while len(selected_ids) > limit:
        removable = next(
            (
                item[2]
                for item in reversed(ranked)
                if item[2] in selected_ids and item[2] not in required
            ),
            None,
        )
        if removable is None:
            raise ValueError("unable to retain required routing candidates")
        selected_ids.remove(removable)
    selected = [item for item in ranked if item[2] in selected_ids]

    routing_signals = {
        "architecture_family": hardware_profile.get("architecture_family"),
        "effective_gemm_tflops": signals.effective_gemm_tflops,
        "effective_bandwidth_gbps": signals.effective_bandwidth_gbps,
        "machine_ridge_flops_per_byte": signals.machine_ridge_flops_per_byte,
        "workload_intensity_to_ridge": signals.intensity_to_ridge,
        "dense_attention_to_device_memory": signals.dense_attention_to_memory,
        "estimated_peak_to_device_memory": signals.estimated_peak_to_memory,
        "dense_attention_to_l2": signals.attention_to_l2,
        "estimated_blocks_per_sm": signals.blocks_per_sm,
        "launch_dominant": signals.launch_dominant,
        "graph_replay_is_cheaper": signals.graph_replay_is_cheaper,
    }
    return {
        "source": "hardware_cost_model",
        "decision_scope": "candidate_order_only",
        "requires_full_workload_measurement": True,
        "bottleneck_class": signals.bottleneck_class,
        "workload_analysis": analysis.as_dict(),
        "routing_signals": {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in routing_signals.items()
        },
        "candidate_order": [item[2] for item in selected],
        "selection_reasons": {item[2]: item[3] for item in selected},
        "capability_rejections": rejections,
    }


__all__ = ["WorkloadAnalysis", "analyze_workload", "build_routing_plan"]
