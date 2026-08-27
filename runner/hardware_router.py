"""Explainable cold-start candidate routing for previously unseen hardware.

The model in this module is intentionally a candidate ranker, not a latency
predictor and not a replacement for the measured dispatch table.  It combines
static workload estimates with a compact hardware profile so a device
calibration run can measure a small, relevant candidate set.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from runner.candidates import (
    CapabilityTag,
    RoutingTag,
    candidate_spec,
)
from runner.contracts import WorkloadCase

_DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}

_SOFTMAX_REDUCTION_MIN_SHARE = 0.15


@dataclass(frozen=True)
class WorkloadAnalysis:
    """Hardware-independent estimates used to classify a Transformer case."""

    case_id: str
    dtype: str
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    tokens: int
    head_dim: int
    projection_ffn_flops: int
    attention_flops: int
    total_flops: int
    estimated_bytes: int
    arithmetic_intensity_flops_per_byte: float
    attention_matrix_elements: int
    attention_matrix_bytes: int
    estimated_parallel_blocks: int
    estimated_kernel_launches: int
    dense_gemm_fraction: float
    attention_fraction: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)


def analyze_workload(case: WorkloadCase) -> WorkloadAnalysis:
    """Estimate the main compute, traffic, and parallelism of one case.

    The byte count is a consistent lower-detail model of the explicit
    Transformer path.  It is useful for relative routing signals, but it is not
    presented as a prediction of measured traffic or latency.
    """

    if not isinstance(case, WorkloadCase):
        raise TypeError("case must be a WorkloadCase")
    case.validate()

    batch = case.batch_size
    sequence = case.seq_len
    model_dim = case.d_model
    ffn_dim = case.ffn_dim
    layers = case.num_layers
    tokens = batch * sequence
    dtype_bytes = _DTYPE_BYTES[case.dtype]

    # Q/K/V and output projections contribute 8*T*D^2 FLOPs.  The two FFN
    # projections contribute 4*T*D*F FLOPs.  Multiply-add counts as two FLOPs.
    dense_flops_per_layer = (
        8 * tokens * model_dim * model_dim + 4 * tokens * model_dim * ffn_dim
    )
    # QK and probability-value matmuls each contribute 2*B*S^2*D FLOPs.
    attention_flops_per_layer = 4 * batch * sequence * sequence * model_dim
    projection_ffn_flops = dense_flops_per_layer * layers
    attention_flops = attention_flops_per_layer * layers
    total_flops = projection_ffn_flops + attention_flops

    attention_matrix_elements = batch * case.num_heads * sequence * sequence
    attention_matrix_bytes = attention_matrix_elements * dtype_bytes

    # The estimate includes one read of each layer's weights, representative
    # dense activation traffic, and explicit score/FP32-softmax/probability
    # materialization.  It deliberately avoids pretending to model cache hits.
    weight_bytes_per_layer = (
        4 * model_dim * model_dim + 2 * model_dim * ffn_dim
    ) * dtype_bytes
    dense_activation_bytes_per_layer = (
        12 * tokens * model_dim + 2 * tokens * ffn_dim
    ) * dtype_bytes
    attention_bytes_per_layer = attention_matrix_elements * (2 * dtype_bytes + 4)
    estimated_bytes = layers * (
        weight_bytes_per_layer
        + dense_activation_bytes_per_layer
        + attention_bytes_per_layer
    )

    # A 128x128 output tile is only a coarse occupancy signal.  Taking the
    # larger dense/attention grid catches both tiny FFNs and long attention.
    dense_blocks = math.ceil(tokens / 128) * math.ceil(max(model_dim, ffn_dim) / 128)
    attention_blocks = (
        batch * case.num_heads * math.ceil(sequence / 128) * math.ceil(sequence / 128)
    )
    estimated_parallel_blocks = max(dense_blocks, attention_blocks)
    estimated_kernel_launches = 2 + 18 * layers

    arithmetic_intensity = total_flops / max(estimated_bytes, 1)
    return WorkloadAnalysis(
        case_id=case.case_id,
        dtype=case.dtype,
        batch_size=batch,
        seq_len=sequence,
        d_model=model_dim,
        num_heads=case.num_heads,
        ffn_dim=ffn_dim,
        num_layers=layers,
        tokens=tokens,
        head_dim=model_dim // case.num_heads,
        projection_ffn_flops=projection_ffn_flops,
        attention_flops=attention_flops,
        total_flops=total_flops,
        estimated_bytes=estimated_bytes,
        arithmetic_intensity_flops_per_byte=round(arithmetic_intensity, 6),
        attention_matrix_elements=attention_matrix_elements,
        attention_matrix_bytes=attention_matrix_bytes,
        estimated_parallel_blocks=estimated_parallel_blocks,
        estimated_kernel_launches=estimated_kernel_launches,
        dense_gemm_fraction=round(projection_ffn_flops / total_flops, 6),
        attention_fraction=round(attention_flops / total_flops, 6),
    )


def _positive_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _positive_int(value: object) -> int | None:
    number = _positive_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _version_pair(value: object) -> tuple[int, int] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    parts = value.strip().split(".")
    if not parts or not parts[0].isdigit():
        return None
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return major, minor


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
    hardware_profile: Mapping[str, Any], anchors: Mapping[str, Any]
) -> float | None:
    measured = _positive_float(anchors.get("memory_bandwidth_gbps"))
    theoretical = _positive_float(
        hardware_profile.get("theoretical_memory_bandwidth_gbps")
    )
    if measured is not None and theoretical is not None:
        # A compact copy anchor can be served partly from L2 and therefore
        # exceed DRAM peak.  Roofline routing needs an HBM/GDDR ceiling, so the
        # static peak is an upper bound rather than an alternative estimate.
        return min(measured, theoretical)
    return measured if measured is not None else theoretical


def _capability_rejection(
    candidate_id: str,
    case: WorkloadCase,
    hardware_profile: Mapping[str, Any],
) -> str | None:
    spec = candidate_spec(candidate_id)
    if spec is None:
        return None
    if CapabilityTag.CUDA not in spec.capability_tags:
        return (
            None
            if spec.applies(case)
            else f"candidate applies only to {spec.applicability_description}"
        )

    if str(hardware_profile.get("device_type", "")).lower() != "cuda":
        return "CUDA device capability was not established"

    cuda_version = _version_pair(hardware_profile.get("cuda_runtime"))
    if cuda_version is None:
        return "CUDA runtime capability was not established"
    if cuda_version < (11, 0):
        return "candidate requires CUDA 11 or newer"

    capability = _version_pair(hardware_profile.get("compute_capability"))
    if CapabilityTag.TRITON in spec.capability_tags:
        if hardware_profile.get("triton_available") is False:
            return "Triton runtime is unavailable"
        if capability is None:
            return "compute capability was not established for the Triton route"
        if capability < (7, 0):
            return "Triton route requires compute capability 7.0 or newer"

    if (
        spec.minimum_compute_capability is not None
        and capability is not None
        and capability < spec.minimum_compute_capability
    ):
        required = ".".join(str(value) for value in spec.minimum_compute_capability)
        return f"candidate requires compute capability {required} or newer"

    if (
        case.dtype == "bfloat16"
        and CapabilityTag.TRITON in spec.capability_tags
        and hardware_profile.get("bf16_supported") is False
    ):
        return "native BF16 support is unavailable"

    if (
        case.dtype == "bfloat16"
        and CapabilityTag.TRITON in spec.capability_tags
        and capability is not None
        and capability < (8, 0)
    ):
        return "native BF16 Triton route requires compute capability 8.0 or newer"

    if (
        CapabilityTag.CUDA_GRAPH in spec.capability_tags
        and hardware_profile.get("cuda_graph_available") is False
    ):
        return "CUDA Graph runtime capability is unavailable"

    if not spec.applies(case):
        return f"candidate applies only to {spec.applicability_description}"
    if not spec.supports_case_on_hardware(case):
        description = (
            spec.hardware_case_support_description or spec.applicability_description
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
    attention_to_l2: float | None
    blocks_per_sm: float | None
    softmax_to_compute_lower_bound: float | None
    launch_dominant: bool
    graph_replay_is_cheaper: bool


def _hardware_signals(
    analysis: WorkloadAnalysis, hardware_profile: Mapping[str, Any]
) -> _HardwareSignals:
    anchors = _mapping(hardware_profile.get("performance_anchors"))
    gemm_tflops = _lookup_gemm_tflops(anchors, analysis.dtype)
    bandwidth_gbps = _effective_bandwidth_gbps(hardware_profile, anchors)
    ridge = None
    intensity_to_ridge = None
    if gemm_tflops is not None and bandwidth_gbps is not None:
        ridge = gemm_tflops * 1000.0 / bandwidth_gbps
        intensity_to_ridge = analysis.arithmetic_intensity_flops_per_byte / ridge

    l2_bytes = _positive_float(hardware_profile.get("l2_cache_bytes"))
    attention_to_l2 = None
    if l2_bytes is not None:
        attention_to_l2 = analysis.attention_matrix_bytes / l2_bytes

    sm_count = _positive_int(hardware_profile.get("sm_count"))
    blocks_per_sm = None
    if sm_count is not None:
        blocks_per_sm = analysis.estimated_parallel_blocks / sm_count

    launch_latency_us = _positive_float(anchors.get("launch_latency_us"))
    graph_replay_per_node_us = _positive_float(anchors.get("graph_replay_per_node_us"))
    graph_replay_is_cheaper = (
        launch_latency_us is not None
        and graph_replay_per_node_us is not None
        and graph_replay_per_node_us < launch_latency_us
    )

    # The lower bounds are used only for a broad launch classification.  They
    # are never returned as a latency estimate.
    compute_seconds = None
    if gemm_tflops is not None:
        compute_seconds = analysis.total_flops / (gemm_tflops * 1e12)
    memory_seconds = None
    if bandwidth_gbps is not None:
        memory_seconds = analysis.estimated_bytes / (bandwidth_gbps * 1e9)
    launch_seconds = None
    if launch_latency_us is not None:
        launch_seconds = analysis.estimated_kernel_launches * launch_latency_us * 1e-6

    softmax_throughput = _positive_float(anchors.get("softmax_giga_elements_per_s"))
    softmax_to_compute_lower_bound = None
    if softmax_throughput is not None and compute_seconds is not None:
        softmax_seconds = (
            analysis.attention_matrix_elements
            * analysis.num_layers
            / (softmax_throughput * 1e9)
        )
        softmax_to_compute_lower_bound = softmax_seconds / max(compute_seconds, 1e-12)

    launch_dominant = analysis.tokens <= 64
    if blocks_per_sm is not None and blocks_per_sm < 0.5:
        launch_dominant = True
    if launch_seconds is not None and (
        compute_seconds is not None or memory_seconds is not None
    ):
        device_work = max(compute_seconds or 0.0, memory_seconds or 0.0)
        launch_dominant = launch_seconds > device_work

    if launch_dominant:
        bottleneck_class = "launch_underfill"
    elif analysis.seq_len >= 1024 and (
        attention_to_l2 is None
        or attention_to_l2 >= 0.5
        or analysis.attention_fraction >= 0.2
    ):
        bottleneck_class = "attention_memory"
    elif (
        analysis.seq_len >= 256
        and analysis.attention_matrix_elements >= 4_000_000
        and (
            softmax_to_compute_lower_bound is None
            and analysis.attention_fraction >= 0.1
            or softmax_to_compute_lower_bound is not None
            and softmax_to_compute_lower_bound >= _SOFTMAX_REDUCTION_MIN_SHARE
        )
    ):
        bottleneck_class = "softmax_reduction"
    elif (
        intensity_to_ridge is not None
        and intensity_to_ridge >= 1.25
        and analysis.dense_gemm_fraction >= 0.7
    ) or (
        intensity_to_ridge is None
        and analysis.dense_gemm_fraction >= 0.85
        and analysis.tokens >= 1024
    ):
        bottleneck_class = "tensor_compute"
    elif intensity_to_ridge is not None and intensity_to_ridge <= 0.8:
        bottleneck_class = "memory_bandwidth"
    else:
        bottleneck_class = "balanced"

    return _HardwareSignals(
        bottleneck_class=bottleneck_class,
        effective_gemm_tflops=gemm_tflops,
        effective_bandwidth_gbps=bandwidth_gbps,
        machine_ridge_flops_per_byte=ridge,
        intensity_to_ridge=intensity_to_ridge,
        attention_to_l2=attention_to_l2,
        blocks_per_sm=blocks_per_sm,
        softmax_to_compute_lower_bound=softmax_to_compute_lower_bound,
        launch_dominant=launch_dominant,
        graph_replay_is_cheaper=graph_replay_is_cheaper,
    )


def _candidate_score(
    candidate_id: str,
    case: WorkloadCase,
    analysis: WorkloadAnalysis,
    signals: _HardwareSignals,
) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    spec = candidate_spec(candidate_id)
    tags = spec.routing_tags if spec is not None else frozenset()

    if RoutingTag.SAFE_FALLBACK in tags or candidate_id == "auto":
        score += 35
        reasons.append("retained as the safe general fallback")
    elif RoutingTag.TORCH_CONTROL in tags:
        score += 10
        reasons.append("portable eager control for a new device")
    elif RoutingTag.REFERENCE_CONTROL in tags:
        score -= 20
        reasons.append("correctness control rather than a preferred optimized route")
    elif RoutingTag.GENERAL_TRITON in tags:
        score += 12
        reasons.append("general custom-kernel comparison on supported CUDA hardware")
    else:
        reasons.append("eligible project candidate")

    if RoutingTag.GRAPH in tags:
        score += 85 if signals.bottleneck_class == "launch_underfill" else 15
        reasons.append("reduces repeated CPU and kernel-launch submission")
        if signals.graph_replay_is_cheaper:
            score += 15
            reasons.append("device anchor reports cheaper graph replay than launch")
        if RoutingTag.SOLUTION_GRAPH in tags:
            score += 10
            reasons.append("deployable Solution-owned graph route")
        elif RoutingTag.RUNNER_GRAPH in tags:
            reasons.append("Runner-owned graph control for measuring the graph ceiling")
    if RoutingTag.BALANCED_GRAPH in tags and case.seq_len <= 128:
        score += 20
        reasons.append("matches the short balanced static-shape family")

    if RoutingTag.COMPILE_REDUCE_OVERHEAD in tags:
        score += 70 if signals.bottleneck_class == "launch_underfill" else 20
        reasons.append("targets launch and Python overhead for short static shapes")
    elif RoutingTag.COMPILE_DEFAULT in tags:
        score += 45 if case.seq_len <= 128 else 12
        reasons.append("tests compiler fusion without an aggressive autotune search")
    elif RoutingTag.COMPILE_MAX_AUTOTUNE in tags:
        score += 65 if signals.bottleneck_class == "tensor_compute" else 5
        reasons.append("screens library and compiler choices for GEMM-heavy work")

    if RoutingTag.ATTENTION_ONLINE in tags:
        score += 105 if signals.bottleneck_class == "attention_memory" else 20
        reasons.append("reduces long-sequence score and probability materialization")
        if signals.attention_to_l2 is not None and signals.attention_to_l2 >= 1.0:
            score += 15
            reasons.append("one attention matrix is at least as large as L2")
    elif RoutingTag.ATTENTION_PREPROCESS in tags:
        score += 65 if case.seq_len >= 512 else 20
        reasons.append("fuses attention scale, mask, and promotion traffic")
    elif RoutingTag.S512_NATIVE_SOFTMAX in tags:
        score += 105 if case.dtype == "float16" and case.seq_len == 512 else 10
        reasons.append("targets the FP16 S512 softmax and conversion boundary")

    if RoutingTag.WIDE_INPLACE in tags:
        score += 105 if signals.bottleneck_class == "tensor_compute" else 55
        reasons.append(
            "keeps tuned GEMMs while removing a wide GELU allocation and write"
        )

    if RoutingTag.PADDING_PACKED in tags:
        if case.padding_ratio >= 0.5:
            score += 90
            reasons.append(
                "high static padding ratio can repay pack and scatter overhead"
            )
        else:
            score -= 10
            reasons.append("limited padding gives little token-skipping headroom")
    elif RoutingTag.PADDING_FUSED in tags:
        score += 65 if case.padding_ratio > 0 else 15
        reasons.append(
            "combines residual and padding work without a broad graph rewrite"
        )

    if signals.bottleneck_class in {
        "attention_memory",
        "memory_bandwidth",
    } and tags.intersection(
        {RoutingTag.ATTENTION_PREPROCESS, RoutingTag.ATTENTION_ONLINE}
    ):
        score += 20
    if signals.bottleneck_class == "softmax_reduction" and tags.intersection(
        {RoutingTag.ATTENTION_PREPROCESS, RoutingTag.S512_NATIVE_SOFTMAX}
    ):
        score += 20

    return score, reasons


def build_routing_plan(
    case: WorkloadCase,
    hardware_profile: Mapping[str, Any],
    candidate_ids: Sequence[str],
    *,
    limit: int = 4,
    required_candidate_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Rank a bounded candidate set for Smoke screening on new hardware.

    Only caller-supplied candidate IDs can be returned.  The general ``auto``
    candidate is retained when supplied; the resulting order is a prior for
    measured tuning and must not be written directly into deployed dispatch.
    """

    if not isinstance(case, WorkloadCase):
        raise TypeError("case must be a WorkloadCase")
    if not isinstance(hardware_profile, Mapping):
        raise TypeError("hardware_profile must be a mapping")
    if isinstance(candidate_ids, (str, bytes)) or not isinstance(
        candidate_ids, Sequence
    ):
        raise TypeError("candidate_ids must be a sequence of strings")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if isinstance(required_candidate_ids, (str, bytes)) or not isinstance(
        required_candidate_ids, Sequence
    ):
        raise TypeError("required_candidate_ids must be a sequence of strings")

    unique_candidates: list[str] = []
    seen: set[str] = set()
    for candidate_id in candidate_ids:
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate_ids must contain non-empty strings")
        if candidate_id not in seen:
            unique_candidates.append(candidate_id)
            seen.add(candidate_id)

    analysis = analyze_workload(case)
    signals = _hardware_signals(analysis, hardware_profile)
    rejections: dict[str, str] = {}
    ranked: list[tuple[int, int, str, list[str]]] = []
    for index, candidate_id in enumerate(unique_candidates):
        rejection = _capability_rejection(candidate_id, case, hardware_profile)
        if rejection is not None:
            rejections[candidate_id] = rejection
            continue
        score, reasons = _candidate_score(candidate_id, case, analysis, signals)
        ranked.append((score, index, candidate_id, reasons))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    required: list[str] = []
    fallback = next(
        (
            item[2]
            for item in ranked
            if item[2] == "auto"
            or (
                (spec := candidate_spec(item[2])) is not None
                and RoutingTag.SAFE_FALLBACK in spec.routing_tags
            )
        ),
        None,
    )
    if fallback is not None:
        required.append(fallback)
    for candidate_id in required_candidate_ids:
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("required_candidate_ids must contain non-empty strings")
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
        raise ValueError(
            "candidate limit is too small to retain auto and the current incumbent"
        )

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

    candidate_order = [item[2] for item in selected]
    selection_reasons = {item[2]: item[3] for item in selected}
    routing_signals = {
        "architecture_family": hardware_profile.get("architecture_family"),
        "effective_gemm_tflops": signals.effective_gemm_tflops,
        "effective_bandwidth_gbps": signals.effective_bandwidth_gbps,
        "machine_ridge_flops_per_byte": signals.machine_ridge_flops_per_byte,
        "workload_intensity_to_ridge": signals.intensity_to_ridge,
        "attention_matrix_to_l2": signals.attention_to_l2,
        "estimated_blocks_per_sm": signals.blocks_per_sm,
        "softmax_to_compute_lower_bound": (signals.softmax_to_compute_lower_bound),
        "launch_dominant": signals.launch_dominant,
        "graph_replay_is_cheaper": signals.graph_replay_is_cheaper,
    }
    routing_signals = {
        name: round(value, 6) if isinstance(value, float) else value
        for name, value in routing_signals.items()
    }
    return {
        "source": "hardware_cost_model",
        "decision_scope": "candidate_order_only",
        "requires_full_workload_measurement": True,
        "bottleneck_class": signals.bottleneck_class,
        "workload_analysis": analysis.as_dict(),
        "routing_signals": routing_signals,
        "candidate_order": candidate_order,
        "selection_reasons": selection_reasons,
        "capability_rejections": rejections,
    }


__all__ = ["WorkloadAnalysis", "analyze_workload", "build_routing_plan"]
