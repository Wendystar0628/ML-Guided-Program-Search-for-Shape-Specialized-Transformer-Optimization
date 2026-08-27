"""Typed runtime-policy definitions for the optimized Transformer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

QKV_LAYOUTS = frozenset({"auto", "view", "triton", "torch_contiguous"})
ATTENTION_POLICIES = frozenset(
    {
        "auto",
        "explicit",
        "reference",
        "fp32_sdpa",
        "s512_native_half_softmax",
        "triton_preprocess",
        "triton_softmax",
        "triton_online",
    }
)
FFN_POLICIES = frozenset(
    {
        "exact",
        "inplace_exact_gelu",
    }
)


class ExecutionComponent(StrEnum):
    """Specialized implementation pieces that define a policy's identity."""

    TRITON_QKV_LAYOUT = "triton_qkv_layout"
    TRITON_ATTENTION_SOFTMAX = "triton_attention_softmax"
    TRITON_ATTENTION_PREPROCESS = "triton_attention_preprocess"
    S512_NATIVE_HALF_SOFTMAX = "s512_native_half_softmax"
    TAIL_ONLINE_ATTENTION = "tail_online_attention"
    WIDE_INPLACE_FFN = "wide_inplace_ffn"
    PACKED_FFN = "packed_ffn"
    TRITON_RESIDUAL = "triton_residual"
    CUDA_GRAPH = "cuda_graph"


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """One complete policy, including what must actually execute."""

    policy_id: str
    qkv_layout: str = "auto"
    attention: str = "auto"
    use_packed_ffn: bool = False
    use_triton_residual: bool = False
    ffn: str = "exact"
    tail_attention: str | None = None
    cuda_graph_shape: tuple[int, int, int, int, int, int] | None = None
    cuda_graph_route: str | None = None
    required_components: frozenset[ExecutionComponent] = frozenset()
    allow_partial_application: bool = False

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id must not be empty")
        if self.qkv_layout not in QKV_LAYOUTS:
            raise ValueError(f"unsupported qkv_layout: {self.qkv_layout}")
        if self.attention not in ATTENTION_POLICIES:
            raise ValueError(f"unsupported attention policy: {self.attention}")
        if self.ffn not in FFN_POLICIES:
            raise ValueError(f"unsupported FFN policy: {self.ffn}")
        if (
            self.tail_attention is not None
            and self.tail_attention not in ATTENTION_POLICIES
        ):
            raise ValueError(
                f"unsupported tail attention policy: {self.tail_attention}"
            )
        if (self.cuda_graph_shape is None) != (self.cuda_graph_route is None):
            raise ValueError(
                "cuda_graph_shape and cuda_graph_route must be configured together"
            )
        if self.allow_partial_application and len(self.required_components) < 2:
            raise ValueError(
                "partial application requires at least two required components"
            )

    @property
    def use_cuda_graph(self) -> bool:
        """Return whether this policy requests the Solution graph wrapper."""

        return self.cuda_graph_shape is not None


_POLICY_SPECS = {
    "auto": PolicySpec("auto"),
    "reference": PolicySpec(
        "reference",
        qkv_layout="view",
        attention="reference",
    ),
    "torch": PolicySpec("torch", qkv_layout="torch_contiguous"),
    "triton": PolicySpec(
        "triton",
        qkv_layout="triton",
        attention="triton_softmax",
        required_components=frozenset(
            {
                ExecutionComponent.TRITON_QKV_LAYOUT,
                ExecutionComponent.TRITON_ATTENTION_SOFTMAX,
            }
        ),
        allow_partial_application=True,
    ),
    "preprocess": PolicySpec(
        "preprocess",
        qkv_layout="view",
        attention="triton_preprocess",
        required_components=frozenset(
            {ExecutionComponent.TRITON_ATTENTION_PREPROCESS}
        ),
    ),
    "s512-native-softmax": PolicySpec(
        "s512-native-softmax",
        qkv_layout="view",
        attention="s512_native_half_softmax",
        required_components=frozenset(
            {ExecutionComponent.S512_NATIVE_HALF_SOFTMAX}
        ),
    ),
    "long-tail-online": PolicySpec(
        "long-tail-online",
        qkv_layout="view",
        tail_attention="triton_online",
        required_components=frozenset(
            {ExecutionComponent.TAIL_ONLINE_ATTENTION}
        ),
    ),
    "wide-triton-inplace": PolicySpec(
        "wide-triton-inplace",
        qkv_layout="triton",
        ffn="inplace_exact_gelu",
        required_components=frozenset(
            {
                ExecutionComponent.TRITON_QKV_LAYOUT,
                ExecutionComponent.WIDE_INPLACE_FFN,
            }
        ),
    ),
    "cuda-graph": PolicySpec(
        "cuda-graph",
        cuda_graph_shape=(1, 64, 256, 8, 1024, 4),
        cuda_graph_route="launch_fp16_eager_cuda_graph",
        required_components=frozenset({ExecutionComponent.CUDA_GRAPH}),
    ),
    "balanced-cuda-graph": PolicySpec(
        "balanced-cuda-graph",
        cuda_graph_shape=(8, 128, 512, 8, 2048, 6),
        cuda_graph_route="balanced_fp16_eager_cuda_graph",
        required_components=frozenset({ExecutionComponent.CUDA_GRAPH}),
    ),
    "padding": PolicySpec(
        "padding",
        qkv_layout="view",
        use_triton_residual=True,
        required_components=frozenset({ExecutionComponent.TRITON_RESIDUAL}),
    ),
    "packed": PolicySpec(
        "packed",
        qkv_layout="view",
        use_packed_ffn=True,
        required_components=frozenset({ExecutionComponent.PACKED_FFN}),
    ),
}

POLICY_SPECS: Mapping[str, PolicySpec] = MappingProxyType(_POLICY_SPECS)
ROUTABLE_POLICY_IDS = frozenset(POLICY_SPECS)
POLICY_SELECTORS = frozenset({"dispatch"})


def get_policy_spec(policy: str) -> PolicySpec:
    """Resolve a named policy through the single Solution policy registry."""

    normalized = policy.strip().lower()
    try:
        return POLICY_SPECS[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(ROUTABLE_POLICY_IDS))
        raise ValueError(
            f"unknown runtime policy={policy!r}; expected one of {choices}"
        ) from exc


def policy_ids() -> frozenset[str]:
    """Return all concrete policies accepted by the offline dispatcher."""

    return ROUTABLE_POLICY_IDS


__all__ = [
    "ATTENTION_POLICIES",
    "FFN_POLICIES",
    "POLICY_SELECTORS",
    "POLICY_SPECS",
    "QKV_LAYOUTS",
    "ROUTABLE_POLICY_IDS",
    "ExecutionComponent",
    "PolicySpec",
    "get_policy_spec",
    "policy_ids",
]
