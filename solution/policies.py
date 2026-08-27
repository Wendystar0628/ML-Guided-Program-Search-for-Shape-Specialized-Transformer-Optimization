"""Typed runtime-policy definitions for the optimized Transformer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """One complete, immutable execution-policy configuration."""

    policy_id: str
    qkv_layout: str = "auto"
    attention: str = "auto"
    use_packed_ffn: bool = False
    use_triton_residual: bool = False
    ffn: str = "exact"
    use_cuda_graph: bool = False
    tail_attention: str | None = None

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

    def attention_for_layer(self, layer_index: int, layer_count: int) -> str:
        """Return the requested attention implementation for one layer."""

        if self.tail_attention is not None and layer_index == layer_count - 1:
            return self.tail_attention
        return self.attention


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
    ),
    "preprocess": PolicySpec(
        "preprocess",
        qkv_layout="view",
        attention="triton_preprocess",
    ),
    "s512-native-softmax": PolicySpec(
        "s512-native-softmax",
        qkv_layout="view",
        attention="s512_native_half_softmax",
    ),
    "long-tail-online": PolicySpec(
        "long-tail-online",
        qkv_layout="view",
        tail_attention="triton_online",
    ),
    "wide-triton-inplace": PolicySpec(
        "wide-triton-inplace",
        qkv_layout="triton",
        ffn="inplace_exact_gelu",
    ),
    "cuda-graph": PolicySpec("cuda-graph", use_cuda_graph=True),
    "balanced-cuda-graph": PolicySpec(
        "balanced-cuda-graph",
        use_cuda_graph=True,
    ),
    "padding": PolicySpec(
        "padding",
        qkv_layout="view",
        use_triton_residual=True,
    ),
    "packed": PolicySpec(
        "packed",
        qkv_layout="view",
        use_packed_ffn=True,
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
    "PolicySpec",
    "get_policy_spec",
    "policy_ids",
]
