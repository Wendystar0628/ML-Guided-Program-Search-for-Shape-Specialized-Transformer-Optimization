"""Single, shape-independent registry for Transformer execution policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class ExecutionComponent(StrEnum):
    """Observable capabilities that make one policy distinct from another."""

    CAUSAL_SDPA = "causal_sdpa"
    CUDA_GRAPH = "cuda_graph"
    COMPILED_RESIDUAL_LAYER_NORM = "compiled_residual_layer_norm"
    MIXED_FP16_EFFICIENT_ATTENTION = "mixed_fp16_efficient_attention"


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """One execution composition, independent of workload shape and hardware."""

    policy_id: str
    attention: str = "safe_streaming"
    use_cuda_graph: bool = False
    use_compiled_residual_layer_norm: bool = False
    routable: bool = True

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id must not be empty")
        if self.attention not in {
            "safe_streaming",
            "causal_sdpa",
            "mixed_fp16_efficient",
        }:
            raise ValueError(f"unsupported attention backend: {self.attention}")

    @property
    def required_components(self) -> frozenset[ExecutionComponent]:
        """Derive capabilities from behavior instead of duplicating policy state."""

        components: set[ExecutionComponent] = set()
        if self.attention == "causal_sdpa":
            components.add(ExecutionComponent.CAUSAL_SDPA)
        elif self.attention == "mixed_fp16_efficient":
            components.add(ExecutionComponent.MIXED_FP16_EFFICIENT_ATTENTION)
        if self.use_cuda_graph:
            components.add(ExecutionComponent.CUDA_GRAPH)
        if self.use_compiled_residual_layer_norm:
            components.add(ExecutionComponent.COMPILED_RESIDUAL_LAYER_NORM)
        return frozenset(components)


_POLICY_SPECS = {
    "auto": PolicySpec(
        "auto",
        attention="causal_sdpa",
    ),
    "safe": PolicySpec("safe", routable=False),
    "graph": PolicySpec(
        "graph",
        attention="causal_sdpa",
        use_cuda_graph=True,
    ),
    "graph-fused-norm": PolicySpec(
        "graph-fused-norm",
        attention="causal_sdpa",
        use_cuda_graph=True,
        use_compiled_residual_layer_norm=True,
    ),
    "mixed-fp16-efficient": PolicySpec(
        "mixed-fp16-efficient",
        attention="mixed_fp16_efficient",
    ),
    "graph-mixed-fp16-efficient": PolicySpec(
        "graph-mixed-fp16-efficient",
        attention="mixed_fp16_efficient",
        use_cuda_graph=True,
    ),
}

POLICY_SPECS: Mapping[str, PolicySpec] = MappingProxyType(_POLICY_SPECS)
ROUTABLE_POLICY_IDS = frozenset(
    policy_id for policy_id, spec in POLICY_SPECS.items() if spec.routable
)
POLICY_SELECTORS = frozenset({"dispatch"})


def get_policy_spec(policy: str) -> PolicySpec:
    """Return the registered definition for an explicit execution policy."""

    normalized = policy.strip().lower()
    try:
        return POLICY_SPECS[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(POLICY_SPECS))
        raise ValueError(
            f"unknown runtime policy={policy!r}; expected one of {choices}"
        ) from exc


def policy_ids() -> frozenset[str]:
    """Return every policy accepted by explicit execution."""

    return frozenset(POLICY_SPECS)


__all__ = [
    "POLICY_SELECTORS",
    "POLICY_SPECS",
    "ROUTABLE_POLICY_IDS",
    "ExecutionComponent",
    "PolicySpec",
    "get_policy_spec",
    "policy_ids",
]
