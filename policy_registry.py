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
    INPLACE_EXACT_GELU = "inplace_exact_gelu"


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """One execution composition, independent of workload shape and hardware."""

    policy_id: str
    attention: str = "safe"
    use_cuda_graph: bool = False
    use_inplace_exact_gelu: bool = False
    required_components: frozenset[ExecutionComponent] = frozenset()
    routable: bool = True

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id must not be empty")
        if self.attention not in {"safe", "causal_sdpa"}:
            raise ValueError(f"unsupported attention backend: {self.attention}")


_CAUSAL_SDPA = frozenset({ExecutionComponent.CAUSAL_SDPA})

_POLICY_SPECS = {
    "auto": PolicySpec(
        "auto",
        attention="causal_sdpa",
        required_components=_CAUSAL_SDPA,
    ),
    "safe": PolicySpec("safe", routable=False),
    "graph": PolicySpec(
        "graph",
        attention="causal_sdpa",
        use_cuda_graph=True,
        required_components=frozenset(
            {ExecutionComponent.CAUSAL_SDPA, ExecutionComponent.CUDA_GRAPH}
        ),
    ),
    "inplace-block": PolicySpec(
        "inplace-block",
        attention="causal_sdpa",
        use_inplace_exact_gelu=True,
        required_components=frozenset(
            {
                ExecutionComponent.CAUSAL_SDPA,
                ExecutionComponent.INPLACE_EXACT_GELU,
            }
        ),
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
