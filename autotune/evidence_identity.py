"""Stable identities for reusable Screen, Enhanced, and Formal evidence."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarking.protocols import MeasurementProtocol
from deployment.environment import (
    ImplementationScope,
    official_definitions_digest,
    solution_implementation_digest,
    stable_digest,
)

from .evaluation import RESIDENT_PROTOCOLS, STREAMED_PROTOCOLS, Fidelity
from .promotion import (
    PROMOTION_BASE_RATIO,
    PROMOTION_BASE_WINS,
    PROMOTION_MAX_BLOCKS,
    PROMOTION_STAGES,
)

_MEASUREMENT_SOURCE_PATHS = (
    Path("autotune/evaluation.py"),
    Path("autotune/search_sweep.py"),
    Path("benchmarking/measure.py"),
)
_SCOPE_MEASUREMENT_SOURCE_PATHS = {
    ImplementationScope.RESIDENT: (Path("benchmarking/resident_measure.py"),),
    ImplementationScope.SHAPE14: (Path("benchmarking/shape14_measure.py"),),
}


def _source_digest(project_root: Path, relative_paths: tuple[Path, ...]) -> str:
    records = []
    for relative_path in sorted(relative_paths, key=lambda path: path.as_posix()):
        source = project_root / relative_path
        if not source.is_file():
            continue
        content = source.read_bytes()
        records.append(
            {
                "path": relative_path.as_posix(),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return stable_digest(records)


def _protocol_payload(
    fidelity: Fidelity,
    scope: ImplementationScope,
) -> dict[str, object]:
    defaults = MeasurementProtocol(
        accuracy_trials=1,
        warmup=0,
        repeats=1,
        rounds=1,
    )
    protocols = (
        STREAMED_PROTOCOLS
        if scope is ImplementationScope.SHAPE14
        else RESIDENT_PROTOCOLS
    )
    return {
        "measurement": asdict(protocols[fidelity]),
        "seed": defaults.seed,
        "rtol": defaults.rtol,
        "atol": defaults.atol,
    }


@dataclass(frozen=True, slots=True)
class EvidenceIdentity:
    """Independent compatibility keys for each reusable measurement stage."""

    search: str
    enhanced: str
    promotion: str


def evidence_identity(
    project_root: Path,
    *,
    scope: ImplementationScope = ImplementationScope.RESIDENT,
) -> EvidenceIdentity:
    """Describe only sources and parameters that can change measured evidence."""

    root = Path(project_root).resolve()
    normalized_scope = ImplementationScope(scope)
    measurement_sources = (
        _MEASUREMENT_SOURCE_PATHS + (_SCOPE_MEASUREMENT_SOURCE_PATHS[normalized_scope])
    )
    official = official_definitions_digest(root)
    solution = solution_implementation_digest(root, scope=normalized_scope)
    search = stable_digest(
        {
            "schema": 3,
            "scope": normalized_scope.value,
            "official": official,
            "solution": solution,
            "sources": _source_digest(root, measurement_sources),
            "protocol": _protocol_payload(Fidelity.SCREEN, normalized_scope),
            "objective": "median_ms_with_feasibility_constraints",
        }
    )
    enhanced = stable_digest(
        {
            "schema": 3,
            "scope": normalized_scope.value,
            "official": official,
            "solution": solution,
            "sources": _source_digest(root, measurement_sources),
            "protocol": _protocol_payload(Fidelity.ENHANCED, normalized_scope),
            "objective": "median_ms_with_feasibility_constraints",
        }
    )
    promotion = stable_digest(
        {
            "schema": 3,
            "scope": normalized_scope.value,
            "official": official,
            "solution": solution,
            "sources": _source_digest(root, measurement_sources),
            "protocol": _protocol_payload(Fidelity.FORMAL, normalized_scope),
            "rule": {
                "base_ratio": PROMOTION_BASE_RATIO,
                "base_wins": PROMOTION_BASE_WINS,
                "max_blocks": PROMOTION_MAX_BLOCKS,
                "stages": PROMOTION_STAGES,
            },
        }
    )
    return EvidenceIdentity(
        search=search,
        enhanced=enhanced,
        promotion=promotion,
    )


__all__ = ["EvidenceIdentity", "evidence_identity"]
