"""Stable identities for reusable Screen, Enhanced, and Formal evidence."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarking.protocols import MeasurementProtocol
from deployment.environment import (
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

_SCREEN_SOURCE_PATHS = (
    Path("autotune/evaluation.py"),
    Path("autotune/search_sweep.py"),
    Path("benchmarking/measure.py"),
)

_ENHANCED_SOURCE_PATHS = (
    Path("autotune/evaluation.py"),
    Path("autotune/search_sweep.py"),
    Path("benchmarking/measure.py"),
)

_PROMOTION_SOURCE_PATHS = (
    Path("autotune/evaluation.py"),
    Path("autotune/promotion.py"),
    Path("autotune/search_sweep.py"),
    Path("benchmarking/measure.py"),
)


def _source_digest(project_root: Path, relative_paths: tuple[Path, ...]) -> str:
    records = []
    for relative_path in sorted(relative_paths, key=lambda path: path.as_posix()):
        content = (project_root / relative_path).read_bytes()
        records.append(
            {
                "path": relative_path.as_posix(),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return stable_digest(records)


def _protocol_payload(fidelity: Fidelity) -> dict[str, object]:
    defaults = MeasurementProtocol(
        accuracy_trials=1,
        warmup=0,
        repeats=1,
        rounds=1,
    )
    return {
        "resident": asdict(RESIDENT_PROTOCOLS[fidelity]),
        "streamed": asdict(STREAMED_PROTOCOLS[fidelity]),
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


def evidence_identity(project_root: Path) -> EvidenceIdentity:
    """Describe only sources and parameters that can change measured evidence."""

    root = Path(project_root).resolve()
    execution = {
        "official": official_definitions_digest(root),
        "solution": solution_implementation_digest(root),
    }
    search = stable_digest(
        {
            "schema": 1,
            "execution": execution,
            "sources": _source_digest(root, _SCREEN_SOURCE_PATHS),
            "protocol": _protocol_payload(Fidelity.SCREEN),
            "objective": "median_ms_with_feasibility_constraints",
        }
    )
    enhanced = stable_digest(
        {
            "schema": 1,
            "execution": execution,
            "sources": _source_digest(root, _ENHANCED_SOURCE_PATHS),
            "protocol": _protocol_payload(Fidelity.ENHANCED),
            "objective": "median_ms_with_feasibility_constraints",
        }
    )
    promotion = stable_digest(
        {
            "schema": 1,
            "execution": execution,
            "sources": _source_digest(root, _PROMOTION_SOURCE_PATHS),
            "protocol": _protocol_payload(Fidelity.FORMAL),
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
