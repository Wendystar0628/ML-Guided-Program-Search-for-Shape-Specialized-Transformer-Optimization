"""Cross-shape warm starts for branch-local program search."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import optuna
from optuna.study import StudySummary

from benchmarking.protocols import RunVariant, TransformerShape
from solution.config import ConfigSpec

from .evaluation import Fidelity
from .optuna_backend import measurement_from_frozen_trial
from .search_space import ProgramSearchSpace
from .study_storage import SearchStorage, study_name_prefix

META_NEIGHBOR_COUNT = 3
MAX_META_WARM_STARTS = 4


@dataclass(frozen=True, slots=True)
class WarmStartCandidate:
    """One source-task configuration that may initialize a target search."""

    shape: TransformerShape
    variant: RunVariant
    config: ConfigSpec
    evidence_priority: int
    source_order: int


def load_study_summaries(storage: SearchStorage) -> tuple[StudySummary, ...]:
    """Load Optuna summaries once; an absent database has no history."""

    if not storage.database_path.exists():
        return ()
    return tuple(optuna.get_all_study_summaries(storage=storage.database_url))


def best_screen_candidates(
    summaries: Sequence[StudySummary],
    *,
    shape: TransformerShape,
    variant: RunVariant,
    environment: str,
    search_identity: str,
    source_order: int,
) -> tuple[WarmStartCandidate, ...]:
    """Return the best feasible Screen configuration for one historical task."""

    prefix = study_name_prefix(shape.case_id, environment, search_identity)
    best: tuple[float, ConfigSpec] | None = None
    for summary in summaries:
        if not summary.study_name.startswith(prefix):
            continue
        trial = summary.best_trial
        if trial is None:
            continue
        try:
            measurement = measurement_from_frozen_trial(trial)
            config = ConfigSpec.from_dict(trial.user_attrs.get("config"))
        except (TypeError, ValueError):
            continue
        if measurement.fidelity is not Fidelity.SCREEN or not measurement.feasible:
            continue
        item = (measurement.objective_ms, config)
        if best is None or item[0] < best[0]:
            best = item
    if best is None:
        return ()
    return (
        WarmStartCandidate(
            shape=shape,
            variant=variant,
            config=best[1],
            evidence_priority=2,
            source_order=source_order,
        ),
    )


def _shape_vector(shape: TransformerShape) -> tuple[float, ...]:
    # Head dimension is derived from model width and heads, so including all
    # three would count the same degree of freedom twice.
    return (
        math.log2(shape.batch_size),
        math.log2(shape.seq_len),
        math.log2(shape.d_model),
        math.log2(shape.num_heads),
        math.log2(shape.ffn_dim),
        float(shape.num_layers),
    )


def _standardized_distance(
    left: TransformerShape,
    right: TransformerShape,
    reference_shapes: Sequence[TransformerShape],
) -> float:
    vectors = [_shape_vector(shape) for shape in reference_shapes]
    left_vector = _shape_vector(left)
    right_vector = _shape_vector(right)
    distance_squared = 0.0
    for index, column in enumerate(zip(*vectors, strict=True)):
        mean = sum(column) / len(column)
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        if variance == 0.0:
            continue
        scale = math.sqrt(variance)
        delta = (left_vector[index] - right_vector[index]) / scale
        distance_squared += delta * delta
    return math.sqrt(distance_squared)


def select_meta_warm_starts(
    *,
    candidates: Iterable[WarmStartCandidate],
    target: TransformerShape,
    variant: RunVariant,
    reference_shapes: Sequence[TransformerShape],
    incumbent: ConfigSpec | None,
    search_space: ProgramSearchSpace,
    neighbor_count: int = META_NEIGHBOR_COUNT,
    limit: int = MAX_META_WARM_STARTS,
) -> tuple[ConfigSpec, ...]:
    """Select accepted seeds from the nearest measured source tasks."""

    if neighbor_count <= 0 or limit <= 0:
        return ()
    references = tuple(
        shape for shape in reference_shapes if shape.streamed == target.streamed
    )
    if not references:
        return ()

    by_task: dict[str, list[WarmStartCandidate]] = defaultdict(list)
    source_shapes: dict[str, TransformerShape] = {}
    for candidate in candidates:
        source = candidate.shape
        if source.case_id == target.case_id or source.streamed != target.streamed:
            continue
        if candidate.variant != variant or source.causal != target.causal:
            continue
        by_task[source.case_id].append(candidate)
        source_shapes[source.case_id] = source
    nearest = sorted(
        by_task,
        key=lambda case_id: (
            _standardized_distance(
                source_shapes[case_id],
                target,
                references,
            ),
            case_id,
        ),
    )[:neighbor_count]
    for case_id in nearest:
        by_task[case_id].sort(
            key=lambda candidate: (
                candidate.evidence_priority,
                candidate.source_order,
                candidate.config.config_id,
            )
        )

    selected: list[ConfigSpec] = []
    seen = {incumbent.config_id} if incumbent is not None else set()
    depth = 0
    while nearest and len(selected) < limit:
        progressed = False
        for case_id in nearest:
            task_candidates = by_task[case_id]
            if depth >= len(task_candidates):
                continue
            progressed = True
            config = task_candidates[depth].config
            if config.config_id in seen or not search_space.accepted(config):
                continue
            seen.add(config.config_id)
            selected.append(config)
            if len(selected) >= limit:
                break
        if not progressed:
            break
        depth += 1
    return tuple(selected)


__all__ = [
    "MAX_META_WARM_STARTS",
    "META_NEIGHBOR_COUNT",
    "WarmStartCandidate",
    "best_screen_candidates",
    "load_study_summaries",
    "select_meta_warm_starts",
]
