from __future__ import annotations

import sqlite3
from dataclasses import replace

from autotune.evaluation import (
    ConstraintVector,
    EvaluationScope,
    Fidelity,
    TrialMeasurement,
)
from autotune.evaluation_cache import EvaluationCache


def _measurement(
    fidelity: Fidelity = Fidelity.ENHANCED,
    *,
    objective_ms: float = 1.25,
) -> TrialMeasurement:
    return TrialMeasurement(
        config_id="config-1",
        fidelity=fidelity,
        scope=EvaluationScope.RESIDENT,
        objective_ms=objective_ms,
        constraints=ConstraintVector(),
        median_ms=objective_ms,
        p90_ms=objective_ms + 0.1,
        peak_memory_bytes=4096,
        max_tolerance_ratio=0.25,
        expected_execution_signature={"attention": "sdpa"},
        actual_execution_signature={"attention": "sdpa"},
        metrics={"round_medians_ms": [objective_ms, objective_ms + 0.02]},
    )


def _identity(
    *,
    environment: str = "hardware-and-software",
) -> str:
    return f"enhanced-v1-{environment}"


def test_enhanced_measurement_round_trips_through_search_database(tmp_path) -> None:
    cache = EvaluationCache(tmp_path)
    measurement = _measurement()

    assert cache.put(
        case_id="official_01",
        evidence_identity=_identity(),
        measurement=measurement,
    )

    assert (
        cache.get(
            case_id="official_01",
            evidence_identity=_identity(),
            config_id=measurement.config_id,
            fidelity=Fidelity.ENHANCED,
        )
        == measurement
    )
    assert cache.database_path == (tmp_path / "search.sqlite3").resolve()


def test_evidence_or_environment_change_naturally_misses(tmp_path) -> None:
    cache = EvaluationCache(tmp_path)
    measurement = _measurement()
    cache.put(
        case_id="official_01",
        evidence_identity=_identity(),
        measurement=measurement,
    )

    changed_evidence = "enhanced-v2-hardware-and-software"
    changed_environment = _identity(environment="different-software")
    for identity in (changed_evidence, changed_environment):
        assert (
            cache.get(
                case_id="official_01",
                evidence_identity=identity,
                config_id=measurement.config_id,
                fidelity=Fidelity.ENHANCED,
            )
            is None
        )


def test_screen_and_formal_measurements_never_enter_cache(tmp_path) -> None:
    database_path = tmp_path / "existing.sqlite3"
    cache = EvaluationCache(database_path)

    for fidelity in (Fidelity.SCREEN, Fidelity.FORMAL):
        measurement = _measurement(fidelity)
        assert not cache.put(
            case_id="official_01",
            evidence_identity=_identity(),
            measurement=measurement,
        )
        assert (
            cache.get(
                case_id="official_01",
                evidence_identity=_identity(),
                config_id=measurement.config_id,
                fidelity=fidelity,
            )
            is None
        )

    assert not database_path.exists()


def test_put_replaces_same_evidence_key_without_adding_rows(tmp_path) -> None:
    cache = EvaluationCache(tmp_path)
    first = _measurement()
    second = replace(first, objective_ms=0.9, median_ms=0.9, p90_ms=1.0)
    for measurement in (first, second):
        cache.put(
            case_id="official_01",
            evidence_identity=_identity(),
            measurement=measurement,
        )

    cached = cache.get(
        case_id="official_01",
        evidence_identity=_identity(),
        config_id=first.config_id,
        fidelity=Fidelity.ENHANCED,
    )
    assert cached == second
    with sqlite3.connect(cache.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM evaluations").fetchone() == (1,)


def test_cache_key_dimensions_are_independent(tmp_path) -> None:
    cache = EvaluationCache(tmp_path)
    measurement = _measurement()
    cache.put(
        case_id="official_01",
        evidence_identity=_identity(),
        measurement=measurement,
    )

    for case_id, config_id in (
        ("official_02", measurement.config_id),
        ("official_01", "config-2"),
    ):
        assert (
            cache.get(
                case_id=case_id,
                evidence_identity=_identity(),
                config_id=config_id,
                fidelity=Fidelity.ENHANCED,
            )
            is None
        )
