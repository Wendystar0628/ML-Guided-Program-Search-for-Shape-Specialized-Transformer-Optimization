"""Persistent cache for Enhanced measurements only."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .evaluation import (
    ConstraintVector,
    EvaluationScope,
    Fidelity,
    TrialMeasurement,
)


def _measurement_payload(measurement: TrialMeasurement) -> str:
    document = {
        "scope": measurement.scope.value,
        "objective_ms": measurement.objective_ms,
        "constraints": list(measurement.constraints.as_tuple()),
        "median_ms": measurement.median_ms,
        "p90_ms": measurement.p90_ms,
        "peak_memory_bytes": measurement.peak_memory_bytes,
        "max_tolerance_ratio": measurement.max_tolerance_ratio,
        "expected_execution_signature": (
            None
            if measurement.expected_execution_signature is None
            else dict(measurement.expected_execution_signature)
        ),
        "actual_execution_signature": (
            None
            if measurement.actual_execution_signature is None
            else dict(measurement.actual_execution_signature)
        ),
        "failure_kind": measurement.failure_kind,
        "metrics": dict(measurement.metrics),
    }
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _measurement_from_payload(
    *,
    config_id: str,
    fidelity: Fidelity,
    payload: str,
) -> TrialMeasurement:
    value: Any = json.loads(payload)
    if not isinstance(value, dict):
        raise TypeError("cached evaluation payload must be an object")
    return TrialMeasurement(
        config_id=config_id,
        fidelity=fidelity,
        scope=EvaluationScope(value["scope"]),
        objective_ms=float(value["objective_ms"]),
        constraints=ConstraintVector.from_value(value["constraints"]),
        median_ms=value.get("median_ms"),
        p90_ms=value.get("p90_ms"),
        peak_memory_bytes=value.get("peak_memory_bytes"),
        max_tolerance_ratio=value.get("max_tolerance_ratio"),
        expected_execution_signature=value.get("expected_execution_signature"),
        actual_execution_signature=value.get("actual_execution_signature"),
        failure_kind=value.get("failure_kind"),
        metrics=value.get("metrics") or {},
    )


class EvaluationCache:
    """Store reusable Enhanced evidence beside the Optuna database.

    ``location`` may be either the search-state root or the existing SQLite
    database path. Screen observations remain exclusively in Optuna, while
    Formal comparisons are always measured again.
    """

    def __init__(self, location: Path) -> None:
        path = Path(location).resolve()
        self.database_path = (
            path if path.suffix == ".sqlite3" else path / "search.sqlite3"
        )

    def get(
        self,
        *,
        case_id: str,
        evidence_identity: str,
        config_id: str,
        fidelity: Fidelity,
    ) -> TrialMeasurement | None:
        """Return matching Enhanced evidence; all other fidelities bypass."""

        normalized = Fidelity(fidelity)
        if normalized is not Fidelity.ENHANCED:
            return None
        self._validate_key(case_id, evidence_identity, config_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload
                FROM evaluations
                WHERE case_id = ?
                  AND evidence_identity = ?
                  AND config_id = ?
                  AND fidelity = ?
                """,
                (case_id, evidence_identity, config_id, normalized.value),
            ).fetchone()
        if row is None:
            return None
        return _measurement_from_payload(
            config_id=config_id,
            fidelity=normalized,
            payload=str(row[0]),
        )

    def put(
        self,
        *,
        case_id: str,
        evidence_identity: str,
        measurement: TrialMeasurement,
    ) -> bool:
        """Persist Enhanced evidence and report whether it was cacheable."""

        if measurement.fidelity is not Fidelity.ENHANCED:
            return False
        self._validate_key(case_id, evidence_identity, measurement.config_id)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO evaluations (
                    case_id,
                    evidence_identity,
                    config_id,
                    fidelity,
                    payload
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (
                    case_id,
                    evidence_identity,
                    config_id,
                    fidelity
                ) DO UPDATE SET payload = excluded.payload
                """,
                (
                    case_id,
                    evidence_identity,
                    measurement.config_id,
                    measurement.fidelity.value,
                    _measurement_payload(measurement),
                ),
            )
        return True

    @staticmethod
    def _validate_key(
        case_id: str,
        evidence_identity: str,
        config_id: str,
    ) -> None:
        for name, value in (
            ("case_id", case_id),
            ("evidence_identity", evidence_identity),
            ("config_id", config_id),
        ):
            if not value:
                raise ValueError(f"{name} must not be empty")

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                case_id TEXT NOT NULL,
                evidence_identity TEXT NOT NULL,
                config_id TEXT NOT NULL,
                fidelity TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (
                    case_id,
                    evidence_identity,
                    config_id,
                    fidelity
                )
            )
            """
        )
        return connection


__all__ = ["EvaluationCache"]
