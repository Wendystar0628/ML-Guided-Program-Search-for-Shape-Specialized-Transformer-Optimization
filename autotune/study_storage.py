"""Minimal Optuna persistence for branch-local TPE studies."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


def _slug(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() else "-" for character in value
    )
    return normalized.strip("-").lower() or "unknown"


@dataclass(frozen=True, slots=True)
class StudyIdentity:
    case_id: str
    branch_id: str
    environment: str
    search_identity: str

    @property
    def study_name(self) -> str:
        return (
            study_name_prefix(
                self.case_id,
                self.environment,
                self.search_identity,
            )
            + self.branch_id
        )


def study_name_prefix(
    case_id: str,
    environment: str,
    search_identity: str,
) -> str:
    """Return the stable prefix shared by one task's branch studies."""

    return (
        f"{_slug(case_id)}-{_slug(environment)}-"
        f"evidence-{_slug(search_identity)}-"
    )


class SearchStorage:
    """One SQLite database for Optuna trials and rejected challenger memory."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.database_path = self.root / "search.sqlite3"

    @property
    def database_url(self) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.database_path.as_posix()}"

    def attempted_challenger_ids(
        self,
        *,
        case_id: str,
        environment: str,
        incumbent_id: str | None,
        promotion_identity: str,
    ) -> frozenset[str]:
        """Return challengers already decided against the same incumbent."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT challenger_id
                FROM formal_attempts_by_evidence
                WHERE case_id = ? AND environment = ?
                  AND promotion_identity = ? AND incumbent_id = ?
                """,
                (
                    case_id,
                    environment,
                    promotion_identity,
                    incumbent_id or "",
                ),
            )
        return frozenset(str(row[0]) for row in rows)

    def record_challenger_attempt(
        self,
        *,
        case_id: str,
        environment: str,
        incumbent_id: str | None,
        challenger_id: str,
        promotion_identity: str,
    ) -> None:
        """Persist one rejected Formal decision for duplicate suppression."""

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO formal_attempts_by_evidence (
                    case_id, environment, promotion_identity,
                    incumbent_id, challenger_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    environment,
                    promotion_identity,
                    incumbent_id or "",
                    challenger_id,
                ),
            )

    def _connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS formal_attempts_by_evidence (
                case_id TEXT NOT NULL,
                environment TEXT NOT NULL,
                promotion_identity TEXT NOT NULL,
                incumbent_id TEXT NOT NULL,
                challenger_id TEXT NOT NULL,
                PRIMARY KEY (
                    case_id, environment, promotion_identity,
                    incumbent_id, challenger_id
                )
            )
            """
        )
        return connection


__all__ = ["SearchStorage", "StudyIdentity", "study_name_prefix"]
