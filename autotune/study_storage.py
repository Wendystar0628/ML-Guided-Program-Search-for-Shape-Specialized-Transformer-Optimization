"""Minimal Optuna persistence for branch-local TPE studies."""

from __future__ import annotations

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

    @property
    def study_name(self) -> str:
        return f"{_slug(self.case_id)}-{_slug(self.environment)}-{self.branch_id[:12]}"


class SearchStorage:
    """One SQLite database; Optuna owns all trial persistence."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.database_path = self.root / "search.sqlite3"

    @property
    def database_url(self) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.database_path.as_posix()}"


__all__ = ["SearchStorage", "StudyIdentity"]
