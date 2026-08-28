"""Canonical locations for generated benchmark artifacts."""

from __future__ import annotations

from pathlib import Path


def intermediate_results_root(project_root: Path) -> Path:
    """Return the single root for mutable, non-release measurement artifacts."""

    return project_root.resolve() / "results" / "intermediate"


def intermediate_results_dir(project_root: Path, category: str) -> Path:
    """Return one semantic directory below the intermediate artifact root."""

    return intermediate_results_root(project_root) / category


def final_results_root(project_root: Path) -> Path:
    """Return the tracked root for concise final performance artifacts."""

    return project_root.resolve() / "results" / "final"


def final_performance_path(project_root: Path, hardware_id: str) -> Path:
    """Return the one final performance artifact owned by a hardware bundle."""

    if not hardware_id or Path(hardware_id).name != hardware_id:
        raise ValueError("hardware_id must be one non-empty path component")
    return final_results_root(project_root) / f"{hardware_id}.json"


__all__ = [
    "final_performance_path",
    "final_results_root",
    "intermediate_results_dir",
    "intermediate_results_root",
]
