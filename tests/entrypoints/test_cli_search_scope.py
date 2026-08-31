from __future__ import annotations

from pathlib import Path

import pytest

from autotune import search_sweep
from autotune.search_sweep import SearchSweepRequest
from benchmarking.protocols import ContractError
from cli import (
    _run_log_request,
    _search_scope,
    _search_storage_root,
    build_parser,
)
from deployment.environment import ImplementationScope


def test_benchmark_cli_has_no_combined_shape_group() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["benchmark", "--group", "all"])


def test_search_scope_rejects_resident_and_shape14_together() -> None:
    project_root = Path(__file__).resolve().parents[2]

    assert _search_scope(project_root, ("official_01",)) is ImplementationScope.RESIDENT
    assert _search_scope(project_root, ("official_14",)) is ImplementationScope.SHAPE14
    with pytest.raises(ContractError, match="cannot mix"):
        _search_scope(project_root, ("official_01", "official_14"))


@pytest.mark.parametrize("scope", tuple(ImplementationScope))
def test_default_search_database_is_partitioned_by_scope(
    tmp_path: Path,
    scope: ImplementationScope,
) -> None:
    expected_root = tmp_path / "observations" / "search" / scope.value
    assert _search_storage_root(tmp_path, scope, None) == expected_root

    request = SearchSweepRequest(
        project_root=tmp_path,
        case_ids=(
            "official_14" if scope is ImplementationScope.SHAPE14 else "official_01",
        ),
        scope=scope,
    )
    logged = _run_log_request(request)

    assert logged["scope"] == scope.value
    assert logged["structure_seed"] == request.structure_seed
    assert logged["study_database"] == str(expected_root / "search.sqlite3")


def test_explicit_search_storage_is_partitioned_by_scope(tmp_path: Path) -> None:
    requested = tmp_path / "custom-study"

    assert (
        _search_storage_root(tmp_path, ImplementationScope.SHAPE14, requested)
        == requested / "shape14"
    )


def test_sweep_rejects_a_request_with_the_wrong_scope(monkeypatch) -> None:
    project_root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(search_sweep.torch.cuda, "is_available", lambda: True)

    with pytest.raises(ValueError, match="does not match"):
        search_sweep.SearchSweep(isolate_shapes=False).run(
            SearchSweepRequest(
                project_root=project_root,
                case_ids=("official_14",),
                scope=ImplementationScope.RESIDENT,
            )
        )
