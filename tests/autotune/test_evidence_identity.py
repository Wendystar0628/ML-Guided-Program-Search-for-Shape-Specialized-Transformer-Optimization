from __future__ import annotations

from pathlib import Path

from autotune import evidence_identity as identity_module
from autotune.evaluation import (
    RESIDENT_PROTOCOLS,
    STREAMED_PROTOCOLS,
    Fidelity,
    FidelityProtocol,
)
from autotune.study_storage import SearchStorage, StudyIdentity, study_name_prefix
from deployment.environment import ImplementationScope


def _project(root: Path) -> Path:
    files = {
        "official/test_shapes.json": "{}",
        "official/torch_transformer_benchmark.py": "# official\n",
        "solution/transformer.py": "# solution\n",
        "solution/kernels/attention/triton_dh8.py": "# resident\n",
        "solution/shape14/triton_streaming_dh64.py": "# shape14\n",
        "autotune/evaluation.py": "# evaluation\n",
        "autotune/promotion.py": "# promotion\n",
        "autotune/search_space.py": "# search space\n",
        "autotune/search_sweep.py": "# evaluator bridge\n",
        "benchmarking/measure.py": "# measurement dispatch\n",
        "benchmarking/measurement_core.py": "# shared measurement\n",
        "benchmarking/resident_measure.py": "# resident measurement\n",
        "benchmarking/shape14_measure.py": "# Shape 14 measurement\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def test_identity_ignores_docs_and_search_algorithm_sources(tmp_path) -> None:
    root = _project(tmp_path)
    initial = identity_module.evidence_identity(root)

    documentation = root / "docs" / "notes.md"
    documentation.parent.mkdir()
    documentation.write_text("not evidence", encoding="utf-8")
    assert identity_module.evidence_identity(root) == initial

    (root / "autotune" / "search_space.py").write_text(
        "# changed search space\n",
        encoding="utf-8",
    )
    search_changed = identity_module.evidence_identity(root)
    assert search_changed == initial

    (root / "autotune" / "promotion.py").write_text(
        "# changed non-measure source\n",
        encoding="utf-8",
    )
    assert identity_module.evidence_identity(root) == initial


def test_execution_change_invalidates_all_evidence_kinds(tmp_path) -> None:
    root = _project(tmp_path)
    initial = identity_module.evidence_identity(root)
    (root / "solution" / "transformer.py").write_text(
        "# changed execution\n",
        encoding="utf-8",
    )

    changed = identity_module.evidence_identity(root)

    assert changed.search != initial.search
    assert changed.enhanced != initial.enhanced
    assert changed.promotion != initial.promotion


def test_official_definition_change_invalidates_all_evidence_kinds(tmp_path) -> None:
    root = _project(tmp_path)
    initial = identity_module.evidence_identity(root)
    (root / "official/test_shapes.json").write_text(
        '{"changed": true}',
        encoding="utf-8",
    )

    changed = identity_module.evidence_identity(root)

    assert changed.search != initial.search
    assert changed.enhanced != initial.enhanced
    assert changed.promotion != initial.promotion


def test_evaluator_bridge_change_invalidates_all_evidence_kinds(tmp_path) -> None:
    root = _project(tmp_path)
    initial = identity_module.evidence_identity(root)
    (root / "autotune" / "search_sweep.py").write_text(
        "# changed evaluator bridge\n",
        encoding="utf-8",
    )

    changed = identity_module.evidence_identity(root)

    assert changed.search != initial.search
    assert changed.enhanced != initial.enhanced
    assert changed.promotion != initial.promotion


def test_shape14_sources_and_protocols_do_not_invalidate_resident_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    root = _project(tmp_path)
    resident = identity_module.evidence_identity(
        root,
        scope=ImplementationScope.RESIDENT,
    )
    shape14 = identity_module.evidence_identity(
        root,
        scope=ImplementationScope.SHAPE14,
    )

    (root / "solution/shape14/triton_streaming_dh64.py").write_text(
        "# changed Shape 14 implementation\n",
        encoding="utf-8",
    )
    (root / "benchmarking/shape14_measure.py").write_text(
        "# changed Shape 14 measurement\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(
        STREAMED_PROTOCOLS,
        Fidelity.ENHANCED,
        FidelityProtocol(2, 3, 5, 2, full_logical_batch=False),
    )

    assert (
        identity_module.evidence_identity(
            root,
            scope=ImplementationScope.RESIDENT,
        )
        == resident
    )
    assert (
        identity_module.evidence_identity(
            root,
            scope=ImplementationScope.SHAPE14,
        )
        != shape14
    )


def test_screen_protocol_changes_only_search_identity(
    tmp_path,
    monkeypatch,
) -> None:
    root = _project(tmp_path)
    initial = identity_module.evidence_identity(root)
    monkeypatch.setitem(
        RESIDENT_PROTOCOLS,
        Fidelity.SCREEN,
        FidelityProtocol(2, 3, 5, 2),
    )

    changed = identity_module.evidence_identity(root)

    assert changed.search != initial.search
    assert changed.enhanced == initial.enhanced
    assert changed.promotion == initial.promotion


def test_enhanced_protocol_changes_only_enhanced_identity(
    tmp_path,
    monkeypatch,
) -> None:
    root = _project(tmp_path)
    initial = identity_module.evidence_identity(root)
    monkeypatch.setitem(
        RESIDENT_PROTOCOLS,
        Fidelity.ENHANCED,
        FidelityProtocol(2, 4, 10, 3),
    )

    changed = identity_module.evidence_identity(root)

    assert changed.search == initial.search
    assert changed.enhanced != initial.enhanced
    assert changed.promotion == initial.promotion


def test_formal_protocol_changes_only_promotion_identity(
    tmp_path,
    monkeypatch,
) -> None:
    root = _project(tmp_path)
    initial = identity_module.evidence_identity(root)
    monkeypatch.setitem(
        RESIDENT_PROTOCOLS,
        Fidelity.FORMAL,
        FidelityProtocol(5, 20, 25, 15),
    )

    changed = identity_module.evidence_identity(root)

    assert changed.search == initial.search
    assert changed.enhanced == initial.enhanced
    assert changed.promotion != initial.promotion


def test_study_identity_can_partition_screen_evidence() -> None:
    first = StudyIdentity(
        "official_01",
        "branch-a",
        "gpu",
        search_identity="screen-v1",
    ).study_name
    second = StudyIdentity(
        "official_01",
        "branch-a",
        "gpu",
        search_identity="screen-v2",
    ).study_name

    assert first != second
    assert first.startswith(study_name_prefix("official_01", "gpu", "screen-v1"))


def test_formal_attempts_are_partitioned_by_promotion_evidence(
    tmp_path,
) -> None:
    storage = SearchStorage(tmp_path)
    request = {
        "case_id": "official_01",
        "environment": "gpu",
        "incumbent_id": "incumbent",
        "challenger_id": "challenger",
    }
    storage.record_challenger_attempt(**request, promotion_identity="formal-v1")

    lookup = {key: request[key] for key in request if key != "challenger_id"}
    assert storage.attempted_challenger_ids(
        **lookup,
        promotion_identity="formal-v1",
    ) == frozenset({"challenger"})
    assert (
        storage.attempted_challenger_ids(
            **lookup,
            promotion_identity="formal-v2",
        )
        == frozenset()
    )
