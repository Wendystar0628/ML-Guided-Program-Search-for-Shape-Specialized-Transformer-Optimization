from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from runner.contracts import ContractError
from runner.streamed_service import StreamedBenchmarkRequest, StreamedBenchmarkService
from tests.support.runner_fixtures import (
    PROJECT_ROOT,
    WORKLOAD_SET_ID,
    official_variant,
    tiny_protocol,
)


def _request(**changes: Any) -> StreamedBenchmarkRequest:
    values: dict[str, Any] = {
        "project_root": PROJECT_ROOT,
        "workload_set_id": WORKLOAD_SET_ID,
        "protocol": tiny_protocol(),
        "device": "cuda:0",
        "variant": official_variant(),
    }
    values.update(changes)
    return StreamedBenchmarkRequest(**values)


def test_service_discovers_streamed_shapes_and_uses_target_only_supervisor_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    events: list[tuple[str, str]] = []

    def fake_run(_project_root: Path, **kwargs: Any):
        calls.append(kwargs)
        result = {"outcome": "success"}
        return (
            result,
            PROJECT_ROOT / "results" / "intermediate" / "streamed" / "run.json",
        )

    monkeypatch.setattr("runner.streamed_service.run_managed_benchmark", fake_run)

    completed = StreamedBenchmarkService().run(
        _request(),
        on_case_started=lambda _index, _total, shape: events.append(
            ("started", shape.case_id)
        ),
        on_case_completed=lambda _index, _total, shape, _result, _path: events.append(
            ("completed", shape.case_id)
        ),
    )

    assert len(completed.runs) == 1
    assert [call["shape"].case_id for call in calls] == ["official_14"]
    assert calls[0]["target"] == "solution"
    assert calls[0]["solution_policy"] == "screen"
    assert calls[0]["result_dir"] == (
        PROJECT_ROOT / "results" / "intermediate" / "streamed"
    )
    assert events == [
        ("started", "official_14"),
        ("completed", "official_14"),
    ]


@pytest.mark.parametrize(
    "case_ids, message",
    [
        (("official_02",), "resident=['official_02']"),
        (("official_14", "official_14"), "must not be repeated"),
    ],
)
def test_service_rejects_invalid_explicit_case_selection(
    case_ids: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(
        ContractError, match=message.replace("[", r"\[").replace("]", r"\]")
    ):
        StreamedBenchmarkService().run(_request(case_ids=case_ids))
