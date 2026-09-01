from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_report_module() -> ModuleType:
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_final_performance.py"
    spec = importlib.util.spec_from_file_location("final_performance_report", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_report_keeps_resident_baseline_and_marks_shape14() -> None:
    module = _load_report_module()
    report = module.build_final_report(
        {
            "status": "completed",
            "preset": "formal",
            "variant": {"dtype": "float32"},
            "resident_geomean_speedup": 2.0,
            "shapes": [
                {
                    "case_id": "official_01",
                    "config_id": "resident",
                    "passed": True,
                    "execution_matches": True,
                    "baseline_median_ms": 2.0,
                    "median_ms": 1.0,
                    "p90_ms": 1.1,
                    "speedup": 2.0,
                    "peak_memory_bytes": 1024,
                    "max_tolerance_ratio": 0.5,
                },
            ],
        },
        {
            "status": "completed",
            "preset": "formal",
            "variant": {"dtype": "float32"},
            "shapes": [
                {
                    "case_id": "official_14",
                    "config_id": "streamed",
                    "passed": True,
                    "execution_matches": True,
                    "baseline_median_ms": None,
                    "median_ms": 10.0,
                    "p90_ms": 11.0,
                    "speedup": None,
                    "peak_memory_bytes": 2048,
                    "max_tolerance_ratio": 0.7,
                    "output_digest": "legacy-sampled-digest",
                },
            ],
        },
        {"name": "Test GPU", "compute_capability": "9.0"},
        {
            "ordered_shapes": [
                {"case_id": "official_01", "batch_size": 1},
                {"case_id": "official_14", "batch_size": 32},
            ]
        },
    )

    resident, streamed = report["shapes"]
    assert resident["baseline_status"] == "measured"
    assert resident["speedup"] == 2.0
    assert streamed["baseline_status"] == "not_measured_dense_s2"
    assert streamed["speedup"] is None
    assert streamed["correctness_passed"] is None
    assert streamed["local_b1_semantic_pass"] is True
    assert streamed["max_tolerance_ratio"] is None
    assert streamed["local_b1_max_tolerance_ratio"] == 0.7
    assert streamed["full_logical_execution_completed"] is False
    assert streamed["sampled_execution_digest"] == "legacy-sampled-digest"
    assert streamed["official_b32_io_pass"] is None
    assert streamed["official_b32_io_status"] == "not_available"
    assert report["schema_version"] == 2
    assert report["groups"]["resident"]["status"] == "completed"
    assert report["groups"]["shape14"]["status"] == "completed"
    markdown = module.render_markdown(report)
    assert "Shapes 01-13 geometric mean speedup: 2.0000x" in markdown
    assert (
        "| official_14 | N/A | 10.000 | 11.000 | 0.00 | PASS | NOT RUN | "
        "NOT AVAILABLE |"
    ) in markdown
    assert "Shape 14 task: `completed`" in markdown


def test_main_keeps_each_completed_result_in_timestamp_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    output_root = tmp_path / "result"
    monkeypatch.setattr(module, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(module, "WORKING_ROOT", output_root / ".working")
    timestamps = iter(
        (
            ("2026-08-30T13:00:00.000001Z", "20260830T130000.000001Z"),
            ("2026-08-30T13:00:01.000001Z", "20260830T130001.000001Z"),
            ("2026-08-30T13:00:02.000001Z", "20260830T130002.000001Z"),
            ("2026-08-30T13:00:03.000001Z", "20260830T130003.000001Z"),
        )
    )
    monkeypatch.setattr(module, "_timestamp", lambda: next(timestamps))
    workload_path = tmp_path / "official" / "test_shapes.json"
    workload_path.parent.mkdir()
    workload_path.write_text(
        json.dumps(
            {
                "ordered_shapes": [
                    {"case_id": "official_01", "batch_size": 1},
                    {"case_id": "official_14", "batch_size": 32},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    benchmark_groups: list[str] = []

    def fake_run(command: list[str]) -> int:
        output = Path(command[command.index("--output") + 1])
        if "probe" in command:
            output.write_text(json.dumps({"name": "Test GPU"}), encoding="utf-8")
            return 0
        group = command[command.index("--group") + 1]
        benchmark_groups.append(group)
        output.mkdir(parents=True)
        if group == "resident":
            summary = {
                "status": "completed",
                "preset": "smoke",
                "variant": {"dtype": "float32"},
                "resident_geomean_speedup": 2.0,
                "shapes": [
                    {
                        "case_id": "official_01",
                        "passed": True,
                        "execution_matches": True,
                        "baseline_median_ms": 2.0,
                        "median_ms": 1.0,
                        "p90_ms": 1.1,
                        "speedup": 2.0,
                        "peak_memory_bytes": 1024,
                    }
                ],
            }
        else:
            summary = {
                "status": "completed",
                "preset": "smoke",
                "variant": {"dtype": "float32"},
                "shapes": [
                    {
                        "case_id": "official_14",
                        "passed": True,
                        "execution_matches": True,
                        "baseline_median_ms": None,
                        "median_ms": 10.0,
                        "p90_ms": 11.0,
                        "speedup": None,
                        "peak_memory_bytes": 2048,
                    }
                ],
            }
        (output / "summary.json").write_text(
            json.dumps(summary),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(module, "_run", fake_run)

    assert module.main(["--preset", "smoke"]) == 0
    assert module.main(["--preset", "smoke"]) == 0
    assert benchmark_groups == ["resident", "shape14", "resident", "shape14"]

    run_directories = sorted(path for path in output_root.iterdir() if path.is_dir())
    assert [path.name for path in run_directories] == [
        "20260830T130001.000001Z",
        "20260830T130003.000001Z",
    ]
    for run_directory in run_directories:
        json_path = run_directory / "final_performance.json"
        assert json_path.is_file()
        assert (run_directory / "final_performance.md").is_file()
        result = json.loads(json_path.read_text(encoding="utf-8"))
        assert result["groups"]["resident"]["status"] == "completed"
        assert result["groups"]["shape14"]["status"] == "completed"
        assert result["resident_geomean_speedup"] == 2.0
        assert (
            result["completed_at"]
            .replace(":", "")
            .replace("-", "")
            .startswith(run_directory.name[:15])
        )
    assert not module.WORKING_ROOT.exists()


def test_shape14_failure_does_not_discard_resident_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    output_root = tmp_path / "result"
    monkeypatch.setattr(module, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(module, "WORKING_ROOT", output_root / ".working")
    timestamps = iter(
        (
            ("2026-08-30T14:00:00.000001Z", "20260830T140000.000001Z"),
            ("2026-08-30T14:00:01.000001Z", "20260830T140001.000001Z"),
        )
    )
    monkeypatch.setattr(module, "_timestamp", lambda: next(timestamps))
    workload_path = tmp_path / "official" / "test_shapes.json"
    workload_path.parent.mkdir()
    workload_path.write_text(
        json.dumps(
            {
                "ordered_shapes": [
                    {"case_id": "official_01", "batch_size": 1},
                    {"case_id": "official_14", "batch_size": 32},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)

    def fake_run(command: list[str]) -> int:
        output = Path(command[command.index("--output") + 1])
        if "probe" in command:
            output.write_text(json.dumps({"name": "Test GPU"}), encoding="utf-8")
            return 0
        group = command[command.index("--group") + 1]
        output.mkdir(parents=True)
        if group == "resident":
            summary = {
                "status": "completed",
                "preset": "smoke",
                "resident_geomean_speedup": 2.0,
                "shapes": [
                    {
                        "case_id": "official_01",
                        "passed": True,
                        "execution_matches": True,
                        "baseline_median_ms": 2.0,
                        "median_ms": 1.0,
                        "p90_ms": 1.1,
                        "speedup": 2.0,
                        "peak_memory_bytes": 1024,
                    }
                ],
            }
            exit_code = 0
        else:
            summary = {
                "status": "completed_with_failures",
                "preset": "smoke",
                "shapes": [
                    {
                        "case_id": "official_14",
                        "passed": False,
                        "execution_matches": False,
                        "median_ms": None,
                        "p90_ms": None,
                        "peak_memory_bytes": None,
                        "error": "out of memory",
                    }
                ],
                "error": "Shape 14 failed",
            }
            exit_code = 1
        (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return exit_code

    monkeypatch.setattr(module, "_run", fake_run)

    assert module.main(["--preset", "smoke"]) == 1

    result_path = output_root / "20260830T140001.000001Z" / "final_performance.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "completed_with_failures"
    assert result["groups"]["resident"]["status"] == "completed"
    assert result["groups"]["shape14"]["status"] == "completed_with_failures"
    assert result["resident_geomean_speedup"] == 2.0
    resident = next(
        shape for shape in result["shapes"] if shape["case_id"] == "official_01"
    )
    assert resident["speedup"] == 2.0
