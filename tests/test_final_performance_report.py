from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_report_module() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "04_最终交付物"
        / "01_最终性能测试"
        / "run_final_performance.py"
    )
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
    markdown = module.render_markdown(report)
    assert "Shapes 01-13 geometric mean speedup: 2.0000x" in markdown
    assert "| official_14 | N/A | 10.000 | 11.000 | N/A |" in markdown


def test_main_writes_one_json_and_one_markdown_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_report_module()
    output_root = tmp_path / "result"
    monkeypatch.setattr(module, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(module, "WORKING_ROOT", output_root / ".working")
    monkeypatch.setattr(module, "JSON_OUTPUT", output_root / "final_performance.json")
    monkeypatch.setattr(module, "MARKDOWN_OUTPUT", output_root / "final_performance.md")
    workload_path = tmp_path / "official" / "test_shapes.json"
    workload_path.parent.mkdir()
    workload_path.write_text(
        json.dumps(
            {
                "ordered_shapes": [
                    {"case_id": "official_01", "batch_size": 1},
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
        output.mkdir(parents=True)
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "preset": "smoke",
                    "variant": {"dtype": "float32"},
                    "resident_geomean_speedup": None,
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
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(module, "_run", fake_run)

    assert module.main(["--preset", "smoke"]) == 0
    assert module.JSON_OUTPUT.is_file()
    assert module.MARKDOWN_OUTPUT.is_file()
    assert not module.WORKING_ROOT.exists()
