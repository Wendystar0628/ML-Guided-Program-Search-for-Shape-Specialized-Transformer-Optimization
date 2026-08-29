from runner.benchmark import BenchmarkResult, TimingStats
from solution.config import portable_config


def test_benchmark_result_serializes_only_the_compact_public_fields() -> None:
    config = portable_config()
    signature = {"config_id": config.config_id, "runtime_backend": "eager"}
    result = BenchmarkResult(
        case_id="official_01",
        config=config,
        passed=True,
        max_tolerance_ratio=0.25,
        optimized=TimingStats(median_ms=2.0, p90_ms=2.2),
        baseline=TimingStats(median_ms=4.0, p90_ms=4.5),
        peak_memory_bytes=1024,
        expected_execution_signature=signature,
        actual_execution_signature=dict(signature),
    )

    assert result.to_dict() == {
        "case_id": "official_01",
        "config_id": config.config_id,
        "passed": True,
        "max_tolerance_ratio": 0.25,
        "median_ms": 2.0,
        "p90_ms": 2.2,
        "baseline_median_ms": 4.0,
        "speedup": 2.0,
        "peak_memory_bytes": 1024,
        "execution_matches": True,
    }
