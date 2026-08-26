"""Tests for offline calibration promotion and deterministic dispatch."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import triton

from runner.contracts import ContractError, solution_implementation_hash
from runner.route_promotion import (
    build_promoted_route_document,
    promote_tuning_summaries,
    promote_tuning_summary,
    select_deployable_winner,
)
from solution.dispatch import (
    ROUTE_FIELDS,
    OfflineDispatcher,
    make_route_key,
    resolve_route,
    resolve_route_result,
    validate_route_table,
)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=1,
        seq_len=2048,
        d_model=512,
        num_heads=8,
        ffn_dim=2048,
        num_layers=4,
        causal=False,
    )


def _formal_summary() -> dict[str, object]:
    return {
        "schema_version": 1,
        "tuning_id": "fixture-tuning",
        "complete": True,
        "protocol": {
            "preset": "formal",
            "seed": 1234,
            "accuracy_trials": 5,
            "rtol": 0.01,
            "atol": 0.001,
            "warmup": 20,
            "repeats": 100,
            "rounds": 3,
            "matmul_precision": "high",
            "allow_tf32": True,
        },
        "source_consistent": True,
        "source_solution_sha256": "fixture-solution-hash",
        "implementation_consistent": True,
        "source_implementation_sha256": "fixture-implementation-hash",
        "device_profile": {
            "device_type": "cuda",
            "device_name": "Fixture GPU",
            "compute_capability": "8.9",
            "platform_system": platform.system(),
            "torch": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "triton": str(triton.__version__),
        },
        "workload": {
            "case": {
                "case_id": "attention_fixture",
                "batch_size": 1,
                "seq_len": 2048,
                "d_model": 512,
                "num_heads": 8,
                "ffn_dim": 2048,
                "num_layers": 4,
                "dtype": "float16",
                "causal": False,
                "padding_ratio": 0.0,
                "input_scale": 1.0,
            }
        },
        "observations": [
            {
                "candidate_id": "compile-fast",
                "solution_policy": "auto",
                "compile_solution": True,
                "cuda_graph_solution": False,
                "outcome": "success",
                "correctness_passed": True,
                "failed_elements": 0,
                "policy_applied": True,
                "conservative_speedup": 3.0,
                "baseline_round_medians_ms": [3.0, 3.0, 3.0],
                "target_round_medians_ms": [1.0, 1.0, 1.0],
                "target_median_ms": 0.9,
                "target_p90_ms": 1.0,
                "solution_sha256": "fixture-solution-hash",
                "execution_path": {"shape_route": "compile-control"},
            },
            {
                "candidate_id": "eager-auto",
                "solution_policy": "auto",
                "compile_solution": False,
                "cuda_graph_solution": False,
                "outcome": "success",
                "correctness_passed": True,
                "failed_elements": 0,
                "policy_applied": True,
                "conservative_speedup": 1.4,
                "baseline_round_medians_ms": [1.4, 1.4, 1.4],
                "target_round_medians_ms": [1.0, 1.0, 1.0],
                "target_median_ms": 6.5,
                "target_p90_ms": 6.8,
                "solution_sha256": "fixture-solution-hash",
                "execution_path": {"shape_route": "safe-auto"},
            },
            {
                "candidate_id": "long-pv",
                "solution_policy": "long-pv",
                "compile_solution": False,
                "cuda_graph_solution": False,
                "outcome": "success",
                "correctness_passed": True,
                "failed_elements": 0,
                "policy_applied": True,
                "conservative_speedup": 1.5,
                "baseline_round_medians_ms": [1.5, 1.5, 1.5],
                "target_round_medians_ms": [1.0, 1.0, 1.0],
                "target_median_ms": 6.1,
                "target_p90_ms": 6.4,
                "solution_sha256": "fixture-solution-hash",
                "execution_path": {"shape_route": "long-pv"},
            },
        ],
    }


def _candidate_observation(
    *,
    candidate_id: str,
    policy: str,
    speedup: float,
) -> dict[str, object]:
    observation = copy.deepcopy(_formal_summary()["observations"][2])  # type: ignore[index]
    observation["candidate_id"] = candidate_id
    observation["solution_policy"] = policy
    observation["conservative_speedup"] = speedup
    observation["baseline_round_medians_ms"] = [speedup, speedup, speedup]
    observation["execution_path"] = {"shape_route": candidate_id}
    return observation


def _existing_exact_route(policy: str) -> dict[str, object]:
    return {
        "schema_version": 2,
        "default_policy": "auto",
        "routes": [
            {
                "match": {
                    "dtype": "float16",
                    "B": 1,
                    "S": 2048,
                    "D": 512,
                    "heads": 8,
                    "ffn": 2048,
                    "layers": 4,
                    "causal": False,
                    "device_type": "cuda",
                    "device_name": "Fixture GPU",
                    "compute_capability": "8.9",
                    "platform_system": platform.system(),
                    "torch": str(torch.__version__),
                    "cuda_runtime": str(torch.version.cuda),
                    "triton": str(triton.__version__),
                },
                "policy": policy,
            }
        ],
    }


def test_route_table_resolves_an_exact_static_subset() -> None:
    table = validate_route_table(
        {
            "schema_version": 2,
            "default_policy": "auto",
            "routes": [
                {
                    "match": {
                        "device_type": "cuda",
                        "dtype": "float16",
                        "S": 2048,
                        "causal": False,
                    },
                    "policy": "long-pv",
                }
            ],
        }
    )
    matching = make_route_key(
        _config(),
        dtype=torch.float16,
        device_type="cuda",
    )
    fallback = make_route_key(
        _config(),
        dtype=torch.float32,
        device_type="cuda",
    )

    assert resolve_route(table, matching) == "long-pv"
    assert resolve_route(table, fallback) == "auto"


def test_route_key_includes_process_static_platform_and_runtime_facts() -> None:
    key = make_route_key(
        _config(),
        dtype=torch.float16,
        device_type="cuda",
    )

    assert key["platform_system"] == platform.system()
    assert key["torch"] == str(torch.__version__)
    if torch.version.cuda is None:
        assert "cuda_runtime" not in key
    else:
        assert key["cuda_runtime"] == str(torch.version.cuda)
    assert key["triton"] == str(triton.__version__)


def test_route_resolution_distinguishes_calibration_from_fallback() -> None:
    table = validate_route_table(
        {
            "schema_version": 2,
            "default_policy": "auto",
            "routes": [
                {
                    "match": {
                        "device_type": "cuda",
                        "device_name": "Fixture GPU",
                        "compute_capability": "8.9",
                        "platform_system": "Windows",
                        "torch": "2.12.1+cu132",
                        "cuda_runtime": "13.2",
                        "triton": "3.7.1",
                        "dtype": "float16",
                        "S": 2048,
                    },
                    "policy": "long-tail-online",
                }
            ],
        }
    )
    calibrated_key = make_route_key(
        _config(),
        dtype=torch.float16,
        device_type="cuda",
        device_name="Fixture GPU",
        compute_capability="8.9",
        platform_system="Windows",
        torch_version="2.12.1+cu132",
        cuda_runtime="13.2",
        triton_version="3.7.1",
    )
    fallback_key = {
        **calibrated_key,
        "platform_system": "Linux",
    }

    calibrated = resolve_route_result(table, calibrated_key)
    fallback = resolve_route_result(table, fallback_key)

    assert calibrated.policy == "long-tail-online"
    assert calibrated.origin == "calibrated"
    assert fallback.policy == "auto"
    assert fallback.origin == "fallback"


def test_checked_rtx4080_routes_are_runtime_exact_schema_v2() -> None:
    route_path = (
        Path(__file__).resolve().parents[1]
        / "verified_hardware"
        / "nvidia_geforce_rtx_4080"
        / "routes.json"
    )
    document = json.loads(route_path.read_text(encoding="utf-8"))

    assert document["schema_version"] == 2
    assert len(document["routes"]) == 8
    assert {route["policy"] for route in document["routes"]} == {
        "auto",
        "s512-native-softmax",
        "balanced-cuda-graph",
        "cuda-graph",
        "long-tail-online",
        "wide-triton-inplace",
    }
    for route in document["routes"]:
        assert set(route["match"]) == ROUTE_FIELDS
        assert route["match"]["device_name"] == "NVIDIA GeForce RTX 4080"
        assert route["match"]["compute_capability"] == "8.9"
        assert route["match"]["platform_system"] == "Windows"
        assert route["match"]["torch"] == "2.12.1+cu132"
        assert route["match"]["cuda_runtime"] == "13.2"
        assert route["match"]["triton"] == "3.7.1"


def test_dispatcher_default_catalog_reports_portable_source_and_hash() -> None:
    dispatcher = OfflineDispatcher()

    assert dispatcher.source == (
        "verified_hardware/nvidia_geforce_rtx_4080/routes.json"
    )
    assert dispatcher.table_sha256 is not None
    assert len(dispatcher.table_sha256) == 64


def test_explicit_route_table_precedes_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "explicit.json"
    configured = tmp_path / "configured.json"
    explicit.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "default_policy": "reference",
                "routes": [],
            }
        ),
        encoding="utf-8",
    )
    configured.write_text(
        json.dumps(
            {"schema_version": 2, "default_policy": "torch", "routes": []}
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRANSFORMER_ROUTE_TABLE", str(configured))

    dispatcher = OfflineDispatcher(explicit)

    assert dispatcher.table.default_policy == "reference"
    assert dispatcher.path == explicit.resolve()


def test_invalid_configured_route_table_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    monkeypatch.setenv("TRANSFORMER_ROUTE_TABLE", str(missing))

    with pytest.raises(ValueError, match="unable to load route table"):
        OfflineDispatcher()


def test_catalog_rejects_duplicate_exact_routes(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "verified_hardware"
        / "nvidia_geforce_rtx_4080"
        / "routes.json"
    ).read_text(encoding="utf-8")
    for name in ("gpu_a", "gpu_b"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "routes.json").write_text(source, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicates another exact route"):
        OfflineDispatcher(catalog_root=tmp_path)


def test_catalog_attributes_a_match_to_its_device_table(tmp_path: Path) -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "verified_hardware"
        / "nvidia_geforce_rtx_4080"
        / "routes.json"
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    route = next(
        item
        for item in source["routes"]
        if item["match"]["S"] == 2048 and item["match"]["causal"] is False
    )
    first = copy.deepcopy(route)
    second = copy.deepcopy(route)
    second["match"]["device_name"] = "Fixture GPU B"
    paths: list[Path] = []
    for directory_name, entry in (("gpu_a", first), ("gpu_b", second)):
        directory = tmp_path / directory_name
        directory.mkdir()
        path = directory / "routes.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "default_policy": "auto",
                    "routes": [entry],
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    resolution = OfflineDispatcher(catalog_root=tmp_path).resolve_result(
        _config(),
        device="cuda",
        dtype=torch.float16,
        device_name="Fixture GPU B",
        compute_capability="8.9",
        platform_system="Windows",
        torch_version="2.12.1+cu132",
        cuda_runtime="13.2",
        triton_version="3.7.1",
    )

    assert resolution.origin == "calibrated"
    assert resolution.source == str(paths[1].resolve())
    assert resolution.table_sha256 == hashlib.sha256(paths[1].read_bytes()).hexdigest()


def test_checked_rtx4080_route_fails_closed_on_runtime_drift() -> None:
    dispatcher = OfflineDispatcher()
    calibrated = dispatcher.resolve_result(
        _config(),
        device="cuda",
        dtype=torch.float16,
        device_name="NVIDIA GeForce RTX 4080",
        compute_capability="8.9",
        platform_system="Windows",
        torch_version="2.12.1+cu132",
        cuda_runtime="13.2",
        triton_version="3.7.1",
    )
    fallback = dispatcher.resolve_result(
        _config(),
        device="cuda",
        dtype=torch.float16,
        device_name="NVIDIA GeForce RTX 4080",
        compute_capability="8.9",
        platform_system="Windows",
        torch_version="2.12.1+cu132",
        cuda_runtime="13.3",
        triton_version="3.7.1",
    )

    assert calibrated.policy == "long-tail-online"
    assert calibrated.origin == "calibrated"
    assert fallback.policy == "auto"
    assert fallback.origin == "fallback"


def test_dispatcher_loads_once_and_never_reads_mask_content(tmp_path) -> None:
    route_path = tmp_path / "dispatch_routes.json"
    route_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "default_policy": "auto",
                "routes": [
                    {
                        "match": {"device_type": "cpu", "S": 2048},
                        "policy": "reference",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dispatcher = OfflineDispatcher(route_path)
    route_path.write_text("not valid json", encoding="utf-8")

    assert (
        dispatcher.resolve(
            _config(),
            device="cpu",
            dtype=torch.float32,
            shape=(1, 2048, 512),
        )
        == "reference"
    )


def test_invalid_route_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown fields"):
        validate_route_table(
            {
                "schema_version": 2,
                "default_policy": "auto",
                "routes": [
                    {
                        "match": {"padding_ratio": 0.5},
                        "policy": "packed",
                    }
                ],
            }
        )


def test_legacy_route_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="schema_version must be 2"):
        validate_route_table(
            {"schema_version": 1, "default_policy": "auto", "routes": []}
        )


def test_formal_promotion_excludes_compile_and_writes_only_static_match() -> None:
    summary = _formal_summary()
    winner = select_deployable_winner(summary)
    document, promoted = build_promoted_route_document(None, summary)

    assert winner["candidate_id"] == "long-pv"
    assert promoted == winner
    route = document["routes"][0]
    assert route["policy"] == "long-pv"
    assert route["match"]["device_name"] == "Fixture GPU"
    assert route["match"]["platform_system"] == platform.system()
    assert route["match"]["torch"] == str(torch.__version__)
    assert route["match"]["cuda_runtime"] == str(torch.version.cuda)
    assert route["match"]["triton"] == str(triton.__version__)
    assert route["match"]["B"] == 1
    assert "case_id" not in route["match"]
    assert "padding_ratio" not in route["match"]


def test_smoke_summary_cannot_change_the_dispatch_table() -> None:
    summary = _formal_summary()
    summary["protocol"] = {"preset": "smoke"}

    with pytest.raises(ContractError, match="formal"):
        build_promoted_route_document(None, summary)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("matmul_precision", "highest", "matmul_precision=high"),
        ("allow_tf32", False, "allow_tf32=true"),
    ],
)
def test_promotion_uses_one_canonical_precision_policy(
    field: str, value: object, message: str
) -> None:
    summary = _formal_summary()
    summary["protocol"][field] = value  # type: ignore[index]

    with pytest.raises(ContractError, match=message):
        build_promoted_route_document(None, summary)


def _s512_summary(case_id: str, *, padding_ratio: float) -> dict[str, object]:
    summary = _formal_summary()
    case = summary["workload"]["case"]  # type: ignore[index]
    assert isinstance(case, dict)
    case.update(
        {
            "case_id": case_id,
            "batch_size": 8,
            "seq_len": 512,
            "num_layers": 4,
            "padding_ratio": padding_ratio,
        }
    )
    winner = summary["observations"][2]  # type: ignore[index]
    assert isinstance(winner, dict)
    winner["candidate_id"] = "s512-native-softmax"
    winner["solution_policy"] = "s512-native-softmax"
    winner["execution_path"] = {"shape_route": "s512-native-softmax"}
    return summary


def test_shared_s512_route_requires_full_and_padding_formal_summaries(
    tmp_path: Path,
) -> None:
    solution_root = tmp_path / "solution"
    solution_root.mkdir()
    (solution_root / "transformer.py").write_text("VALUE = 1\n", encoding="utf-8")
    route_path = solution_root / "dispatch_routes.json"
    route_path.write_text(
        json.dumps({"schema_version": 2, "default_policy": "auto", "routes": []}),
        encoding="utf-8",
    )
    implementation_hash = solution_implementation_hash(solution_root)
    full = _s512_summary("mask_s512_full_fp16", padding_ratio=0.0)
    padding = _s512_summary("mask_s512_padding_fp16", padding_ratio=0.75)
    full["source_implementation_sha256"] = implementation_hash
    padding["source_implementation_sha256"] = implementation_hash

    with pytest.raises(ContractError, match="shared runtime route"):
        promote_tuning_summaries(tmp_path, [full], route_path=route_path)

    document, winners, _ = promote_tuning_summaries(
        tmp_path,
        [full, padding],
        route_path=route_path,
    )

    assert len(winners) == 2
    assert len(document["routes"]) == 1
    assert document["routes"][0]["policy"] == "s512-native-softmax"


def test_promotion_rejects_a_route_for_another_verified_profile(
    tmp_path: Path,
) -> None:
    solution_root = tmp_path / "solution"
    solution_root.mkdir()
    (solution_root / "transformer.py").write_text("VALUE = 1\n", encoding="utf-8")
    package = tmp_path / "verified_hardware" / "fixture_gpu"
    package.mkdir(parents=True)
    (package / "profile.json").write_text(
        json.dumps(
            {
                "hardware_profile": {
                    "device_type": "cuda",
                    "gpu": {
                        "name": "Another GPU",
                        "compute_capability": "9.0",
                    },
                    "platform": {"system": platform.system()},
                    "software": {
                        "torch": str(torch.__version__),
                        "cuda_runtime": str(torch.version.cuda),
                        "triton": str(triton.__version__),
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    summary = _formal_summary()
    summary["source_implementation_sha256"] = solution_implementation_hash(
        solution_root
    )

    with pytest.raises(ContractError, match="does not match the verified profile"):
        promote_tuning_summaries(
            tmp_path,
            [summary],
            route_path=package / "routes.json",
        )


def test_promotion_rejects_a_misspelled_verified_route_filename(
    tmp_path: Path,
) -> None:
    solution_root = tmp_path / "solution"
    solution_root.mkdir()
    (solution_root / "transformer.py").write_text("VALUE = 1\n", encoding="utf-8")
    package = tmp_path / "verified_hardware" / "fixture_gpu"
    package.mkdir(parents=True)
    summary = _formal_summary()
    summary["source_implementation_sha256"] = solution_implementation_hash(
        solution_root
    )

    with pytest.raises(ContractError, match="must be named routes.json"):
        promote_tuning_summaries(
            tmp_path,
            [summary],
            route_path=package / "route.json",
        )


def test_promotion_requires_paired_round_ranking() -> None:
    summary = _formal_summary()
    del summary["observations"][2]["conservative_speedup"]  # type: ignore[index]

    winner = select_deployable_winner(summary)

    assert winner["candidate_id"] == "eager-auto"


def test_promotion_rejects_inconsistent_observation_source() -> None:
    summary = _formal_summary()
    summary["observations"][2]["solution_sha256"] = "stale"  # type: ignore[index]

    with pytest.raises(ContractError, match="source hashes"):
        build_promoted_route_document(None, summary)


def test_promotion_rejects_a_stale_current_solution(tmp_path) -> None:
    solution_root = tmp_path / "solution"
    solution_root.mkdir()
    (solution_root / "transformer.py").write_text("VALUE = 1\n", encoding="utf-8")
    (solution_root / "dispatch_routes.json").write_text(
        json.dumps({"schema_version": 2, "default_policy": "auto", "routes": []}),
        encoding="utf-8",
    )
    summary = _formal_summary()
    summary["source_implementation_sha256"] = solution_implementation_hash(
        solution_root
    )
    for observation in summary["observations"]:  # type: ignore[union-attr]
        observation["solution_sha256"] = summary["source_solution_sha256"]
    (solution_root / "transformer.py").write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(ContractError, match="does not match"):
        promote_tuning_summary(
            tmp_path,
            summary,
            route_path=solution_root / "dispatch_routes.json",
        )


def test_promotion_rejects_an_incomplete_tuning_summary() -> None:
    summary = _formal_summary()
    summary["complete"] = False

    with pytest.raises(ContractError, match="complete"):
        build_promoted_route_document(None, summary)


def test_specialized_route_requires_a_meaningful_gain_over_auto() -> None:
    summary = _formal_summary()
    specialized = summary["observations"][2]  # type: ignore[index]
    specialized["conservative_speedup"] = 1.41
    specialized["baseline_round_medians_ms"] = [1.41, 1.41, 1.41]

    with pytest.raises(ContractError, match="promotion margin"):
        build_promoted_route_document(None, summary)


def test_replacing_an_exact_route_requires_a_gain_over_the_incumbent() -> None:
    summary = _formal_summary()
    observations = summary["observations"]
    assert isinstance(observations, list)
    observations.append(
        _candidate_observation(
            candidate_id="incumbent-preprocess",
            policy="preprocess",
            speedup=1.49,
        )
    )

    with pytest.raises(ContractError, match="incumbent.*promotion margin"):
        build_promoted_route_document(
            _existing_exact_route("preprocess"),
            summary,
        )


def test_replacing_an_exact_route_requires_an_incumbent_observation() -> None:
    with pytest.raises(ContractError, match="missing a correct incumbent observation"):
        build_promoted_route_document(
            _existing_exact_route("preprocess"),
            _formal_summary(),
        )


def test_a_clear_gain_replaces_only_the_matching_incumbent_route() -> None:
    summary = _formal_summary()
    observations = summary["observations"]
    assert isinstance(observations, list)
    observations.append(
        _candidate_observation(
            candidate_id="incumbent-preprocess",
            policy="preprocess",
            speedup=1.45,
        )
    )
    existing = _existing_exact_route("preprocess")
    routes = existing["routes"]
    assert isinstance(routes, list)
    unrelated = {"match": {"B": 8}, "policy": "reference"}
    routes.append(unrelated)

    document, winner = build_promoted_route_document(existing, summary)

    assert winner["solution_policy"] == "long-pv"
    assert unrelated in document["routes"]
    assert {
        "match": _existing_exact_route("long-pv")["routes"][0]["match"],  # type: ignore[index]
        "policy": "long-pv",
    } in document["routes"]


def test_an_incumbent_winner_keeps_its_route_without_a_new_margin() -> None:
    summary = _formal_summary()
    incumbent = summary["observations"][2]  # type: ignore[index]
    incumbent["conservative_speedup"] = 1.41
    incumbent["baseline_round_medians_ms"] = [1.41, 1.41, 1.41]
    existing = _existing_exact_route("long-pv")

    document, winner = build_promoted_route_document(existing, summary)

    assert winner["solution_policy"] == "long-pv"
    assert document == existing


def test_promoted_exact_route_precedes_a_broad_fallback() -> None:
    summary = _formal_summary()
    observations = summary["observations"]
    assert isinstance(observations, list)
    observations.append(
        _candidate_observation(
            candidate_id="incumbent-reference",
            policy="reference",
            speedup=1.45,
        )
    )
    existing = {
        "schema_version": 2,
        "default_policy": "auto",
        "routes": [{"match": {"B": 1}, "policy": "reference"}],
    }

    document, _ = build_promoted_route_document(existing, summary)
    table = validate_route_table(document)
    key = make_route_key(
        _config(),
        dtype=torch.float16,
        device_type="cuda",
        device_name="Fixture GPU",
        compute_capability="8.9",
    )

    assert resolve_route(table, key) == "long-pv"


def test_a_broad_incumbent_requires_a_formal_observation() -> None:
    existing = {
        "schema_version": 2,
        "default_policy": "auto",
        "routes": [{"match": {"B": 1}, "policy": "preprocess"}],
    }

    with pytest.raises(ContractError, match="missing a correct incumbent observation"):
        build_promoted_route_document(existing, _formal_summary())


def test_a_broad_incumbent_is_protected_by_the_promotion_margin() -> None:
    summary = _formal_summary()
    observations = summary["observations"]
    assert isinstance(observations, list)
    observations.append(
        _candidate_observation(
            candidate_id="incumbent-preprocess",
            policy="preprocess",
            speedup=1.49,
        )
    )
    existing = {
        "schema_version": 2,
        "default_policy": "auto",
        "routes": [{"match": {"B": 1}, "policy": "preprocess"}],
    }

    with pytest.raises(ContractError, match="incumbent.*promotion margin"):
        build_promoted_route_document(existing, summary)


def test_exact_auto_can_override_a_broad_specialized_route() -> None:
    summary = _formal_summary()
    observations = summary["observations"]
    assert isinstance(observations, list)
    auto = observations[1]
    assert isinstance(auto, dict)
    auto["conservative_speedup"] = 1.6
    auto["baseline_round_medians_ms"] = [1.6, 1.6, 1.6]
    observations.append(
        _candidate_observation(
            candidate_id="incumbent-preprocess",
            policy="preprocess",
            speedup=1.5,
        )
    )
    existing = {
        "schema_version": 2,
        "default_policy": "auto",
        "routes": [{"match": {"B": 1}, "policy": "preprocess"}],
    }

    document, winner = build_promoted_route_document(existing, summary)
    table = validate_route_table(document)
    key = make_route_key(
        _config(),
        dtype=torch.float16,
        device_type="cuda",
        device_name="Fixture GPU",
        compute_capability="8.9",
    )

    assert winner["solution_policy"] == "auto"
    assert resolve_route(table, key) == "auto"
    assert document["routes"][0]["policy"] == "auto"


def test_promotion_preserves_unrelated_overlapping_route_order() -> None:
    summary = _formal_summary()
    observations = summary["observations"]
    assert isinstance(observations, list)
    observations.append(
        _candidate_observation(
            candidate_id="incumbent-reference",
            policy="reference",
            speedup=1.45,
        )
    )
    existing = {
        "schema_version": 2,
        "default_policy": "auto",
        "routes": [
            {"match": {"dtype": "float16"}, "policy": "reference"},
            {
                "match": {"B": 2, "dtype": "float16"},
                "policy": "torch",
            },
        ],
    }
    unrelated_key = {
        "B": 2,
        "dtype": "float16",
    }
    before = resolve_route(validate_route_table(existing), unrelated_key)

    document, _ = build_promoted_route_document(existing, summary)
    after = resolve_route(validate_route_table(document), unrelated_key)

    assert before == "reference"
    assert after == before
    assert document["routes"][1:] == existing["routes"]


def test_a_matching_broad_winner_is_recorded_as_an_exact_verified_route() -> None:
    existing = {
        "schema_version": 2,
        "default_policy": "auto",
        "routes": [{"match": {"B": 1}, "policy": "long-pv"}],
    }

    document, winner = build_promoted_route_document(existing, _formal_summary())

    assert winner["solution_policy"] == "long-pv"
    assert len(document["routes"]) == 2
    assert set(document["routes"][0]["match"]) == ROUTE_FIELDS
    assert document["routes"][0]["policy"] == "long-pv"
    assert document["routes"][1] == existing["routes"][0]


def test_a_verified_auto_winner_is_recorded_explicitly() -> None:
    summary = _formal_summary()
    summary["observations"][2]["correctness_passed"] = False  # type: ignore[index]

    document, winner = build_promoted_route_document(None, summary)

    assert winner["candidate_id"] == "eager-auto"
    assert len(document["routes"]) == 1
    assert set(document["routes"][0]["match"]) == ROUTE_FIELDS
    assert document["routes"][0]["policy"] == "auto"
