"""Focused contracts for memory-bounded batch-streamed execution."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from official import torch_transformer_benchmark as official
from runner.candidates import CANDIDATE_SPECS, CandidateSpec
from runner.contracts import (
    ContractError,
    MeasurementProtocol,
    RunVariant,
    TransformerShape,
)
from runner.result_contracts import WorkerRequest, validate_workload_execution
from runner.streamed_execution import (
    _execution_fingerprint,
    _full_workload_samples,
    _validated_execution_path,
    execute_streamed_benchmark,
)
from runner.workload_execution import WorkloadExecutionPlan

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _FakeCudaEvent:
    def __init__(self, *, enable_timing: bool) -> None:
        assert enable_timing is True

    def record(self) -> None:
        pass

    def elapsed_time(self, _other: object) -> float:
        return 1.0


class _PathModel(nn.Module):
    def __init__(
        self,
        candidate_spec: CandidateSpec,
        *,
        drift_after: int | None = None,
    ) -> None:
        super().__init__()
        self.candidate_spec = candidate_spec
        self.forward_count = 0
        self.drift_after = drift_after

    def forward(
        self,
        inputs: torch.Tensor,
        _valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.forward_count += 1
        return inputs + 1

    def describe_execution_path(self) -> dict[str, Any]:
        policy = self.candidate_spec.solution_policy
        if self.drift_after is not None and self.forward_count > self.drift_after:
            policy = "safe"
        attention_backend = (
            "mixed_fp16_cudnn"
            if self.candidate_spec.solution_policy == "mixed-fp16-cudnn"
            else "mixed_fp16_efficient"
        )
        return {
            "requested_policy": self.candidate_spec.solution_policy,
            "selected_policy": policy,
            "attention_backend": attention_backend,
            "runtime_wrapper": "eager",
            "residual_norm_backend": "torch",
            "fallback_reasons": [] if policy != "safe" else ["forced_drift"],
            "observed_execution": {
                "attention_backends": [attention_backend],
                "residual_norm_backends": ["torch"],
                "expected_layers": 1,
                "complete": True,
            },
        }


class _PlanOnlyPathModel(_PathModel):
    """Mimic a selected policy after reconfiguration clears observations."""

    def describe_execution_path(self) -> dict[str, Any]:
        path = super().describe_execution_path()
        path.pop("observed_execution", None)
        return path


def _micro_config() -> official.TransformerConfig:
    return official.TransformerConfig(
        batch_size=1,
        seq_len=2,
        d_model=4,
        num_heads=1,
        ffn_dim=4,
        num_layers=1,
        causal=True,
    )


def test_full_streamed_sample_executes_all_32_microbatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_spec = CANDIDATE_SPECS["mixed-fp16-efficient"]
    model = _PlanOnlyPathModel(candidate_spec)
    path = model.describe_execution_path()
    monkeypatch.setattr(torch.cuda, "Event", _FakeCudaEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args, **_kwargs: None)

    samples, completed = _full_workload_samples(
        model,
        config=_micro_config(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        seed=17,
        input_scale=1.0,
        microbatch_count=32,
        candidate_spec=candidate_spec,
        expected_fingerprint=_execution_fingerprint(path),
        repeats=1,
        rounds=1,
    )

    assert samples == [32.0]
    assert completed == 32
    assert model.forward_count == 32


def test_streamed_sample_rejects_policy_drift_between_microbatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_spec = CANDIDATE_SPECS["mixed-fp16-efficient"]
    model = _PathModel(candidate_spec, drift_after=2)
    path = model.describe_execution_path()
    monkeypatch.setattr(torch.cuda, "Event", _FakeCudaEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args, **_kwargs: None)

    with pytest.raises(ContractError, match="registered execution plan"):
        _full_workload_samples(
            model,
            config=_micro_config(),
            device=torch.device("cpu"),
            dtype=torch.float32,
            seed=17,
            input_scale=1.0,
            microbatch_count=4,
            candidate_spec=candidate_spec,
            expected_fingerprint=_execution_fingerprint(path),
            repeats=1,
            rounds=1,
        )


def test_fallback_cannot_count_as_a_successful_streamed_candidate() -> None:
    candidate_spec = CANDIDATE_SPECS["mixed-fp16-cudnn"]
    model = _PathModel(candidate_spec, drift_after=-1)

    with pytest.raises(ContractError, match="registered execution evidence"):
        _validated_execution_path(model, candidate_spec)


def test_screening_still_requires_observed_backend_evidence() -> None:
    candidate_spec = CANDIDATE_SPECS["mixed-fp16-efficient"]
    model = _PlanOnlyPathModel(candidate_spec)

    with pytest.raises(ContractError, match="registered execution evidence"):
        _validated_execution_path(model, candidate_spec)


class _ReferenceModel(nn.Module):
    def forward(
        self,
        inputs: torch.Tensor,
        _valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return inputs + 1


class _TargetModel(_ReferenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.policy = "safe"
        self.observation = False
        self.forward_batches: list[tuple[str, int]] = []

    def forward(
        self,
        inputs: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.forward_batches.append((self.policy, inputs.shape[0]))
        return super().forward(inputs, valid_mask)

    def configure_runtime_policy(self, *, policy: str) -> None:
        self.policy = policy

    def set_execution_observation(self, enabled: bool) -> None:
        self.observation = enabled

    def describe_execution_path(self) -> dict[str, Any]:
        attention_backend = {
            "mixed-fp16-efficient": "mixed_fp16_efficient",
            "mixed-fp16-cudnn": "mixed_fp16_cudnn",
        }[self.policy]
        return {
            "requested_policy": self.policy,
            "selected_policy": self.policy,
            "attention_backend": attention_backend,
            "runtime_wrapper": "eager",
            "residual_norm_backend": "torch",
            "fallback_reasons": [],
            "observed_execution": {
                "attention_backends": [attention_backend],
                "residual_norm_backends": ["torch"],
                "expected_layers": 1,
                "complete": True,
            },
        }


def test_streamed_result_is_target_only_and_measures_peak_after_stream_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runner.streamed_execution as streamed

    shape = TransformerShape(
        case_id="shape14_fixture",
        batch_size=32,
        seq_len=2,
        d_model=4,
        num_heads=1,
        ffn_dim=4,
        num_layers=1,
        causal=True,
    )
    variant = RunVariant()
    protocol = MeasurementProtocol(
        preset="smoke",
        seed=17,
        accuracy_trials=1,
        warmup=1,
        repeats=1,
        rounds=1,
    )
    request = WorkerRequest(
        run_kind="benchmark",
        project_root=PROJECT_ROOT,
        shape=shape,
        variant=variant,
        protocol=protocol,
        device="cuda:0",
        target="solution",
        comparison_mode="target_only",
        solution_policy="screen",
    )
    plan = WorkloadExecutionPlan(
        execution_mode="batch_streamed",
        reference_kind="internal_query_block",
        estimated_dense_attention_bytes=10**12,
        resident_attention_limit_bytes=1,
        validation_microbatch_size=1,
        timing_microbatch_candidates=(1, 2, 4, 8),
        formal_eligible=False,
    )
    efficient = CANDIDATE_SPECS["mixed-fp16-efficient"]
    cudnn = CANDIDATE_SPECS["mixed-fp16-cudnn"]
    constructed_batch_sizes: list[int] = []
    order: list[str] = []
    target_model = _TargetModel()
    built_models = iter((_ReferenceModel(), target_model))

    def fake_baseline(config: official.TransformerConfig) -> nn.Module:
        constructed_batch_sizes.append(config.batch_size)
        return nn.Identity()

    def fake_build(
        _solution_module: Any,
        config: official.TransformerConfig,
        _policy: str,
    ) -> nn.Module:
        constructed_batch_sizes.append(config.batch_size)
        return next(built_models)

    def fake_random_case(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        config = kwargs["config"]
        constructed_batch_sizes.append(config.batch_size)
        return (
            torch.zeros(config.batch_size, 2, 4),
            torch.ones(config.batch_size, 2, dtype=torch.bool),
        )

    def fake_candidate_timing(
        _model: nn.Module,
        _inputs: torch.Tensor,
        _mask: torch.Tensor,
        *,
        candidate_spec: CandidateSpec,
        **_kwargs: Any,
    ) -> list[float]:
        batch_size = _inputs.shape[0]
        if batch_size > 4:
            raise torch.cuda.OutOfMemoryError("fixture schedule does not fit")
        if candidate_spec.solution_policy == "mixed-fp16-efficient":
            return [{1: 2.0, 2: 3.0, 4: 5.0}[batch_size]]
        return [{1: 1.5, 2: 2.0, 4: 3.0}[batch_size]]

    def fake_full_samples(*_args: Any, **kwargs: Any) -> tuple[list[float], int]:
        assert kwargs["microbatch_count"] == 8
        assert kwargs["config"].batch_size == 4
        return [12.0], 8

    def fake_end_to_end(*_args: Any, **kwargs: Any) -> tuple[float, int]:
        assert kwargs["microbatch_count"] == 8
        assert kwargs["config"].batch_size == 4
        order.append("end_to_end")
        return 20.0, 8

    monkeypatch.setattr(
        official, "resolve_device", lambda _value: torch.device("cuda:0")
    )
    monkeypatch.setattr(official, "BaselineTransformer", fake_baseline)
    monkeypatch.setattr(official, "generate_random_case", fake_random_case)
    monkeypatch.setattr(streamed, "collect_environment", lambda _device: {})
    monkeypatch.setattr(streamed, "configure_runtime", lambda *_args: None)
    monkeypatch.setattr(
        streamed,
        "load_solution_source",
        lambda _root: (SimpleNamespace(), "solution-hash"),
    )
    monkeypatch.setattr(
        streamed,
        "config_for_shape",
        lambda _shape: official.TransformerConfig(
            batch_size=32,
            seq_len=2,
            d_model=4,
            num_heads=1,
            ffn_dim=4,
            num_layers=1,
            causal=True,
        ),
    )
    monkeypatch.setattr(streamed, "build_solution", fake_build)
    monkeypatch.setattr(streamed, "_copy_weights", lambda *_args: None)
    monkeypatch.setattr(
        streamed,
        "_candidate_specs",
        lambda *_args: (efficient, cudnn),
    )
    monkeypatch.setattr(streamed, "_cuda_forward_ms", fake_candidate_timing)
    monkeypatch.setattr(streamed, "_streamed_memory_budget", lambda _device: None)
    monkeypatch.setattr(
        streamed,
        "_run_logical_batch",
        lambda *_args, **kwargs: (1.0, kwargs["microbatch_count"]),
    )
    monkeypatch.setattr(streamed, "_full_workload_samples", fake_full_samples)
    monkeypatch.setattr(streamed, "_end_to_end_ms", fake_end_to_end)
    monkeypatch.setattr(
        streamed,
        "solution_implementation_hash",
        lambda _path: "solution-hash",
    )
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda _device: order.append("peak") or 123,
    )
    monkeypatch.setattr(torch.cuda, "Event", _FakeCudaEvent)

    result = execute_streamed_benchmark(request, plan)

    assert result["outcome"] == "success", result["failure"]
    assert constructed_batch_sizes[:4] == [1, 1, 1, 1]
    assert set(constructed_batch_sizes[4:]) == {1, 2, 4, 8}
    # Schedule screening measures B=1 peak once per validated policy before the
    # final deployment peak is sampled after the end-to-end stream buffers.
    assert order[-2:] == ["end_to_end", "peak"]
    assert result["execution_path"]["requested_policy"] == "mixed-fp16-cudnn"
    assert result["execution_path"].get("dispatch_policy") is None
    assert ("mixed-fp16-efficient", 1) in target_model.forward_batches
    assert ("mixed-fp16-cudnn", 1) in target_model.forward_batches
    assert ("mixed-fp16-cudnn", 4) in target_model.forward_batches
    assert result["performance"]["comparison_mode"] == "target_only"
    assert "baseline" not in result["performance"]
    assert "speedup" not in result["performance"]
    assert result["performance"]["peak_device_allocated_bytes"] == 123
    workload = result["workload_execution"]
    assert validate_workload_execution(workload) is None
    assert workload["validation_microbatch_size"] == 1
    assert workload["timing_microbatch_candidates"] == [1, 2, 4, 8]
    assert workload["timing_microbatch_size"] == 4
    assert workload["microbatches_per_sample"] == 8
    assert workload["completed_microbatches"] == 8
    assert workload["end_to_end_microbatches"] == 8
    assert workload["selection"]["policy"] == "mixed-fp16-cudnn"
    assert workload["selection"]["timing_microbatch_size"] == 4
    assert len(workload["selection"]["evidence_sha256"]) == 64
    cudnn_screen = next(
        item
        for item in workload["candidate_screening"]
        if item["policy"] == "mixed-fp16-cudnn"
    )
    assert [item["passed"] for item in cudnn_screen["timing_schedules"]] == [
        True,
        True,
        True,
        False,
    ]
    assert cudnn_screen["timing_schedules"][-1]["failure_kind"] == "oom"
