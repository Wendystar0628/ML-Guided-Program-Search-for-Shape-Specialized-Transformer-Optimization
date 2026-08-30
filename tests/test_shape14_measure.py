from __future__ import annotations

import torch

import benchmarking.measure as measure_module
import benchmarking.shape14_measure as shape14_module
from benchmarking.protocols import MeasurementProtocol, RunVariant, TransformerShape
from solution.config import RuntimeBackend, portable_streamed_config


def _shape14() -> TransformerShape:
    return TransformerShape(
        case_id="official_14",
        batch_size=8,
        seq_len=16,
        d_model=32,
        num_heads=4,
        ffn_dim=32,
        num_layers=2,
        causal=True,
    )


def _protocol() -> MeasurementProtocol:
    return MeasurementProtocol(
        accuracy_trials=1,
        warmup=0,
        repeats=1,
        rounds=1,
    )


def test_inner_config_preserves_program_and_removes_streamed_runtime() -> None:
    outer = portable_streamed_config(microbatch_size=4)

    inner = shape14_module._inner_streamed_config(outer)

    assert inner.program == outer.program
    assert inner.schedule.runtime is RuntimeBackend.EAGER
    assert inner.schedule.microbatch_size is None
    assert inner.schedule.attention_launch == outer.schedule.attention_launch
    assert inner.schedule.qkv_launch == outer.schedule.qkv_launch


def test_model_compute_estimate_covers_full_causal_logical_batch() -> None:
    shape = _shape14()

    assert shape14_module._estimated_model_flops(shape) == 3_407_872


def test_outer_signature_reports_the_deployed_streamed_config() -> None:
    outer = portable_streamed_config(microbatch_size=4)

    signature = shape14_module._as_outer_streamed_signature(
        {"config_id": "inner", "runtime_backend": "eager", "attention": "sdpa"},
        outer,
    )

    assert signature == {
        "config_id": outer.config_id,
        "runtime_backend": RuntimeBackend.STREAMED.value,
        "attention": "sdpa",
    }


def test_full_logical_batch_uses_distinct_inputs_and_restores_buffer() -> None:
    observed: list[torch.Tensor] = []

    class Model(torch.nn.Module):
        def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            observed.append(value.clone())
            return value * mask[..., None]

    value = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    original = value.clone()
    mask = torch.ones((2, 3), dtype=torch.bool)
    logical = shape14_module._DistinctLogicalBatch(Model(), chunks=3)

    summary = logical(value, mask)

    assert summary.numel() > 0
    assert len(observed) == 3
    assert not torch.equal(observed[0], observed[1])
    assert not torch.equal(observed[1], observed[2])
    torch.testing.assert_close(value, original)
    digest = shape14_module._logical_output_digest(logical)
    assert digest is not None and len(digest) == 64


def test_public_measurement_entry_delegates_shape14_lazily(monkeypatch) -> None:
    expected = object()
    calls: list[tuple[object, ...]] = []

    def measure_shape14(*args: object) -> object:
        calls.append(args)
        return expected

    monkeypatch.setattr(shape14_module, "measure_shape14_config", measure_shape14)
    shape = _shape14()
    config = portable_streamed_config(microbatch_size=4)
    variant = RunVariant()
    protocol = _protocol()

    result = measure_module.measure_config(
        shape,
        config,
        variant,
        protocol,
        "cpu",
    )

    assert result is expected
    assert calls == [(shape, config, variant, protocol, torch.device("cpu"))]


def test_public_paired_entry_delegates_shape14_lazily(monkeypatch) -> None:
    expected = object()
    calls: list[tuple[object, ...]] = []

    def measure_paired_shape14(*args: object) -> object:
        calls.append(args)
        return expected

    monkeypatch.setattr(
        shape14_module,
        "measure_paired_shape14_configs",
        measure_paired_shape14,
    )
    shape = _shape14()
    challenger = portable_streamed_config(microbatch_size=2)
    incumbent = portable_streamed_config(microbatch_size=4)
    variant = RunVariant()
    protocol = _protocol()

    result = measure_module.measure_paired_configs(
        shape,
        challenger,
        incumbent,
        variant,
        protocol,
        "cpu",
    )

    assert result is expected
    assert calls == [
        (
            shape,
            challenger,
            incumbent,
            variant,
            protocol,
            torch.device("cpu"),
            None,
        )
    ]
