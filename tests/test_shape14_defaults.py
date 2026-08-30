from __future__ import annotations

import benchmarking.configuration as configuration
from solution.config import AttentionBackend, RuntimeBackend
from solution.shape14.defaults import conservative_streamed_config
from solution.shape14 import triton_streaming_dh64


def test_conservative_shape14_config_is_stable_and_streamed() -> None:
    config = conservative_streamed_config()

    assert config.config_id == (
        "cfg-3d94f979e8424ea78bfd0656a0d32182cb9f0e706f85b805e6d01571496b75ec"
    )
    assert config.program.attention is AttentionBackend.TRITON_STREAMING_DH64
    assert config.schedule.runtime is RuntimeBackend.STREAMED
    assert config.schedule.microbatch_size == 1
    assert config.schedule.attention_launch is not None
    assert config.schedule.attention_launch.to_dict() == {
        "block_m": 32,
        "block_n": 64,
        "num_warps": 4,
        "num_stages": 2,
    }


def test_streamed_fallback_prefers_the_conservative_kernel(monkeypatch) -> None:
    monkeypatch.setattr(configuration.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        configuration.torch.cuda,
        "get_device_capability",
        lambda device: (8, 9),
    )
    monkeypatch.setattr(
        triton_streaming_dh64,
        "triton_streaming_dh64_causal_attention_available",
        lambda: True,
    )

    assert (
        configuration._streamed_fallback_config("cuda:0")
        == conservative_streamed_config()
    )
