from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from solution.kernels.mixed_attention import (
    MIXED_FP16_CUDNN_BACKEND,
    MIXED_FP16_EFFICIENT_BACKEND,
    can_use_mixed_fp16_cudnn_attention,
    can_use_mixed_fp16_efficient_attention,
    mixed_fp16_cudnn_attention,
    mixed_fp16_efficient_attention,
)


def test_mixed_attention_rejects_cpu_requests() -> None:
    query = torch.randn(1, 1, 1024, 32)

    with torch.inference_mode():
        assert not can_use_mixed_fp16_efficient_attention(query, query, query)
        with pytest.raises(ValueError, match="incompatible"):
            mixed_fp16_efficient_attention(query, query, query)
        assert not can_use_mixed_fp16_cudnn_attention(query, query, query)
        with pytest.raises(ValueError, match="incompatible"):
            mixed_fp16_cudnn_attention(query, query, query)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_mixed_attention_executes_the_forced_backend_and_meets_comparator() -> None:
    generator = torch.Generator(device="cuda").manual_seed(23)
    query = torch.randn(2, 4, 1024, 32, generator=generator, device="cuda")
    key = torch.randn(2, 4, 1024, 32, generator=generator, device="cuda")
    value = torch.randn(2, 4, 1024, 32, generator=generator, device="cuda")
    scale = query.shape[-1] ** -0.5

    with torch.inference_mode():
        assert can_use_mixed_fp16_efficient_attention(query, key, value)
        actual, backend = mixed_fp16_efficient_attention(
            query,
            key,
            value,
            scale=scale,
        )
        reference = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
        )

    absolute_error = (actual - reference).abs()
    comparator_passed = (absolute_error <= 0.002) | (
        absolute_error <= 0.02 * reference.abs()
    )
    assert bool(comparator_passed.all())
    assert backend == MIXED_FP16_EFFICIENT_BACKEND
    assert actual.shape == query.shape
    assert actual.dtype == torch.float32
    assert actual.device == query.device
    assert not can_use_mixed_fp16_efficient_attention(query, key, value)

    with torch.inference_mode():
        assert not can_use_mixed_fp16_efficient_attention(
            query,
            key,
            value,
            torch.ones(2, 1024, dtype=torch.bool, device="cuda"),
        )
        assert not can_use_mixed_fp16_efficient_attention(
            query[..., :512, :],
            key[..., :512, :],
            value[..., :512, :],
        )

        narrow = torch.randn(
            64,
            4,
            128,
            8,
            generator=generator,
            device="cuda",
        )
        assert can_use_mixed_fp16_efficient_attention(narrow, narrow, narrow)
        wider_head = torch.randn(1, 1, 1024, 64, device="cuda")
        assert can_use_mixed_fp16_efficient_attention(
            wider_head,
            wider_head,
            wider_head,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_mixed_attention_executes_head_dim_64_and_meets_comparator() -> None:
    generator = torch.Generator(device="cuda").manual_seed(29)
    query = torch.randn(1, 1, 1024, 64, generator=generator, device="cuda")
    key = torch.randn(1, 1, 1024, 64, generator=generator, device="cuda")
    value = torch.randn(1, 1, 1024, 64, generator=generator, device="cuda")
    scale = query.shape[-1] ** -0.5

    with torch.inference_mode():
        actual, backend = mixed_fp16_efficient_attention(
            query,
            key,
            value,
            scale=scale,
        )
        reference = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
        )

    absolute_error = (actual - reference).abs()
    comparator_passed = (absolute_error <= 0.002) | (
        absolute_error <= 0.02 * reference.abs()
    )
    assert bool(comparator_passed.all())
    assert backend == MIXED_FP16_EFFICIENT_BACKEND


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_mixed_cudnn_attention_is_forced_observed_and_comparator_safe() -> None:
    if not torch.backends.cudnn.is_available():
        pytest.skip("cuDNN is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(31)
    query = torch.randn(1, 1, 1024, 64, generator=generator, device="cuda")
    key = torch.randn(1, 1, 1024, 64, generator=generator, device="cuda")
    value = torch.randn(1, 1, 1024, 64, generator=generator, device="cuda")
    scale = query.shape[-1] ** -0.5

    with torch.inference_mode():
        assert can_use_mixed_fp16_cudnn_attention(query, key, value)
        actual, backend = mixed_fp16_cudnn_attention(
            query,
            key,
            value,
            scale=scale,
        )
        reference = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
        )

    absolute_error = (actual - reference).abs()
    comparator_passed = (absolute_error <= 0.002) | (
        absolute_error <= 0.02 * reference.abs()
    )
    assert bool(comparator_passed.all())
    assert backend == MIXED_FP16_CUDNN_BACKEND
    assert actual.shape == query.shape
    assert actual.dtype == torch.float32

    with torch.inference_mode():
        assert not can_use_mixed_fp16_cudnn_attention(
            query[..., :512, :],
            key[..., :512, :],
            value[..., :512, :],
        )
        assert not can_use_mixed_fp16_cudnn_attention(
            query[..., :32],
            key[..., :32],
            value[..., :32],
        )
        assert not can_use_mixed_fp16_cudnn_attention(
            query,
            key,
            value,
            torch.ones(1, 1024, dtype=torch.bool, device="cuda"),
        )
        assert not can_use_mixed_fp16_cudnn_attention(
            query,
            key,
            value,
            causal=False,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_mixed_cudnn_attention_never_falls_through_on_backend_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.backends.cudnn.is_available():
        pytest.skip("cuDNN is unavailable")
    query = torch.randn(1, 1, 1024, 64, device="cuda")

    def fail_sdpa(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("forced failure")

    monkeypatch.setattr(F, "scaled_dot_product_attention", fail_sdpa)
    with (
        torch.inference_mode(),
        pytest.raises(
            RuntimeError,
            match="forced FP16 cuDNN SDPA is unavailable",
        ),
    ):
        mixed_fp16_cudnn_attention(query, query, query)
