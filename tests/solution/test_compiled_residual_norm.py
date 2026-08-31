from __future__ import annotations

import pytest
import torch
from torch import nn

from solution.operators import can_use_residual_layer_norm, residual_layer_norm
from solution.operators.norm import compiled_residual as residual_norm_module
from solution.runtimes.cuda_graph import CudaGraphReplay


def test_cpu_inputs_are_rejected_by_the_compiled_policy() -> None:
    generator = torch.Generator().manual_seed(23)
    value = torch.randn(3, 5, 8, generator=generator)
    update = torch.randn(3, 5, 8, generator=generator)
    layer_norm = nn.LayerNorm(8)
    assert not can_use_residual_layer_norm(value, update, layer_norm)
    with pytest.raises(RuntimeError, match="ineligible"):
        residual_layer_norm(value, update, layer_norm)


def test_compiler_failure_is_exposed_instead_of_silently_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = torch.randn(2, 4)
    update = torch.randn_like(value)
    layer_norm = nn.LayerNorm(4)

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("synthetic compiler failure")

    monkeypatch.setattr(
        residual_norm_module,
        "can_use_residual_layer_norm",
        lambda *_args: True,
    )
    monkeypatch.setattr(residual_norm_module, "_get_compiled_kernel", lambda: fail)
    monkeypatch.setattr(residual_norm_module, "_failed_signatures", set())

    with pytest.raises(RuntimeError, match="execution failed"):
        residual_norm_module.residual_layer_norm(value, update, layer_norm)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compiled_cuda_path_reports_the_backend_and_preserves_results() -> None:
    generator = torch.Generator(device="cuda").manual_seed(29)
    value = torch.randn(512, 128, generator=generator, device="cuda")
    update = torch.randn(512, 128, generator=generator, device="cuda")
    layer_norm = nn.LayerNorm(128).cuda().eval()

    with torch.inference_mode():
        expected_residual = value + update
        expected_normalized = layer_norm(expected_residual)
        residual, normalized, backend = residual_layer_norm(
            value,
            update,
            layer_norm,
        )

    assert backend == "compiled_residual_layer_norm"
    torch.testing.assert_close(residual, expected_residual)
    torch.testing.assert_close(normalized, expected_normalized, rtol=2e-5, atol=2e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_warmed_compiled_path_can_be_captured_by_cuda_graph() -> None:
    generator = torch.Generator(device="cuda").manual_seed(31)
    value = torch.randn(256, 128, generator=generator, device="cuda")
    update = torch.randn(256, 128, generator=generator, device="cuda")
    layer_norm = nn.LayerNorm(128).cuda().eval()

    with torch.inference_mode():
        _, _, backend = residual_layer_norm(value, update, layer_norm)
        assert backend == "compiled_residual_layer_norm"
        replay = CudaGraphReplay()

        def function(
            current: torch.Tensor,
            _mask: torch.Tensor | None,
        ) -> torch.Tensor:
            return residual_layer_norm(current, update, layer_norm)[1]

        replay.run(function, value, None)
        changed = value * 0.75
        actual = replay.run(function, changed, None)
        expected = layer_norm(changed + update)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
