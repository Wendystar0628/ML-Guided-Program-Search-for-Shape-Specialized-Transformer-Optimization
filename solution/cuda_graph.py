"""Narrow CUDA Graph replay state for a fixed eager Transformer route."""

from __future__ import annotations

from collections.abc import Callable

import torch


class CudaGraphReplay:
    """Capture exactly one static signature and replay it safely."""

    def __init__(self) -> None:
        self._signature: tuple[object, ...] | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_input: torch.Tensor | None = None
        self._static_mask: torch.Tensor | None = None
        self._static_output: torch.Tensor | None = None

    @staticmethod
    def _input_signature(
        value: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> tuple[object, ...]:
        mask_signature = None
        if valid_mask is not None:
            mask_signature = (
                valid_mask.device,
                valid_mask.dtype,
                tuple(valid_mask.shape),
                tuple(valid_mask.stride()),
            )
        return (
            value.device,
            value.dtype,
            tuple(value.shape),
            tuple(value.stride()),
            mask_signature,
        )

    def _capture(
        self,
        function: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
        value: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> None:
        signature = self._input_signature(value, valid_mask)
        static_input = value.detach().clone()
        static_mask = None if valid_mask is None else valid_mask.detach().clone()

        current_stream = torch.cuda.current_stream(value.device)
        capture_stream = torch.cuda.Stream(device=value.device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream), torch.inference_mode():
            for _ in range(3):
                function(static_input, static_mask)
        current_stream.wait_stream(capture_stream)

        graph = torch.cuda.CUDAGraph()
        with (
            torch.inference_mode(),
            torch.cuda.graph(
                graph,
                stream=capture_stream,
            ),
        ):
            static_output = function(static_input, static_mask)
        current_stream.wait_stream(capture_stream)

        # Publish the cache only after every warmup and capture operation has
        # succeeded. A failed first capture leaves the instance empty.
        self._signature = signature
        self._graph = graph
        self._static_input = static_input
        self._static_mask = static_mask
        self._static_output = static_output

    def run(
        self,
        function: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
        value: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Copy current inputs, replay the graph, and return independent output."""

        signature = self._input_signature(value, valid_mask)
        if self._graph is not None and signature != self._signature:
            raise RuntimeError(
                "CUDA Graph replay supports one static input signature per model; "
                "create or reconfigure the model for a different signature"
            )
        if self._graph is None:
            self._capture(function, value, valid_mask)

        assert self._static_input is not None
        assert self._static_output is not None
        assert self._graph is not None
        self._static_input.copy_(value)
        if valid_mask is not None:
            assert self._static_mask is not None
            self._static_mask.copy_(valid_mask)
        self._graph.replay()
        return self._static_output.clone()


__all__ = ["CudaGraphReplay"]
