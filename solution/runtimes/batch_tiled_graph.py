"""Cache-blocked batch execution through one fixed-shape CUDA Graph."""

from __future__ import annotations

from collections.abc import Callable

import torch


class BatchTiledGraphReplay:
    """Run independent batch tiles through one captured Transformer graph.

    The caller owns the mathematical justification for batch independence.
    This wrapper only accepts an unmasked contiguous CUDA tensor and writes
    every tile directly into one preallocated full-batch output.
    """

    def __init__(self, tile_size: int) -> None:
        if isinstance(tile_size, bool) or not isinstance(tile_size, int):
            raise TypeError("tile_size must be an integer")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        self._tile_size = tile_size
        self._signature: tuple[object, ...] | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_input: torch.Tensor | None = None
        self._static_output: torch.Tensor | None = None

    @property
    def tile_size(self) -> int:
        """Return the captured batch tile size."""

        return self._tile_size

    @staticmethod
    def _input_signature(value: torch.Tensor) -> tuple[object, ...]:
        return (
            value.device,
            value.dtype,
            tuple(value.shape),
            tuple(value.stride()),
        )

    def _capture(
        self,
        function: Callable[[torch.Tensor, None], torch.Tensor],
        value: torch.Tensor,
    ) -> None:
        static_input = torch.zeros(
            (self._tile_size, *value.shape[1:]),
            device=value.device,
            dtype=value.dtype,
        )
        first_count = min(value.shape[0], self._tile_size)
        static_input[:first_count].copy_(value[:first_count])

        current_stream = torch.cuda.current_stream(value.device)
        capture_stream = torch.cuda.Stream(device=value.device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream), torch.inference_mode():
            for _ in range(3):
                function(static_input, None)
        current_stream.wait_stream(capture_stream)

        graph = torch.cuda.CUDAGraph()
        with (
            torch.inference_mode(),
            torch.cuda.graph(graph, stream=capture_stream),
        ):
            static_output = function(static_input, None)
        current_stream.wait_stream(capture_stream)

        if not isinstance(static_output, torch.Tensor):
            raise TypeError("batch-tiled graph function must return a Tensor")
        if static_output.shape != static_input.shape:
            raise RuntimeError("batch-tiled graph output shape must match its input")
        if static_output.device != value.device or static_output.dtype != value.dtype:
            raise RuntimeError(
                "batch-tiled graph output must preserve input device and dtype"
            )

        self._signature = self._input_signature(value)
        self._graph = graph
        self._static_input = static_input
        self._static_output = static_output

    def run(
        self,
        function: Callable[[torch.Tensor, None], torch.Tensor],
        value: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Replay every fixed tile and assemble one independent full output."""

        if value.device.type != "cuda":
            raise ValueError("batch-tiled CUDA Graph requires a CUDA input")
        if value.ndim != 3 or not value.is_contiguous():
            raise ValueError("batch-tiled CUDA Graph requires a contiguous 3D input")
        if value.shape[0] <= self._tile_size:
            raise ValueError("batch-tiled CUDA Graph requires more than one tile")
        if valid_mask is not None:
            raise ValueError("batch-tiled CUDA Graph does not accept a token mask")

        signature = self._input_signature(value)
        if self._graph is not None and signature != self._signature:
            raise RuntimeError(
                "batch-tiled CUDA Graph supports one full input signature"
            )
        if self._graph is None:
            self._capture(function, value)

        assert self._graph is not None
        assert self._static_input is not None
        assert self._static_output is not None

        output = torch.empty_like(value)
        for start in range(0, value.shape[0], self._tile_size):
            stop = min(start + self._tile_size, value.shape[0])
            count = stop - start
            if count == self._tile_size:
                self._static_input.copy_(value[start:stop])
            else:
                self._static_input.zero_()
                self._static_input[:count].copy_(value[start:stop])
            self._graph.replay()
            output[start:stop].copy_(self._static_output[:count])
        return output


__all__ = ["BatchTiledGraphReplay"]
