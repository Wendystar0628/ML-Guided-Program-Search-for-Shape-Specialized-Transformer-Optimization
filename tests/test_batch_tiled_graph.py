from __future__ import annotations

import pytest
import torch

from solution.runtimes.batch_tiled_graph import BatchTiledGraphReplay


def test_batch_tiled_graph_validates_tile_size() -> None:
    with pytest.raises(TypeError, match="integer"):
        BatchTiledGraphReplay(True)
    with pytest.raises(ValueError, match="positive"):
        BatchTiledGraphReplay(0)


def test_batch_tiled_graph_rejects_unsupported_inputs_before_capture() -> None:
    replay = BatchTiledGraphReplay(2)
    value = torch.zeros(4, 3, 8)

    with pytest.raises(ValueError, match="CUDA input"):
        replay.run(lambda tensor, _mask: tensor, value, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_batch_tiled_graph_replays_full_and_zero_padded_tail() -> None:
    replay = BatchTiledGraphReplay(4)
    value = torch.arange(6 * 3 * 8, device="cuda", dtype=torch.float32).reshape(6, 3, 8)

    with torch.inference_mode():
        output = replay.run(lambda tensor, _mask: tensor + 1, value, None)

    assert replay.tile_size == 4
    assert torch.equal(output, value + 1)
