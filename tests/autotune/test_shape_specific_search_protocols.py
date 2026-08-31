from autotune.evaluation import (
    RESIDENT_PROTOCOLS,
    STREAMED_PROTOCOLS,
    EvaluationScope,
    Fidelity,
    FidelityProtocol,
)
from autotune.search_sweep import _protocol
from benchmarking.protocols import MeasurementProtocol


def _counts(
    protocol: FidelityProtocol | MeasurementProtocol,
) -> tuple[int, int, int, int]:
    return (
        protocol.accuracy_trials,
        protocol.warmup,
        protocol.repeats,
        protocol.rounds,
    )


def test_shape_06_reduces_only_correctness_and_warmup() -> None:
    expected_setup = {
        Fidelity.SCREEN: (1, 1),
        Fidelity.ENHANCED: (1, 2),
        Fidelity.FORMAL: (1, 2),
    }

    for fidelity, (accuracy_trials, warmup) in expected_setup.items():
        protocol = _protocol(EvaluationScope.RESIDENT, fidelity, "official_06")
        default = RESIDENT_PROTOCOLS[fidelity]

        assert (protocol.accuracy_trials, protocol.warmup) == (
            accuracy_trials,
            warmup,
        )
        assert (protocol.repeats, protocol.rounds) == (
            default.repeats,
            default.rounds,
        )


def test_other_shapes_keep_the_default_protocols() -> None:
    for fidelity, default in RESIDENT_PROTOCOLS.items():
        assert _counts(
            _protocol(EvaluationScope.RESIDENT, fidelity, "official_05")
        ) == _counts(default)

    for fidelity, default in STREAMED_PROTOCOLS.items():
        protocol = _protocol(EvaluationScope.STREAMED, fidelity, "official_14")
        assert _counts(protocol) == _counts(default)
        assert protocol.full_logical_batch is default.full_logical_batch
