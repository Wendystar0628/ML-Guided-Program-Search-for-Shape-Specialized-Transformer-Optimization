"""Sequential paired-block rule for automatic deployment promotion."""

from __future__ import annotations

import math
from enum import StrEnum

PROMOTION_MAX_BLOCKS = 13
PROMOTION_BASE_RATIO = 1.02
PROMOTION_BASE_WINS = 11

# (blocks observed, required ratio, required wins)
PROMOTION_STAGES = (
    (6, 1.10, 6),
    (9, 1.05, 8),
    (13, PROMOTION_BASE_RATIO, PROMOTION_BASE_WINS),
)


class PromotionDecision(StrEnum):
    """Terminal or continuing state of one sequential Formal comparison."""

    CONTINUE = "continue"
    PROMOTE = "promote"
    REJECT = "reject"


def promotion_decision(paired_ratios: tuple[float, ...]) -> PromotionDecision:
    """Evaluate the pre-specified group-sequential paired sign test.

    For independent paired blocks under H0: P(block ratio >= 1.02) <= 0.5,
    the three promotion looks have one-sided false-promotion bounds 1/64,
    10/512, and 92/8192. Their union is below 0.05, so optional stopping at
    these looks preserves the error bound.
    """

    ratios = tuple(float(value) for value in paired_ratios)
    if len(ratios) > PROMOTION_MAX_BLOCKS:
        raise ValueError(f"at most {PROMOTION_MAX_BLOCKS} paired blocks are allowed")
    if any(not math.isfinite(value) or value <= 0.0 for value in ratios):
        raise ValueError("paired ratios must be finite and positive")

    blocks = len(ratios)
    for stage_blocks, stage_ratio, required_wins in PROMOTION_STAGES:
        if blocks == stage_blocks:
            wins = sum(ratio >= stage_ratio for ratio in ratios)
            if wins >= required_wins:
                return PromotionDecision.PROMOTE

    base_wins = sum(ratio >= PROMOTION_BASE_RATIO for ratio in ratios)
    remaining = PROMOTION_MAX_BLOCKS - blocks
    if base_wins + remaining < PROMOTION_BASE_WINS:
        return PromotionDecision.REJECT
    if blocks == PROMOTION_MAX_BLOCKS:
        return PromotionDecision.REJECT
    return PromotionDecision.CONTINUE


def promotion_should_stop(paired_ratios: tuple[float, ...]) -> bool:
    """Return whether a growing Formal comparison has reached a decision."""

    return promotion_decision(paired_ratios) is not PromotionDecision.CONTINUE


__all__ = [
    "PROMOTION_BASE_RATIO",
    "PROMOTION_BASE_WINS",
    "PROMOTION_MAX_BLOCKS",
    "PROMOTION_STAGES",
    "PromotionDecision",
    "promotion_decision",
    "promotion_should_stop",
]
