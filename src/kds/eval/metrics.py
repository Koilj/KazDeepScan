"""Uncertainty estimates for count-based binary-classification metrics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist


@dataclass(frozen=True, slots=True)
class WilsonInterval:
    confidence: float
    lower: float
    upper: float


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> WilsonInterval:
    """Return a two-sided Wilson score interval for a binomial proportion.

    Wilson intervals remain well-defined at 0% and 100% recall, unlike the simple Wald interval.
    They quantify finite evaluation-set uncertainty only; they do not claim source robustness.
    """

    if total <= 0:
        raise ValueError("total must be positive.")
    if successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one.")

    proportion = successes / total
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    variance = proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2)
    half_width = z / denominator * variance**0.5
    return WilsonInterval(
        confidence=confidence,
        lower=max(0.0, center - half_width),
        upper=min(1.0, center + half_width),
    )


def classification_confidence_intervals(
    *,
    correct: int,
    examples: int,
    bonafide_correct: int,
    bonafide_examples: int,
    spoof_correct: int,
    spoof_examples: int,
    confidence: float = 0.95,
) -> dict[str, WilsonInterval]:
    """Return intervals only for class metrics with a non-empty denominator."""

    intervals = {"accuracy": wilson_interval(correct, examples, confidence)}
    if bonafide_examples:
        intervals["bonafide_accuracy"] = wilson_interval(
            bonafide_correct, bonafide_examples, confidence
        )
    if spoof_examples:
        intervals["spoof_accuracy"] = wilson_interval(spoof_correct, spoof_examples, confidence)
    return intervals
