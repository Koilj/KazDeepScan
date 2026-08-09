from __future__ import annotations

import pytest
import torch

from kds.eval import (
    TemperatureScaler,
    aggregate_global_logit,
    aggregate_peak_logit,
    brier_score,
    classification_confidence_intervals,
    expected_calibration_error,
    wilson_interval,
)


def test_global_logit_aggregation_weights_short_final_window() -> None:
    logits = torch.tensor([0.0, 2.0])
    real_samples = torch.tensor([64_600, 32_300])

    aggregate = aggregate_global_logit(logits, real_samples)

    assert torch.isclose(aggregate, torch.tensor(2.0 / 3.0))


def test_peak_logit_aggregation_uses_top_twenty_percent_with_minimum_two() -> None:
    logits = torch.tensor([-3.0, -1.0, 0.5, 2.0, 4.0])

    aggregate = aggregate_peak_logit(logits)

    assert torch.isclose(aggregate, torch.tensor(3.0))


def test_temperature_scaling_reduces_dev_negative_log_likelihood() -> None:
    logits = torch.tensor([8.0, 8.0, -8.0, -8.0, 8.0, -8.0])
    labels = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    scaler = TemperatureScaler()

    report = scaler.fit(logits, labels)

    assert report.temperature > 0.0
    assert report.nll_after < report.nll_before
    assert scaler(logits).shape == logits.shape


def test_calibration_metrics_accept_valid_probabilities() -> None:
    probabilities = torch.tensor([0.1, 0.9, 0.2, 0.8])
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

    assert brier_score(probabilities, labels) >= 0.0
    assert expected_calibration_error(probabilities, labels) >= 0.0


def test_wilson_interval_is_stable_at_perfect_recall() -> None:
    interval = wilson_interval(10, 10)

    assert interval.confidence == 0.95
    assert 0.69 < interval.lower < 0.73
    assert interval.upper == 1.0


def test_wilson_interval_rejects_invalid_counts_and_confidence() -> None:
    with pytest.raises(ValueError, match="total"):
        wilson_interval(0, 0)
    with pytest.raises(ValueError, match="successes"):
        wilson_interval(2, 1)
    with pytest.raises(ValueError, match="confidence"):
        wilson_interval(1, 2, confidence=1.0)


def test_classification_intervals_omit_missing_class() -> None:
    intervals = classification_confidence_intervals(
        correct=8,
        examples=10,
        bonafide_correct=8,
        bonafide_examples=10,
        spoof_correct=0,
        spoof_examples=0,
    )

    assert set(intervals) == {"accuracy", "bonafide_accuracy"}
