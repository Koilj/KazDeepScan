from __future__ import annotations

import torch

from kds.eval import (
    TemperatureScaler,
    aggregate_global_logit,
    aggregate_peak_logit,
    brier_score,
    expected_calibration_error,
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
