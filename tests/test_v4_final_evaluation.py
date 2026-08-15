"""Small deterministic tests for the v4 final-evaluation summary primitives."""

from __future__ import annotations

import pytest
import torch

from kds.eval.v4_final_evaluation import _classification_metrics, _eer


def test_final_metrics_use_fixed_zero_logit_boundary() -> None:
    metrics = _classification_metrics(
        torch.tensor([-2.0, -0.5, 0.5, 2.0]), torch.tensor([0.0, 0.0, 1.0, 1.0])
    )

    assert metrics["decision_boundary"] == "raw_logit_zero"
    accuracy = metrics["accuracy"]
    assert isinstance(accuracy, dict)
    assert accuracy["correct"] == 4
    assert metrics["balanced_accuracy"] == 1.0


def test_empirical_eer_is_score_only_descriptive_metric() -> None:
    assert _eer(
        torch.tensor([-2.0, -1.0, 1.0, 2.0]), torch.tensor([0.0, 0.0, 1.0, 1.0])
    ) == 0.0
    assert _eer(
        torch.tensor([0.0, 0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0, 1.0])
    ) == pytest.approx(0.5)
