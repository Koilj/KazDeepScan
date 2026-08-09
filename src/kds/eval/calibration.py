from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    temperature: float
    nll_before: float
    nll_after: float
    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float


class TemperatureScaler(nn.Module):
    """One-parameter record-level temperature scaling fitted only on a dev set."""

    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(()))

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp().clamp(min=1e-3, max=1e3)

    def forward(self, record_logits: Tensor) -> Tensor:
        if record_logits.ndim != 1:
            raise ValueError("record_logits must have shape [records].")
        return record_logits / self.temperature

    def fit(self, record_logits: Tensor, labels: Tensor, max_iter: int = 50) -> CalibrationReport:
        _validate_calibration_inputs(record_logits, labels)
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        logits = record_logits.detach()
        targets = labels.detach().to(dtype=logits.dtype)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.25, max_iter=max_iter)
        before_probabilities = torch.sigmoid(logits)
        nll_before = float(criterion(logits, targets))

        def closure() -> Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(self(logits), targets)
            loss.backward()
            return cast(Tensor, loss)

        optimizer.step(closure)  # type: ignore[no-untyped-call]
        after_logits = self(logits).detach()
        after_probabilities = torch.sigmoid(after_logits)
        return CalibrationReport(
            temperature=float(self.temperature.detach()),
            nll_before=nll_before,
            nll_after=float(criterion(after_logits, targets).detach()),
            brier_before=brier_score(before_probabilities, targets),
            brier_after=brier_score(after_probabilities, targets),
            ece_before=expected_calibration_error(before_probabilities, targets),
            ece_after=expected_calibration_error(after_probabilities, targets),
        )


def brier_score(probabilities: Tensor, labels: Tensor) -> float:
    _validate_calibration_inputs(probabilities, labels, values_are_probabilities=True)
    return float((probabilities - labels.to(dtype=probabilities.dtype)).square().mean().detach())


def expected_calibration_error(probabilities: Tensor, labels: Tensor, bins: int = 15) -> float:
    _validate_calibration_inputs(probabilities, labels, values_are_probabilities=True)
    if bins <= 0:
        raise ValueError("bins must be positive.")
    targets = labels.to(dtype=probabilities.dtype)
    edges = torch.linspace(0.0, 1.0, bins + 1, device=probabilities.device)
    ece = torch.zeros((), device=probabilities.device, dtype=probabilities.dtype)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        membership = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        count = membership.sum()
        if count == 0:
            continue
        confidence = probabilities[membership].mean()
        accuracy = targets[membership].mean()
        ece += count / probabilities.numel() * (confidence - accuracy).abs()
    return float(ece.detach())


def _validate_calibration_inputs(
    values: Tensor, labels: Tensor, values_are_probabilities: bool = False
) -> None:
    same_shape = values.ndim == 1 and labels.ndim == 1 and values.shape == labels.shape
    if not same_shape or values.numel() == 0:
        raise ValueError(
            "Values and labels must be non-empty tensors with matching [records] shape."
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Values must be finite.")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("labels must be binary values 0 or 1.")
    if values_are_probabilities and not bool(((values >= 0) & (values <= 1)).all()):
        raise ValueError("Probabilities must be in [0, 1].")
