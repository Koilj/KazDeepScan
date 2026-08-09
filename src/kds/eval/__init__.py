"""Record-level aggregation, calibration, and evaluation primitives."""

from kds.eval.aggregation import aggregate_global_logit, aggregate_peak_logit
from kds.eval.calibration import TemperatureScaler, brier_score, expected_calibration_error

__all__ = [
    "TemperatureScaler",
    "aggregate_global_logit",
    "aggregate_peak_logit",
    "brier_score",
    "expected_calibration_error",
]
