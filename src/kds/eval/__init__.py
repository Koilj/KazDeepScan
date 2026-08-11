"""Record-level aggregation, calibration, and evaluation primitives."""

from kds.eval.aggregation import aggregate_global_logit, aggregate_peak_logit
from kds.eval.calibration import TemperatureScaler, brier_score, expected_calibration_error
from kds.eval.metrics import WilsonInterval, classification_confidence_intervals, wilson_interval
from kds.eval.stratified import evaluate_b0_with_strata, stratified_metrics

__all__ = [
    "TemperatureScaler",
    "WilsonInterval",
    "aggregate_global_logit",
    "aggregate_peak_logit",
    "brier_score",
    "classification_confidence_intervals",
    "expected_calibration_error",
    "evaluate_b0_with_strata",
    "stratified_metrics",
    "wilson_interval",
]
