from __future__ import annotations

import math

from torch import Tensor


def aggregate_global_logit(window_logits: Tensor, real_samples: Tensor) -> Tensor:
    """Duration-weighted raw-logit aggregation for a fully synthetic recording."""

    _validate_windows(window_logits, real_samples)
    weights = real_samples.to(dtype=window_logits.dtype)
    return (window_logits * weights).sum() / weights.sum()


def aggregate_peak_logit(
    window_logits: Tensor,
    top_fraction: float = 0.2,
    minimum_windows: int = 2,
) -> Tensor:
    """Top-k raw-logit aggregation for possible partial synthetic insertions."""

    if window_logits.ndim != 1 or window_logits.numel() == 0:
        raise ValueError("window_logits must be a non-empty one-dimensional tensor.")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1].")
    if minimum_windows <= 0:
        raise ValueError("minimum_windows must be positive.")
    fraction_count = math.ceil(window_logits.numel() * top_fraction)
    k = min(window_logits.numel(), max(minimum_windows, fraction_count))
    return window_logits.topk(k).values.mean()


def _validate_windows(window_logits: Tensor, real_samples: Tensor) -> None:
    if window_logits.ndim != 1 or real_samples.ndim != 1:
        raise ValueError("window_logits and real_samples must be one-dimensional tensors.")
    if window_logits.shape != real_samples.shape or window_logits.numel() == 0:
        raise ValueError("window_logits and real_samples must have the same non-zero shape.")
    if not bool((real_samples > 0).all()):
        raise ValueError("real_samples must contain only positive values.")
