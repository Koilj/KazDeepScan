from __future__ import annotations

import pytest
import torch

from kds.models import B0Config, B0LogMelCnn


def test_b0_returns_one_logit_per_waveform() -> None:
    model = B0LogMelCnn(B0Config(n_mels=32)).eval()
    waveform = torch.randn((2, 6_400), dtype=torch.float32) * 0.01

    with torch.inference_mode():
        logits = model(waveform)

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_b0_rejects_invalid_waveform_shape() -> None:
    model = B0LogMelCnn()

    with pytest.raises(ValueError, match=r"\[batch, samples\]"):
        model(torch.randn((1, 1, 6_400)))
