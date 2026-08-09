from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from kds.models import SlsBinaryHead, SlsHeadConfig, XlsrSlsClassifier


class FakeEncoder(nn.Module):
    def forward(self, input_values: Tensor, **_kwargs: object) -> SimpleNamespace:
        batch_size = input_values.shape[0]
        hidden_states = tuple(torch.randn(batch_size, 4, 8) for _ in range(3))
        return SimpleNamespace(hidden_states=hidden_states)


def test_sls_head_outputs_one_logit_per_recording_and_backpropagates() -> None:
    head = SlsBinaryHead(SlsHeadConfig(hidden_size=8, hidden_state_count=3, dropout=0.0))
    states = tuple(torch.randn(2, 5, 8, requires_grad=True) for _ in range(3))
    mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])

    logits = head(states, mask)
    logits.sum().backward()

    assert logits.shape == (2,)
    assert head.layer_mix.layer_logits.grad is not None
    assert all(state.grad is not None for state in states)


def test_xlsr_wrapper_converts_sample_mask_to_feature_mask() -> None:
    model = XlsrSlsClassifier(
        encoder=FakeEncoder(),
        head_config=SlsHeadConfig(hidden_size=8, hidden_state_count=3, dropout=0.0),
        conv_kernel=(3, 3),
        conv_stride=(2, 2),
    )
    input_values = torch.randn(2, 20)
    attention_mask = torch.tensor([[1] * 20, [1] * 12 + [0] * 8])

    logits = model(input_values, attention_mask)

    assert logits.shape == (2,)


def test_sls_head_rejects_hidden_state_count_mismatch() -> None:
    head = SlsBinaryHead(SlsHeadConfig(hidden_size=8, hidden_state_count=3))
    states = tuple(torch.randn(1, 4, 8) for _ in range(2))
    mask = torch.ones((1, 4), dtype=torch.bool)

    with pytest.raises(ValueError, match="Expected 3 hidden states"):
        head(states, mask)
