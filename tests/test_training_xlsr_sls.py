from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import Tensor, nn
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader, Dataset

from kds.data.dataset import AudioSample
from kds.models import SlsHeadConfig, XlsrSlsClassifier
from kds.training import (
    collate_audio_samples,
    configure_xlsr_stage_b,
    evaluate_xlsr_sls,
    freeze_xlsr_encoder,
    train_xlsr_sls_head_epoch,
    train_xlsr_sls_stage_b_epoch,
)
from kds.training.b0 import AudioBatch


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1, 8)

    def forward(self, input_values: Tensor, **_kwargs: object) -> SimpleNamespace:
        pooled = input_values[:, :4].unsqueeze(-1)
        first = self.projection(pooled)
        return SimpleNamespace(hidden_states=(first, first + 0.1, first + 0.2))


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(8, 8)

    def forward(self, features: Tensor) -> Tensor:
        return torch.tanh(self.projection(features))


class _TinyStack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_TinyBlock() for _ in range(4))


class _TinyStageBEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1, 8)
        self.encoder = _TinyStack()
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def forward(self, input_values: Tensor, **_kwargs: object) -> SimpleNamespace:
        features = self.projection(input_values[:, :4].unsqueeze(-1))
        hidden_states = [features]
        for layer in self.encoder.layers:
            features = layer(features)
            hidden_states.append(features)
        return SimpleNamespace(hidden_states=tuple(hidden_states))


def _sample(sample_id: str, label: float) -> AudioSample:
    return AudioSample(
        waveform=torch.randn(64, dtype=torch.float32) * 0.01,
        label=torch.tensor(label, dtype=torch.float32),
        sample_id=sample_id,
        parent_group_id=f"parent-{sample_id}",
        language="ru",
    )


def _model() -> XlsrSlsClassifier:
    return XlsrSlsClassifier(
        encoder=_TinyEncoder(),
        head_config=SlsHeadConfig(
            hidden_size=8,
            hidden_state_count=3,
            attention_size=4,
            classifier_size=4,
            dropout=0.0,
        ),
        conv_kernel=(3,),
        conv_stride=(2,),
    )


def _stage_b_model() -> XlsrSlsClassifier:
    return XlsrSlsClassifier(
        encoder=_TinyStageBEncoder(),
        head_config=SlsHeadConfig(
            hidden_size=8,
            hidden_state_count=5,
            attention_size=4,
            classifier_size=4,
            dropout=0.0,
        ),
        conv_kernel=(3,),
        conv_stride=(2,),
    )


def _loader(samples: list[AudioSample], batch_size: int) -> DataLoader[AudioBatch]:
    dataset = cast(Dataset[AudioSample], samples)
    return cast(
        DataLoader[AudioBatch],
        DataLoader(dataset, batch_size=batch_size, collate_fn=collate_audio_samples),
    )


def test_stage_a_trains_only_sls_head_with_gradient_accumulation() -> None:
    model = _model()
    freeze_xlsr_encoder(model)
    encoder_before = {
        name: value.detach().clone() for name, value in model.encoder.state_dict().items()
    }
    head_before = {name: value.detach().clone() for name, value in model.head.state_dict().items()}
    loader = _loader(
        [_sample("a", 0.0), _sample("b", 1.0), _sample("c", 0.0), _sample("d", 1.0)],
        batch_size=1,
    )
    optimizer = AdamW(model.head.parameters(), lr=1e-3)

    result = train_xlsr_sls_head_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        precision="fp32",
        gradient_accumulation_steps=3,
        gradient_clip_norm=1.0,
    )

    assert result.examples == 4
    assert not model.encoder.training
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(
        torch.equal(value, encoder_before[name])
        for name, value in model.encoder.state_dict().items()
    )
    assert any(
        not torch.equal(value, head_before[name])
        for name, value in model.head.state_dict().items()
        if value.is_floating_point()
    )


def test_stage_a_evaluation_returns_binary_metrics_without_encoder_gradients() -> None:
    model = _model()
    freeze_xlsr_encoder(model)
    loader = _loader(
        [_sample("a", 0.0), _sample("b", 1.0)],
        batch_size=2,
    )

    result = evaluate_xlsr_sls(model, loader, torch.device("cpu"), precision="fp32")

    assert result.examples == 2
    assert result.balanced_accuracy is not None
    assert all(parameter.grad is None for parameter in model.parameters())


def test_stage_a_rejects_bf16_on_cpu() -> None:
    model = _model()
    loader = _loader([_sample("a", 0.0)], batch_size=1)

    with pytest.raises(ValueError, match="requires CUDA"):
        evaluate_xlsr_sls(model, loader, torch.device("cpu"), precision="bf16")


def test_stage_b_trains_only_tail_blocks_and_head() -> None:
    model = _stage_b_model()
    report = configure_xlsr_stage_b(
        model,
        last_encoder_blocks=2,
        gradient_checkpointing=True,
    )
    encoder = cast(_TinyStageBEncoder, model.encoder)
    frozen_before = {
        name: value.detach().clone()
        for name, value in model.encoder.state_dict().items()
        if name.startswith("projection.") or name.startswith("encoder.layers.0.")
    }
    trainable_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name.startswith("encoder.encoder.layers.2.") or name.startswith("head.")
    }
    loader = _loader(
        [_sample("a", 0.0), _sample("b", 1.0), _sample("c", 0.0), _sample("d", 1.0)],
        batch_size=2,
    )
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )

    result = train_xlsr_sls_stage_b_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        precision="fp32",
        gradient_accumulation_steps=2,
        gradient_clip_norm=1.0,
        last_encoder_blocks=2,
        gradient_checkpointing=True,
    )

    assert result.examples == 4
    assert report.trainable_encoder_blocks == (2, 3)
    assert encoder.gradient_checkpointing
    assert not encoder.encoder.layers[0].training
    assert encoder.encoder.layers[2].training
    assert all(
        torch.equal(value, frozen_before[name])
        for name, value in model.encoder.state_dict().items()
        if name in frozen_before
    )
    assert any(
        not torch.equal(value, trainable_before[name])
        for name, value in model.state_dict().items()
        if name in trainable_before and value.is_floating_point()
    )


def test_stage_b_rejects_unavailable_encoder_block_count() -> None:
    with pytest.raises(ValueError, match="must be in"):
        configure_xlsr_stage_b(
            _stage_b_model(),
            last_encoder_blocks=5,
            gradient_checkpointing=True,
        )


def test_gradient_accumulation_matches_one_full_batch() -> None:
    torch.manual_seed(7)
    accumulated_model = _model()
    full_batch_model = copy.deepcopy(accumulated_model)
    samples = [_sample("a", 0.0), _sample("b", 1.0), _sample("c", 0.0), _sample("d", 1.0)]
    accumulated_optimizer = SGD(accumulated_model.head.parameters(), lr=1e-3)
    full_batch_optimizer = SGD(full_batch_model.head.parameters(), lr=1e-3)

    train_xlsr_sls_head_epoch(
        accumulated_model,
        _loader(samples, batch_size=1),
        accumulated_optimizer,
        torch.device("cpu"),
        precision="fp32",
        gradient_accumulation_steps=4,
        gradient_clip_norm=1_000_000.0,
    )
    train_xlsr_sls_head_epoch(
        full_batch_model,
        _loader(samples, batch_size=4),
        full_batch_optimizer,
        torch.device("cpu"),
        precision="fp32",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1_000_000.0,
    )

    assert all(
        torch.allclose(value, full_batch_model.head.state_dict()[name], atol=1e-6)
        for name, value in accumulated_model.head.state_dict().items()
    )
