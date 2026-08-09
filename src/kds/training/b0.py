from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from kds.data.dataset import AudioSample
from kds.models import B0LogMelCnn


@dataclass(frozen=True, slots=True)
class AudioBatch:
    waveforms: Tensor
    labels: Tensor
    sample_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EpochResult:
    loss: float
    accuracy: float
    examples: int


def collate_audio_samples(samples: list[AudioSample]) -> AudioBatch:
    if not samples:
        raise ValueError("Cannot collate an empty audio batch.")
    return AudioBatch(
        waveforms=torch.stack([sample.waveform for sample in samples]),
        labels=torch.stack([sample.label for sample in samples]),
        sample_ids=tuple(sample.sample_id for sample in samples),
    )


def make_audio_loader(
    dataset: Dataset[AudioSample], batch_size: int, shuffle: bool, num_workers: int
) -> DataLoader[AudioBatch]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_audio_samples,
    )
    return cast(DataLoader[AudioBatch], loader)


def train_b0_epoch(
    model: B0LogMelCnn,
    data_loader: DataLoader[AudioBatch],
    optimizer: Optimizer,
    device: torch.device,
) -> EpochResult:
    model.train()
    return _run_b0_epoch(model, data_loader, device, optimizer)


def evaluate_b0(
    model: B0LogMelCnn, data_loader: DataLoader[AudioBatch], device: torch.device
) -> EpochResult:
    model.eval()
    with torch.inference_mode():
        return _run_b0_epoch(model, data_loader, device, optimizer=None)


def _run_b0_epoch(
    model: B0LogMelCnn,
    data_loader: DataLoader[AudioBatch],
    device: torch.device,
    optimizer: Optimizer | None,
) -> EpochResult:
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = 0
    examples = 0

    for batch in data_loader:
        waveforms = batch.waveforms.to(device, non_blocking=True)
        labels = batch.labels.to(device, non_blocking=True)
        logits = model(waveforms)
        loss = criterion(logits, labels)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = labels.numel()
        total_loss += float(loss.detach()) * batch_size
        correct += int(((logits.detach() >= 0.0) == (labels >= 0.5)).sum())
        examples += batch_size

    if examples == 0:
        raise ValueError("Data loader yielded no batches.")
    return EpochResult(loss=total_loss / examples, accuracy=correct / examples, examples=examples)
