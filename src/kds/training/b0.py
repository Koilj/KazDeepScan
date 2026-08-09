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
    bonafide_examples: int
    spoof_examples: int
    bonafide_accuracy: float | None
    spoof_accuracy: float | None
    balanced_accuracy: float | None


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
    bonafide_examples = 0
    bonafide_correct = 0
    spoof_examples = 0
    spoof_correct = 0

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
        predictions = logits.detach() >= 0.0
        expected_spoof = labels >= 0.5
        correct += int((predictions == expected_spoof).sum())
        bonafide_mask = ~expected_spoof
        spoof_mask = expected_spoof
        bonafide_examples += int(bonafide_mask.sum())
        bonafide_correct += int((predictions[bonafide_mask] == expected_spoof[bonafide_mask]).sum())
        spoof_examples += int(spoof_mask.sum())
        spoof_correct += int((predictions[spoof_mask] == expected_spoof[spoof_mask]).sum())
        examples += batch_size

    if examples == 0:
        raise ValueError("Data loader yielded no batches.")
    bonafide_accuracy = bonafide_correct / bonafide_examples if bonafide_examples else None
    spoof_accuracy = spoof_correct / spoof_examples if spoof_examples else None
    balanced_accuracy = (
        (bonafide_accuracy + spoof_accuracy) / 2
        if bonafide_accuracy is not None and spoof_accuracy is not None
        else None
    )
    return EpochResult(
        loss=total_loss / examples,
        accuracy=correct / examples,
        examples=examples,
        bonafide_examples=bonafide_examples,
        spoof_examples=spoof_examples,
        bonafide_accuracy=bonafide_accuracy,
        spoof_accuracy=spoof_accuracy,
        balanced_accuracy=balanced_accuracy,
    )
