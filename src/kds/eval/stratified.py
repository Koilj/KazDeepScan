"""Read-only record-level source, generator-family and voice evaluation strata."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from kds.data.dataset import DatasetConfig, ManifestAudioDataset
from kds.data.manifest import ManifestRow
from kds.eval.metrics import wilson_interval
from kds.models import B0LogMelCnn
from kds.training import EpochResult, make_audio_loader


def stratum_keys(row: ManifestRow) -> tuple[str, ...]:
    """Return provenance strata without claiming a source pseudo-ID is a real speaker."""

    if row.label == "bonafide":
        return (f"bonafide_source:{row.source_name}",)
    return (
        f"spoof_generator_family:{row.generator_family}",
        f"spoof_voice_id:{row.voice_id}",
    )


def stratified_metrics(
    model: B0LogMelCnn,
    rows: list[ManifestRow],
    *,
    audio_root: Path,
    batch_size: int,
    seed: str,
    device: torch.device,
    num_workers: int,
) -> dict[str, dict[str, object]]:
    """Return source/generator/voice metrics from one read-only model traversal."""

    _, metrics = evaluate_b0_with_strata(
        model,
        rows,
        audio_root=audio_root,
        batch_size=batch_size,
        seed=seed,
        device=device,
        num_workers=num_workers,
    )
    return metrics


def evaluate_b0_with_strata(
    model: B0LogMelCnn,
    rows: list[ManifestRow],
    *,
    audio_root: Path,
    batch_size: int,
    seed: str,
    device: torch.device,
    num_workers: int,
    sample_rate: int = 16_000,
    window_samples: int = 64_600,
) -> tuple[EpochResult, dict[str, dict[str, object]]]:
    """Evaluate aggregate and provenance metrics from exactly one loader traversal."""

    row_by_sample_id = {row.sample_id: row for row in rows}
    if len(row_by_sample_id) != len(rows):
        raise ValueError("Rows for stratified evaluation must have unique sample_id values.")
    dataset = ManifestAudioDataset(
        rows,
        DatasetConfig(
            audio_root=audio_root,
            sample_rate=sample_rate,
            window_samples=window_samples,
            mode="eval",
            seed=seed,
        ),
    )
    loader = make_audio_loader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    counts: dict[str, list[int]] = {}
    total_loss = 0.0
    total_correct = 0
    examples = 0
    bonafide_examples = 0
    bonafide_correct = 0
    spoof_examples = 0
    spoof_correct = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            waveforms = batch.waveforms.to(device, non_blocking=True)
            labels = batch.labels.to(device, non_blocking=True)
            logits = model(waveforms)
            loss = criterion(logits, labels)
            predictions = logits >= 0.0
            expected_spoof = labels >= 0.5
            current_batch_size = labels.numel()
            total_loss += float(loss) * current_batch_size
            total_correct += int((predictions == expected_spoof).sum())
            bonafide_mask = ~expected_spoof
            spoof_mask = expected_spoof
            bonafide_examples += int(bonafide_mask.sum())
            bonafide_correct += int(
                (predictions[bonafide_mask] == expected_spoof[bonafide_mask]).sum()
            )
            spoof_examples += int(spoof_mask.sum())
            spoof_correct += int((predictions[spoof_mask] == expected_spoof[spoof_mask]).sum())
            examples += current_batch_size
            for sample_id, predicted_spoof in zip(
                batch.sample_ids, predictions.tolist(), strict=True
            ):
                row = row_by_sample_id[sample_id]
                sample_is_correct = bool(predicted_spoof) == (row.label == "spoof")
                for key in stratum_keys(row):
                    bucket = counts.setdefault(key, [0, 0])
                    bucket[0] += int(sample_is_correct)
                    bucket[1] += 1
    if examples == 0:
        raise ValueError("Rows for stratified evaluation must not be empty.")
    bonafide_accuracy = bonafide_correct / bonafide_examples if bonafide_examples else None
    spoof_accuracy = spoof_correct / spoof_examples if spoof_examples else None
    balanced_accuracy = (
        (bonafide_accuracy + spoof_accuracy) / 2
        if bonafide_accuracy is not None and spoof_accuracy is not None
        else None
    )
    result = EpochResult(
        loss=total_loss / examples,
        accuracy=total_correct / examples,
        correct=total_correct,
        examples=examples,
        bonafide_examples=bonafide_examples,
        spoof_examples=spoof_examples,
        bonafide_correct=bonafide_correct,
        spoof_correct=spoof_correct,
        bonafide_accuracy=bonafide_accuracy,
        spoof_accuracy=spoof_accuracy,
        balanced_accuracy=balanced_accuracy,
    )
    metrics = {
        key: {
            "correct": stratum_correct,
            "examples": stratum_examples,
            "accuracy": stratum_correct / stratum_examples,
            "confidence_interval": asdict(wilson_interval(stratum_correct, stratum_examples)),
        }
        for key, (stratum_correct, stratum_examples) in sorted(counts.items())
    }
    return result, metrics
