from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from kds.data.assets import require_valid_assets
from kds.data.dataset import DatasetConfig, ManifestAudioDataset
from kds.data.licenses import load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestRow, load_manifest, validate_manifest
from kds.eval import classification_confidence_intervals, wilson_interval
from kds.models import B0Config, B0LogMelCnn
from kds.training import evaluate_b0, make_audio_loader


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _stratum_keys(row: ManifestRow) -> tuple[str, ...]:
    if row.label == "bonafide":
        return (f"bonafide_source:{row.source_name}",)
    return (
        f"spoof_generator_family:{row.generator_family}",
        f"spoof_voice_id:{row.voice_id}",
    )


def _stratified_metrics(
    model: B0LogMelCnn,
    rows: list[ManifestRow],
    *,
    audio_root: Path,
    batch_size: int,
    seed: str,
    device: torch.device,
    num_workers: int,
) -> dict[str, dict[str, object]]:
    """Evaluate mutually readable source/generator strata without emitting a score API."""

    row_by_sample_id = {row.sample_id: row for row in rows}
    dataset = ManifestAudioDataset(
        rows, DatasetConfig(audio_root=audio_root, mode="eval", seed=seed)
    )
    loader = make_audio_loader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    counts: dict[str, list[int]] = {}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            predictions = model(batch.waveforms.to(device, non_blocking=True)) >= 0.0
            for sample_id, predicted_spoof in zip(
                batch.sample_ids, predictions.tolist(), strict=True
            ):
                row = row_by_sample_id[sample_id]
                correct = bool(predicted_spoof) == (row.label == "spoof")
                for key in _stratum_keys(row):
                    bucket = counts.setdefault(key, [0, 0])
                    bucket[0] += int(correct)
                    bucket[1] += 1
    return {
        key: {
            "correct": correct,
            "examples": examples,
            "accuracy": correct / examples,
            "confidence_interval": asdict(wilson_interval(correct, examples)),
        }
        for key, (correct, examples) in sorted(counts.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a local B0 checkpoint without calibration."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", default="20260809")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    arguments = parser.parse_args()

    if arguments.batch_size <= 0 or arguments.num_workers < 0:
        raise ValueError("batch-size must be positive; num-workers must be non-negative.")
    if not arguments.checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {arguments.checkpoint}")
    device = _device(arguments.device)
    manifest_rows = load_manifest(arguments.manifest)
    validate_manifest(manifest_rows)
    validate_manifest_licenses(manifest_rows, load_license_ledger(arguments.license_ledger))
    rows = [row for row in manifest_rows if row.split == arguments.split]
    if not rows:
        raise ValueError(f"Manifest has no rows for split={arguments.split!r}.")
    require_valid_assets(rows, arguments.audio_root)
    checkpoint = torch.load(arguments.checkpoint, map_location=device, weights_only=True)
    model = B0LogMelCnn(B0Config(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    dataset = ManifestAudioDataset(
        rows, DatasetConfig(audio_root=arguments.audio_root, mode="eval", seed=arguments.seed)
    )
    result = evaluate_b0(
        model,
        make_audio_loader(
            dataset,
            batch_size=arguments.batch_size,
            shuffle=False,
            num_workers=arguments.num_workers,
        ),
        device,
    )
    label_counts = {
        label: sum(row.label == label for row in rows) for label in ("bonafide", "spoof")
    }
    confidence_intervals = classification_confidence_intervals(
        correct=result.correct,
        examples=result.examples,
        bonafide_correct=result.bonafide_correct,
        bonafide_examples=result.bonafide_examples,
        spoof_correct=result.spoof_correct,
        spoof_examples=result.spoof_examples,
    )
    stratified_metrics = _stratified_metrics(
        model,
        rows,
        audio_root=arguments.audio_root,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        device=device,
        num_workers=arguments.num_workers,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "checkpoint": str(arguments.checkpoint),
                "manifest": str(arguments.manifest),
                "split": arguments.split,
                "rows": len(rows),
                "label_counts": label_counts,
                "loss": result.loss,
                "accuracy": result.accuracy,
                "correct": result.correct,
                "bonafide_accuracy": result.bonafide_accuracy,
                "bonafide_correct": result.bonafide_correct,
                "spoof_accuracy": result.spoof_accuracy,
                "spoof_correct": result.spoof_correct,
                "balanced_accuracy": result.balanced_accuracy,
                "confidence_intervals": {
                    name: asdict(interval) for name, interval in confidence_intervals.items()
                },
                "stratified_metrics": stratified_metrics,
                "calibrated": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
