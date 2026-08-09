from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from kds.data.assets import require_valid_assets
from kds.data.dataset import DatasetConfig, ManifestAudioDataset
from kds.data.licenses import load_license_ledger, validate_manifest_licenses
from kds.data.manifest import load_manifest, validate_manifest
from kds.eval import classification_confidence_intervals
from kds.models import B0Config, B0LogMelCnn
from kds.training import evaluate_b0, make_audio_loader


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


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
                "calibrated": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
