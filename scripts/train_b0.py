from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor
from torch.optim import AdamW

from kds.data.assets import require_valid_assets
from kds.data.dataset import DatasetConfig, ManifestAudioDataset
from kds.data.licenses import load_license_ledger, validate_training_protocol
from kds.data.manifest import ManifestRow, load_manifest
from kds.models import B0Config, B0LogMelCnn
from kds.training import evaluate_b0, make_audio_loader, train_b0_epoch


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _rows_for_split(manifest_rows: list[ManifestRow], split: str) -> list[ManifestRow]:
    rows = [row for row in manifest_rows if row.split == split]
    if not rows:
        raise ValueError(f"Manifest has no rows for split={split!r}.")
    labels = {row.label for row in rows}
    if labels != {"bonafide", "spoof"}:
        raise ValueError(f"split={split!r} must include both bonafide and spoof rows.")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train B0 from verified processed manifest assets."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument(
        "--purpose",
        choices=("research", "product"),
        required=True,
        help="Explicit protocol intent; product requires verified groups and independent OOD.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", default="20260808")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    arguments = parser.parse_args()

    if arguments.epochs <= 0 or arguments.batch_size <= 0 or arguments.num_workers < 0:
        raise ValueError(
            "epochs and batch-size must be positive; num-workers must be non-negative."
        )
    if arguments.output.exists():
        raise ValueError(f"Refusing to overwrite checkpoint: {arguments.output}")
    if not arguments.output.parent.is_dir():
        raise ValueError(f"Checkpoint output directory does not exist: {arguments.output.parent}")

    torch.manual_seed(int(arguments.seed))
    device = _device(arguments.device)
    manifest_rows = load_manifest(arguments.manifest)
    protocol_report = validate_training_protocol(
        manifest_rows,
        load_license_ledger(arguments.license_ledger),
        purpose=arguments.purpose,
    )
    train_rows = _rows_for_split(manifest_rows, "train")
    dev_rows = _rows_for_split(manifest_rows, "dev")
    require_valid_assets([*train_rows, *dev_rows], arguments.audio_root)
    train_dataset = ManifestAudioDataset(
        train_rows,
        DatasetConfig(audio_root=arguments.audio_root, mode="train", seed=arguments.seed),
    )
    dev_dataset = ManifestAudioDataset(
        dev_rows,
        DatasetConfig(audio_root=arguments.audio_root, mode="eval", seed=arguments.seed),
    )
    train_loader = make_audio_loader(
        train_dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=arguments.num_workers,
    )
    dev_loader = make_audio_loader(
        dev_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
    )
    model = B0LogMelCnn(B0Config()).to(device)
    optimizer = AdamW(
        model.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay
    )

    history: list[dict[str, float | int | None]] = []
    best_dev_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    for epoch in range(arguments.epochs):
        train_dataset.set_epoch(epoch)
        train_result = train_b0_epoch(model, train_loader, optimizer, device)
        dev_result = evaluate_b0(model, dev_loader, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_result.loss,
                "train_accuracy": train_result.accuracy,
                "train_balanced_accuracy": train_result.balanced_accuracy,
                "dev_loss": dev_result.loss,
                "dev_accuracy": dev_result.accuracy,
                "dev_balanced_accuracy": dev_result.balanced_accuracy,
            }
        )
        if dev_result.loss < best_dev_loss:
            best_dev_loss = dev_result.loss
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("No checkpoint was produced.")
    checkpoint = {
        "model_name": "b0_logmel_cnn",
        "model_config": asdict(model.config),
        "training_seed": arguments.seed,
        "training_purpose": protocol_report.purpose,
        "training_protocol": asdict(protocol_report),
        "best_dev_loss": best_dev_loss,
        "state_dict": best_state,
    }
    torch.save(checkpoint, arguments.output)
    print(
        json.dumps(
            {
                "status": "ok",
                "device": str(device),
                "purpose": protocol_report.purpose,
                "checkpoint": str(arguments.output),
                "history": history,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
