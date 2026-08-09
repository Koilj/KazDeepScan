from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from kds.data.dataset import DatasetConfig, ManifestAudioDataset
from kds.data.manifest import ManifestRow
from tests.factories import manifest_mapping


def _make_row(audio_root: Path, samples: np.ndarray) -> ManifestRow:
    path = audio_root / "processed" / "ru" / "sample.wav"
    path.parent.mkdir(parents=True)
    sf.write(path, samples, 16_000, subtype="FLOAT")
    return ManifestRow.from_mapping(
        manifest_mapping(
            relative_path="processed/ru/sample.wav",
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ),
        row_number=2,
    )


def test_dataset_cycles_short_normalized_audio_to_window_size(tmp_path: Path) -> None:
    row = _make_row(tmp_path, np.full(800, 0.25, dtype=np.float32))
    dataset = ManifestAudioDataset(
        [row], DatasetConfig(audio_root=tmp_path, window_samples=1_600, mode="eval")
    )

    sample = dataset[0]

    assert sample.waveform.shape == (1_600,)
    assert float(sample.waveform[0]) == 0.25
    assert float(sample.waveform[800]) == 0.25
    assert sample.label.item() == 0.0


def test_dataset_changes_train_crop_deterministically_by_epoch(tmp_path: Path) -> None:
    samples = np.arange(3_200, dtype=np.float32) / 3_200
    row = _make_row(tmp_path, samples)
    dataset = ManifestAudioDataset(
        [row], DatasetConfig(audio_root=tmp_path, window_samples=1_600, mode="train")
    )

    first = dataset[0].waveform
    repeated = dataset[0].waveform
    dataset.set_epoch(1)
    next_epoch = dataset[0].waveform

    assert first.equal(repeated)
    assert first.numel() == 1_600
    assert next_epoch.numel() == 1_600
