from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import soundfile as sf  # type: ignore[import-untyped]
import torch
from torch import Tensor
from torch.utils.data import Dataset

from kds.data.assets import resolve_asset_path
from kds.data.manifest import ManifestRow

CropMode = Literal["train", "eval"]
LABEL_TO_INDEX = {"bonafide": 0, "spoof": 1}


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    audio_root: Path
    sample_rate: int = 16_000
    window_samples: int = 64_600
    mode: CropMode = "train"
    seed: str = "20260808"

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.window_samples <= 0:
            raise ValueError("sample_rate and window_samples must be positive.")
        if not self.seed:
            raise ValueError("seed must not be empty.")


@dataclass(frozen=True, slots=True)
class AudioSample:
    waveform: Tensor
    label: Tensor
    sample_id: str
    parent_group_id: str
    language: str


class ManifestAudioDataset(Dataset[AudioSample]):
    """Load only preprocessed mono 16 kHz assets referenced by a validated manifest."""

    def __init__(self, rows: list[ManifestRow], config: DatasetConfig) -> None:
        if not rows:
            raise ValueError("Dataset needs at least one manifest row.")
        self._rows = rows
        self._config = config
        self._epoch = 0

    def __len__(self) -> int:
        return len(self._rows)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must not be negative.")
        self._epoch = epoch

    def __getitem__(self, index: int) -> AudioSample:
        row = self._rows[index]
        path = resolve_asset_path(self._config.audio_root, row.relative_path)
        waveform = self._load_normalized_audio(path)
        cropped = self._crop_or_pad(waveform, row.sample_id)
        return AudioSample(
            waveform=cropped,
            label=torch.tensor(LABEL_TO_INDEX[row.label], dtype=torch.float32),
            sample_id=row.sample_id,
            parent_group_id=row.parent_group_id,
            language=row.language,
        )

    def _load_normalized_audio(self, path: Path) -> Tensor:
        try:
            samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        except RuntimeError as error:
            raise ValueError(f"Cannot decode normalized audio: {path}") from error
        if sample_rate != self._config.sample_rate:
            raise ValueError(
                f"Expected {self._config.sample_rate} Hz normalized audio, "
                f"got {sample_rate} Hz: {path}"
            )
        if samples.shape[1] != 1:
            raise ValueError(
                f"Expected mono normalized audio, got {samples.shape[1]} channels: {path}"
            )
        waveform = torch.from_numpy(samples[:, 0].copy())
        if waveform.numel() == 0:
            raise ValueError(f"Normalized audio has no samples: {path}")
        return waveform

    def _crop_or_pad(self, waveform: Tensor, sample_id: str) -> Tensor:
        target = self._config.window_samples
        if waveform.numel() >= target:
            if self._config.mode == "eval" or waveform.numel() == target:
                return waveform[:target]
            max_start = waveform.numel() - target
            start = self._deterministic_crop_start(sample_id, max_start)
            return waveform[start : start + target]

        repeats = math_ceil_division(target, waveform.numel())
        return waveform.repeat(repeats)[:target]

    def _deterministic_crop_start(self, sample_id: str, max_start: int) -> int:
        identity = f"{self._config.seed}:{self._epoch}:{sample_id}".encode()
        return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % (max_start + 1)


def math_ceil_division(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)
