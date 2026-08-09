from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torchaudio  # type: ignore[import-untyped]
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class B0Config:
    """Fixed feature and architecture parameters for the data-pipeline baseline."""

    sample_rate: int = 16_000
    n_fft: int = 512
    hop_length: int = 160
    n_mels: int = 80
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if self.n_fft <= 0 or self.hop_length <= 0 or self.hop_length > self.n_fft:
            raise ValueError("n_fft and hop_length must be positive and hop_length <= n_fft.")
        if self.n_mels <= 0:
            raise ValueError("n_mels must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")


class _ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2),
        )


class B0LogMelCnn(nn.Module):
    """Small binary sanity baseline; its untrained logits must never be exposed as risk scores."""

    def __init__(self, config: B0Config | None = None) -> None:
        super().__init__()
        self.config = config or B0Config()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            power=2.0,
        )
        self.encoder = nn.Sequential(
            _ConvBlock(1, 32),
            _ConvBlock(32, 64),
            _ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.config.dropout),
            nn.Linear(128, 1),
        )

    def forward(self, waveform: Tensor) -> Tensor:
        """Return one raw spoof logit per waveform of shape ``[batch, samples]``."""

        if waveform.ndim != 2:
            raise ValueError("waveform must have shape [batch, samples].")
        if waveform.shape[1] < self.config.n_fft:
            raise ValueError(f"waveform must contain at least {self.config.n_fft} samples.")
        if not waveform.is_floating_point():
            raise ValueError("waveform must use a floating-point dtype normalized to [-1, 1].")

        mel = cast(Tensor, self.mel_spectrogram(waveform))
        log_mel = torch.log(mel.clamp_min(1e-10)).unsqueeze(1)
        encoded = cast(Tensor, self.encoder(log_mel))
        logits = cast(Tensor, self.classifier(encoded))
        return logits.squeeze(-1)
