from __future__ import annotations

import math
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from kds.audio.contracts import AudioErrorCode, AudioPipelineError, SpeechSegment
from kds.audio.waveform import Waveform


class SpeechDetector(Protocol):
    def detect(self, waveform: Waveform) -> list[SpeechSegment]:
        """Return ordered, non-overlapping spans of detected speech."""


@dataclass(frozen=True, slots=True)
class VadConfig:
    aggressiveness: int = 2
    frame_duration_ms: int = 30
    merge_gap_seconds: float = 0.18
    min_segment_seconds: float = 0.12

    def __post_init__(self) -> None:
        if self.aggressiveness not in {0, 1, 2, 3}:
            raise ValueError("WebRTC VAD aggressiveness must be an integer from 0 to 3.")
        if self.frame_duration_ms not in {10, 20, 30}:
            raise ValueError("WebRTC VAD frame duration must be 10, 20, or 30 ms.")


def _merge_active_frames(
    active_frames: Sequence[bool],
    frame_samples: int,
    sample_rate: int,
    config: VadConfig,
    total_samples: int,
) -> list[SpeechSegment]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_speech in enumerate(active_frames):
        if is_speech and start is None:
            start = index * frame_samples
        if not is_speech and start is not None:
            spans.append((start, index * frame_samples))
            start = None
    if start is not None:
        spans.append((start, total_samples))

    max_gap_samples = round(config.merge_gap_seconds * sample_rate)
    merged: list[tuple[int, int]] = []
    for start_sample, end_sample in spans:
        end_sample = min(end_sample, total_samples)
        if merged and start_sample - merged[-1][1] <= max_gap_samples:
            merged[-1] = (merged[-1][0], end_sample)
        else:
            merged.append((start_sample, end_sample))

    minimum_samples = round(config.min_segment_seconds * sample_rate)
    return [
        SpeechSegment(start, end, sample_rate)
        for start, end in merged
        if end - start >= minimum_samples
    ]


class WebRtcVadDetector:
    """Production VAD backed by WebRTC VAD's fixed PCM frame contract."""

    def __init__(self, config: VadConfig | None = None) -> None:
        self.config = config or VadConfig()
        try:
            import webrtcvad  # type: ignore[import-untyped]
        except ImportError as error:
            raise AudioPipelineError(
                AudioErrorCode.VAD_UNAVAILABLE,
                "webrtcvad-wheels is required for production speech detection.",
            ) from error
        self._engine = webrtcvad.Vad(self.config.aggressiveness)

    def detect(self, waveform: Waveform) -> list[SpeechSegment]:
        if waveform.sample_rate not in {8_000, 16_000, 32_000, 48_000}:
            raise AudioPipelineError(
                AudioErrorCode.INVALID_WAVEFORM,
                f"WebRTC VAD does not accept {waveform.sample_rate} Hz audio.",
            )
        frame_samples = waveform.sample_rate * self.config.frame_duration_ms // 1000
        full_frames = len(waveform.samples) // frame_samples
        active_frames: list[bool] = []
        for index in range(full_frames):
            start = index * frame_samples
            frame = waveform.samples[start : start + frame_samples]
            active_frames.append(self._engine.is_speech(frame.tobytes(), waveform.sample_rate))
        return _merge_active_frames(
            active_frames,
            frame_samples,
            waveform.sample_rate,
            self.config,
            len(waveform.samples),
        )


class EnergyVadDetector:
    """Minimal deterministic detector used only in unit tests, never by the default pipeline."""

    def __init__(self, frame_duration_ms: int = 30, threshold_dbfs: float = -45.0) -> None:
        self._frame_duration_ms = frame_duration_ms
        self._threshold_dbfs = threshold_dbfs

    def detect(self, waveform: Waveform) -> list[SpeechSegment]:
        frame_samples = waveform.sample_rate * self._frame_duration_ms // 1000
        if frame_samples <= 0:
            raise ValueError("Frame size must be positive.")
        active_frames: list[bool] = []
        for start in range(0, len(waveform.samples) - frame_samples + 1, frame_samples):
            frame: array[int] = waveform.samples[start : start + frame_samples]
            rms = math.sqrt(sum(sample * sample for sample in frame) / len(frame)) / 32_768.0
            dbfs = 20.0 * math.log10(max(rms, 1e-12))
            active_frames.append(dbfs >= self._threshold_dbfs)
        return _merge_active_frames(
            active_frames,
            frame_samples,
            waveform.sample_rate,
            VadConfig(frame_duration_ms=self._frame_duration_ms),
            len(waveform.samples),
        )
