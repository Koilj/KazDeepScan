from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from kds.audio.contracts import (
    AudioLimits,
    AudioQuality,
    MediaInfo,
    PreparationStatus,
    SpeechSegment,
    WindowDescriptor,
)
from kds.audio.media import FFmpegClient, validate_declared_mime
from kds.audio.vad import SpeechDetector, WebRtcVadDetector
from kds.audio.waveform import Waveform, measure_quality, read_pcm16_mono_wav
from kds.audio.windows import WindowConfig, build_inference_windows


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    min_rms_dbfs: float = -55.0
    max_clipped_fraction: float = 0.02


@dataclass(frozen=True, slots=True)
class PreparedAudio:
    media: MediaInfo
    waveform: Waveform
    quality: AudioQuality
    speech_segments: tuple[SpeechSegment, ...]
    speech_seconds: float
    windows: tuple[WindowDescriptor, ...]
    status: PreparationStatus
    quality_flags: tuple[str, ...]


class AudioPreparationPipeline:
    """Prepare one local upload or persist an explicitly requested normalized WAV."""

    def __init__(
        self,
        ffmpeg: FFmpegClient | None = None,
        vad: SpeechDetector | None = None,
        limits: AudioLimits | None = None,
        quality_policy: QualityPolicy | None = None,
        window_config: WindowConfig | None = None,
    ) -> None:
        self._ffmpeg = ffmpeg or FFmpegClient()
        self._vad = vad or WebRtcVadDetector()
        self._limits = limits or AudioLimits()
        self._quality_policy = quality_policy or QualityPolicy()
        self._window_config = window_config or WindowConfig()

    def prepare(self, source: Path, declared_mime: str | None = None) -> PreparedAudio:
        with tempfile.TemporaryDirectory(prefix="kds-audio-") as temporary_directory:
            normalized_path = Path(temporary_directory) / "normalized.wav"
            return self.prepare_to_wav(source, normalized_path, declared_mime)

    def prepare_to_wav(
        self, source: Path, destination: Path, declared_mime: str | None = None
    ) -> PreparedAudio:
        """Normalize to a caller-owned new destination and return its quality/readiness result."""

        validate_declared_mime(declared_mime)
        self._ffmpeg.validate_file_size(source, self._limits)
        media = self._ffmpeg.probe(source, self._limits)
        self._ffmpeg.normalize_to_wav(source, destination, self._limits.target_sample_rate)
        waveform = read_pcm16_mono_wav(destination, self._limits.target_sample_rate)

        quality = measure_quality(waveform)
        speech_segments = tuple(self._vad.detect(waveform))
        speech_seconds = sum(segment.duration_seconds for segment in speech_segments)
        flags = self._quality_flags(quality)

        if flags:
            status = PreparationStatus.REJECTED_QUALITY
            windows: tuple[WindowDescriptor, ...] = ()
        elif speech_seconds < self._limits.minimum_speech_seconds:
            status = PreparationStatus.INSUFFICIENT_SPEECH
            flags = ("insufficient_speech",)
            windows = ()
        else:
            status = PreparationStatus.READY
            windows = tuple(build_inference_windows(speech_segments, self._window_config))

        return PreparedAudio(
            media=media,
            waveform=waveform,
            quality=quality,
            speech_segments=speech_segments,
            speech_seconds=speech_seconds,
            windows=windows,
            status=status,
            quality_flags=flags,
        )

    def _quality_flags(self, quality: AudioQuality) -> tuple[str, ...]:
        flags: list[str] = []
        if quality.rms_dbfs < self._quality_policy.min_rms_dbfs:
            flags.append("signal_too_quiet")
        if quality.clipped_fraction > self._quality_policy.max_clipped_fraction:
            flags.append("excessive_clipping")
        return tuple(flags)
