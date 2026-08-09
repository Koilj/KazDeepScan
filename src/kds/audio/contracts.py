from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class AudioErrorCode(StrEnum):
    DEPENDENCY_MISSING = "dependency_missing"
    FILE_NOT_FOUND = "file_not_found"
    INVALID_INPUT = "invalid_input"
    FILE_TOO_LARGE = "file_too_large"
    UNSUPPORTED_MIME = "unsupported_mime"
    UNSUPPORTED_CONTAINER = "unsupported_container"
    DURATION_LIMIT_EXCEEDED = "duration_limit_exceeded"
    DECODE_FAILED = "decode_failed"
    INVALID_WAVEFORM = "invalid_waveform"
    VAD_UNAVAILABLE = "vad_unavailable"


class AudioPipelineError(RuntimeError):
    """Expected user-facing failure from audio preparation."""

    def __init__(self, code: AudioErrorCode, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class AudioLimits:
    max_upload_bytes: int = 50 * 1024 * 1024
    max_duration_seconds: float = 10 * 60
    target_sample_rate: int = 16_000
    minimum_speech_seconds: float = 2.5


@dataclass(frozen=True, slots=True)
class MediaInfo:
    path: Path
    container_names: tuple[str, ...]
    duration_seconds: float
    audio_stream_count: int


@dataclass(frozen=True, slots=True)
class AudioQuality:
    peak: float
    rms_dbfs: float
    clipped_fraction: float
    dc_offset: float


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    start_sample: int
    end_sample: int
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return (self.end_sample - self.start_sample) / self.sample_rate

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_seconds(self) -> float:
        return self.end_sample / self.sample_rate


@dataclass(frozen=True, slots=True)
class WindowDescriptor:
    start_sample: int
    end_sample: int
    target_samples: int
    sample_rate: int

    @property
    def real_samples(self) -> int:
        return self.end_sample - self.start_sample

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_seconds(self) -> float:
        return self.end_sample / self.sample_rate


class PreparationStatus(StrEnum):
    READY = "ready"
    INSUFFICIENT_SPEECH = "insufficient_speech"
    REJECTED_QUALITY = "rejected_quality"
