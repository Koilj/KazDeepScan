from __future__ import annotations

import math
import sys
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path

from kds.audio.contracts import AudioErrorCode, AudioPipelineError, AudioQuality


@dataclass(frozen=True, slots=True)
class Waveform:
    samples: array[int]
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate


def read_pcm16_mono_wav(path: Path, expected_sample_rate: int) -> Waveform:
    """Read the canonical WAV emitted by FFmpeg, rejecting non-canonical audio."""

    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            compression = wav_file.getcomptype()
            frame_count = wav_file.getnframes()
            raw_frames = wav_file.readframes(frame_count)
    except (wave.Error, OSError, EOFError) as error:
        raise AudioPipelineError(
            AudioErrorCode.INVALID_WAVEFORM,
            "Normalized WAV could not be read.",
        ) from error

    is_canonical = (
        channels == 1
        and sample_width == 2
        and sample_rate == expected_sample_rate
        and compression == "NONE"
    )
    if not is_canonical:
        raise AudioPipelineError(
            AudioErrorCode.INVALID_WAVEFORM,
            "Normalized WAV must be uncompressed mono PCM S16LE at the configured sample rate.",
        )
    if len(raw_frames) % 2 != 0:
        raise AudioPipelineError(
            AudioErrorCode.INVALID_WAVEFORM, "WAV contains an incomplete sample."
        )

    samples = array("h")
    samples.frombytes(raw_frames)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        raise AudioPipelineError(AudioErrorCode.INVALID_WAVEFORM, "WAV contains no samples.")
    return Waveform(samples=samples, sample_rate=sample_rate)


def measure_quality(waveform: Waveform) -> AudioQuality:
    """Calculate transparent, model-independent signal diagnostics."""

    samples = waveform.samples
    count = len(samples)
    peak_int = max(abs(sample) for sample in samples)
    total = sum(samples)
    sum_squares = sum(sample * sample for sample in samples)
    rms = math.sqrt(sum_squares / count) / 32_768.0
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))
    clipped_fraction = sum(1 for sample in samples if abs(sample) >= 32_700) / count
    return AudioQuality(
        peak=peak_int / 32_768.0,
        rms_dbfs=rms_dbfs,
        clipped_fraction=clipped_fraction,
        dc_offset=(total / count) / 32_768.0,
    )
