from __future__ import annotations

import unittest
import wave
from array import array
from pathlib import Path

from kds.audio.contracts import AudioLimits, MediaInfo, PreparationStatus, SpeechSegment
from kds.audio.pipeline import AudioPreparationPipeline


class StubFFmpeg:
    def validate_file_size(self, source: Path, limits: AudioLimits) -> None:
        self.validated_source = source
        self.limits = limits

    def probe(self, source: Path, limits: AudioLimits) -> MediaInfo:
        return MediaInfo(source, ("wav",), 3.0, 1)

    def normalize_to_wav(self, source: Path, destination: Path, sample_rate: int) -> None:
        with wave.open(str(destination), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(array("h", [8_000] * (sample_rate * 3)).tobytes())


class StaticVad:
    def __init__(self, segments: list[SpeechSegment]) -> None:
        self._segments = segments

    def detect(self, _waveform: object) -> list[SpeechSegment]:
        return self._segments


class PipelineTests(unittest.TestCase):
    def test_ready_audio_uses_speech_windows_and_discards_temp_normalization(self) -> None:
        ffmpeg = StubFFmpeg()
        pipeline = AudioPreparationPipeline(
            ffmpeg=ffmpeg,
            vad=StaticVad([SpeechSegment(0, 48_000, 16_000)]),
        )

        prepared = pipeline.prepare(Path("upload.wav"), "audio/wav")

        self.assertEqual(prepared.status, PreparationStatus.READY)
        self.assertEqual(prepared.speech_seconds, 3.0)
        self.assertEqual(len(prepared.windows), 1)
        self.assertEqual(prepared.windows[0].real_samples, 48_000)
        self.assertEqual(ffmpeg.validated_source, Path("upload.wav"))

    def test_audio_with_less_than_minimum_speech_returns_no_windows(self) -> None:
        pipeline = AudioPreparationPipeline(
            ffmpeg=StubFFmpeg(),
            vad=StaticVad([SpeechSegment(0, 16_000, 16_000)]),
        )

        prepared = pipeline.prepare(Path("upload.wav"), "audio/wav")

        self.assertEqual(prepared.status, PreparationStatus.INSUFFICIENT_SPEECH)
        self.assertEqual(prepared.quality_flags, ("insufficient_speech",))
        self.assertEqual(prepared.windows, ())


if __name__ == "__main__":
    unittest.main()
