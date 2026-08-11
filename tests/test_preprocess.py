from __future__ import annotations

import hashlib
import wave
from array import array
from dataclasses import replace
from pathlib import Path

from kds.audio.contracts import MediaInfo, SpeechSegment
from kds.audio.pipeline import AudioPreparationPipeline
from kds.data.manifest import ManifestRow
from kds.data.preprocess import (
    preprocess_rows,
    processed_relative_path,
    reuse_preprocessed_rows,
)
from tests.factories import manifest_mapping


class StubFFmpeg:
    def validate_file_size(self, _source: Path, _limits: object) -> None:
        return None

    def probe(self, source: Path, _limits: object) -> MediaInfo:
        return MediaInfo(source, ("wav",), 3.0, 1)

    def normalize_to_wav(self, _source: Path, destination: Path, sample_rate: int) -> None:
        with wave.open(str(destination), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(array("h", [8_000] * (sample_rate * 3)).tobytes())


class StaticVad:
    def detect(self, _waveform: object) -> list[SpeechSegment]:
        return [SpeechSegment(0, 48_000, 16_000)]


class FailingPreparer:
    def __init__(self, pipeline: AudioPreparationPipeline, failing_source: Path) -> None:
        self._pipeline = pipeline
        self._failing_source = failing_source

    def prepare_to_wav(self, source: Path, destination: Path) -> object:
        if source == self._failing_source:
            raise ValueError("intentional test failure")
        return self._pipeline.prepare_to_wav(source, destination)


def test_preprocess_rows_rewrites_path_and_digest_for_ready_audio(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw" / "source.wav"
    raw_path.parent.mkdir()
    raw_path.write_bytes(b"raw-audio")
    raw_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    row = ManifestRow.from_mapping(
        manifest_mapping(relative_path="raw/source.wav", sha256=raw_sha256), row_number=2
    )
    pipeline = AudioPreparationPipeline(ffmpeg=StubFFmpeg(), vad=StaticVad())

    report = preprocess_rows([row], tmp_path, pipeline)

    assert report.is_successful
    processed = report.processed_rows[0]
    assert processed.relative_path == processed_relative_path(row)
    assert processed.sha256 != row.sha256
    assert (tmp_path / processed.relative_path).is_file()
    assert processed.duration_s == 3.0
    assert processed.codec == "wav"


def test_preprocess_rows_publishes_nothing_when_one_asset_fails(tmp_path: Path) -> None:
    raw_first = tmp_path / "raw" / "first.wav"
    raw_second = tmp_path / "raw" / "second.wav"
    raw_first.parent.mkdir()
    raw_first.write_bytes(b"raw-first")
    raw_second.write_bytes(b"raw-second")
    first = ManifestRow.from_mapping(
        manifest_mapping(
            sample_id="first",
            relative_path="raw/first.wav",
            sha256=hashlib.sha256(raw_first.read_bytes()).hexdigest(),
        ),
        row_number=2,
    )
    second = ManifestRow.from_mapping(
        manifest_mapping(
            sample_id="second",
            relative_path="raw/second.wav",
            sha256=hashlib.sha256(raw_second.read_bytes()).hexdigest(),
        ),
        row_number=3,
    )
    pipeline = AudioPreparationPipeline(ffmpeg=StubFFmpeg(), vad=StaticVad())

    report = preprocess_rows([first, second], tmp_path, FailingPreparer(pipeline, raw_second))

    assert not report.is_successful
    assert report.processed_rows == ()
    assert len(report.issues) == 1
    assert not (tmp_path / processed_relative_path(first)).exists()


def test_preprocess_rows_can_publish_ready_assets_when_rejections_are_explicit(
    tmp_path: Path,
) -> None:
    raw_first = tmp_path / "raw" / "first.wav"
    raw_second = tmp_path / "raw" / "second.wav"
    raw_first.parent.mkdir()
    raw_first.write_bytes(b"raw-first")
    raw_second.write_bytes(b"raw-second")
    first = ManifestRow.from_mapping(
        manifest_mapping(
            sample_id="first",
            relative_path="raw/first.wav",
            sha256=hashlib.sha256(raw_first.read_bytes()).hexdigest(),
        ),
        row_number=2,
    )
    second = ManifestRow.from_mapping(
        manifest_mapping(
            sample_id="second",
            relative_path="raw/second.wav",
            sha256=hashlib.sha256(raw_second.read_bytes()).hexdigest(),
        ),
        row_number=3,
    )
    pipeline = AudioPreparationPipeline(ffmpeg=StubFFmpeg(), vad=StaticVad())

    report = preprocess_rows(
        [first, second],
        tmp_path,
        FailingPreparer(pipeline, raw_second),
        allow_rejections=True,
    )

    assert not report.is_successful
    assert [row.sample_id for row in report.processed_rows] == ["first"]
    assert len(report.issues) == 1
    assert (tmp_path / processed_relative_path(first)).is_file()


def test_reuse_preprocessed_rows_requires_same_sample_and_raw_bytes() -> None:
    prior_raw = ManifestRow.from_mapping(
        manifest_mapping(
            sample_id="same",
            relative_path="raw/same.wav",
            sha256="a" * 64,
            label="spoof",
            generator_family="tts",
            generator_name="generator",
            generator_version="bad-old-value",
            voice_id="voice",
        ),
        row_number=2,
    )
    prior_ready = replace(
        prior_raw,
        relative_path=processed_relative_path(prior_raw),
        sha256="b" * 64,
        duration_s=4.25,
        codec="wav",
    )
    current = replace(prior_raw, split="dev", generator_version="normalized-v2")
    different_bytes = replace(current, sample_id="changed", sha256="c" * 64)

    result = reuse_preprocessed_rows(
        [current, different_bytes], [prior_raw], [prior_ready]
    )

    assert result.reused_rows == (
        replace(
            current,
            relative_path=prior_ready.relative_path,
            sha256=prior_ready.sha256,
            duration_s=prior_ready.duration_s,
            codec="wav",
        ),
    )
    assert result.reused_rows[0].generator_version == "normalized-v2"
    assert result.reused_rows[0].split == "dev"
    assert result.remaining_rows == (different_bytes,)
