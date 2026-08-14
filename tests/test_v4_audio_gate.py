from __future__ import annotations

from array import array
from pathlib import Path

import pytest

from kds.audio.contracts import SpeechSegment
from kds.audio.waveform import Waveform
from kds.data.v4_audio_gate import (
    V4AudioGateError,
    V4AudioSignature,
    V4DecodedCandidate,
    V4DecodeResult,
    V4DecodeTask,
    canonical_audio_fingerprint,
    decide_v4_decoded_audio_eligibility,
    decode_tasks_by_id,
    decoded_relative_path,
    find_near_audio_matches,
    fingerprint_hamming_distance,
)


def _wave(samples: list[int]) -> Waveform:
    return Waveform(array("h", samples), 16_000)


def test_audio_fingerprint_is_deterministic_and_gain_tolerant() -> None:
    base = [int(8_000 * ((index % 80) / 40 - 1)) for index in range(16_000)]
    quiet = [sample // 2 for sample in base]
    segment = (SpeechSegment(0, 16_000, 16_000),)

    first = canonical_audio_fingerprint(_wave(base), segment)
    second = canonical_audio_fingerprint(_wave(quiet), segment)

    assert len(first) == 64
    assert fingerprint_hamming_distance(first, second) <= 8


def test_audio_fingerprint_handles_less_than_sixteen_speech_frames() -> None:
    waveform = _wave([1000, -1000] * 400)
    segment = (SpeechSegment(0, 800, 16_000),)

    fingerprint = canonical_audio_fingerprint(waveform, segment)

    assert len(fingerprint) == 64


def test_near_audio_match_uses_hamming_and_speech_duration_gates() -> None:
    reference = V4AudioSignature("history", "1" * 64, "0" * 64, 4.0)
    near = V4AudioSignature("candidate", "2" * 64, "0" * 63 + "1", 4.1)
    wrong_duration = V4AudioSignature("long", "3" * 64, "0" * 64, 8.0)

    matches = find_near_audio_matches((near, wrong_duration), (reference,))

    assert len(matches) == 1
    assert matches[0].candidate_identity == "candidate"
    assert matches[0].hamming_distance == 1


def test_decode_task_paths_are_content_addressed_and_unique(tmp_path: Path) -> None:
    digest = "a" * 64
    relative = decoded_relative_path(digest)
    task = V4DecodeTask(
        sample_id="sample",
        raw_relative_path="raw/sample.wav",
        raw_sha256=digest,
        source_path=str(tmp_path / "raw.wav"),
        decoded_relative_path=relative,
        destination_path=str(tmp_path / "decoded.wav"),
    )

    assert decode_tasks_by_id((task,)) == {"sample": task}
    with pytest.raises(V4AudioGateError, match="duplicate"):
        decode_tasks_by_id((task, task))


def _decoded(sample_id: str, digest: str, fingerprint: str = "f" * 64) -> V4DecodeResult:
    return V4DecodeResult(
        sample_id=sample_id,
        raw_relative_path=f"raw/{sample_id}.wav",
        raw_sha256=("a" if sample_id == "first" else "b") * 64,
        decoded_relative_path=f"processed/{sample_id}.wav",
        decoded_audio_sha256=digest,
        decoded_size_bytes=100,
        duration_s=4.0,
        peak=0.5,
        rms_dbfs=-20.0,
        clipped_fraction=0.0,
        dc_offset=0.0,
        speech_seconds=3.5,
        speech_segment_count=1,
        audio_fingerprint_v1=fingerprint,
        preparation_status="ready",
        rejection_reason="",
    )


def test_decoded_gate_accounts_exact_history_and_pool_duplicates() -> None:
    first = V4DecodedCandidate(
        1, "ru", "bonafide", _decoded("first", "1" * 64, "0" * 64)
    )
    duplicate = V4DecodedCandidate(
        2, "ru", "bonafide", _decoded("second", "1" * 64, "0" * 64)
    )
    historical = V4DecodedCandidate(3, "ru", "bonafide", _decoded("history", "2" * 64))

    decisions = decide_v4_decoded_audio_eligibility(
        (first, duplicate, historical),
        {"2" * 64: ("old-manifest:row",)},
        (),
    )

    by_id = {item.candidate.result.sample_id: item for item in decisions}
    assert by_id["first"].eligibility_status == "eligible"
    assert by_id["second"].exact_duplicate_of_candidate_id == "first"
    assert by_id["history"].rejection_reason == "historical_exact_decoded_audio"
