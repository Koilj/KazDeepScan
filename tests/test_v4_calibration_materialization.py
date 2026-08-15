from __future__ import annotations

from kds.data.v4_audio_gate import V4DecodedCandidate, V4DecodedDecision, V4DecodeResult
from kds.data.v4_calibration_materialization import _decoded_relative_path, _raw_exact_matches


def _decision() -> V4DecodedDecision:
    raw_sha = "a" * 64
    decoded_sha = "b" * 64
    result = V4DecodeResult(
        sample_id="source:one",
        raw_relative_path="raw/example.wav",
        raw_sha256=raw_sha,
        decoded_relative_path="processed/example.wav",
        decoded_audio_sha256=decoded_sha,
        decoded_size_bytes=100,
        duration_s=3.0,
        peak=0.5,
        rms_dbfs=-20.0,
        clipped_fraction=0.0,
        dc_offset=0.0,
        speech_seconds=2.5,
        speech_segment_count=1,
        audio_fingerprint_v1="0" * 64,
        preparation_status="ready",
        rejection_reason="",
    )
    return V4DecodedDecision(
        candidate=V4DecodedCandidate(
            selection_rank=1,
            language="ru",
            label="bonafide",
            result=result,
        ),
        eligibility_status="eligible",
        rejection_reason="",
        exact_duplicate_of_candidate_id="",
        historical_exact_matches=(),
        historical_near_matches=(),
        within_pool_near_matches=(),
    )


def test_raw_history_collision_rejects_an_otherwise_eligible_decode() -> None:
    decision = _decision()

    rejected = _raw_exact_matches(decision, {"a" * 64: ("history:source",)})

    assert rejected.eligibility_status == "rejected"
    assert rejected.rejection_reason == "historical_exact_raw_audio"


def test_calibration_decode_destination_is_content_addressed_and_namespaced() -> None:
    assert _decoded_relative_path("c" * 64, "spoof") == (
        "processed/v4/xlsr_sls_model_v4_calibration_materialization_v1/"
        f"spoof/cc/{'c' * 64}.wav"
    )
