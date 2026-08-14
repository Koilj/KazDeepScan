from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

from kds.data.v4_audio_gate import V4DecodedCandidate, V4DecodedDecision, V4DecodeResult


def _module() -> dict[str, Any]:
    return runpy.run_path("scripts/run_v4_kk_spoof_audio_gate.py")


def _decision(sample_id: str, rank: int) -> V4DecodedDecision:
    result = V4DecodeResult(
        sample_id=sample_id,
        raw_relative_path=f"raw/{sample_id}.wav",
        raw_sha256="a" * 64,
        decoded_relative_path=f"processed/{sample_id}.wav",
        decoded_audio_sha256="b" * 64,
        decoded_size_bytes=1,
        duration_s=3.0,
        peak=0.1,
        rms_dbfs=-20.0,
        clipped_fraction=0.0,
        dc_offset=0.0,
        speech_seconds=3.0,
        speech_segment_count=1,
        audio_fingerprint_v1="c" * 64,
        preparation_status="ready",
        rejection_reason="",
    )
    return V4DecodedDecision(
        candidate=V4DecodedCandidate(rank, "kk", "spoof", result),
        eligibility_status="eligible",
        rejection_reason="",
        exact_duplicate_of_candidate_id="",
        historical_exact_matches=(),
        historical_near_matches=(),
        within_pool_near_matches=(),
    )


def test_v4_kk_spoof_audio_gate_plan_binds_all_completed_routes() -> None:
    module = _module()

    plan = module["load_plan"](
        Path("configs/research/v4/xlsr_sls_model_v4_kk_spoof_audio_gate_v1.json"),
        Path(".").resolve(),
    )

    assert plan.target_per_route == 1_250
    assert {route.route_id for route in plan.routes} == {
        "kk-piper-issai-high-v1",
        "kk-mms-kaz-v1",
        "kk-kazemotts-v1",
        "kk-sparktts-v1",
    }


def test_v4_kk_spoof_audio_gate_uses_target_before_reserve() -> None:
    module = _module()
    context_type = module["CandidateContext"]
    contexts = {}
    decisions = []
    for route_index, route_id in enumerate(("a", "b", "c", "d"), start=1):
        for rank, state in ((3, "reserve"), (2, "target"), (1, "target")):
            sample_id = f"{route_id}-{rank}"
            contexts[sample_id] = context_type(
                selection_rank=rank,
                target_state=state,
                candidate_id=sample_id,
                pair_id=sample_id,
                route_id=route_id,
                generator_family="test",
                model_id=f"model-{route_index}",
                assigned_voice_id="voice",
                actual_voice_id="voice",
                raw_relative_path=f"raw/{sample_id}.wav",
                raw_audio_sha256="a" * 64,
            )
            decisions.append(_decision(sample_id, rank))

    selected = module["select_frozen_ids"](decisions, contexts, 2)

    assert selected == ("a-1", "a-2", "b-1", "b-2", "c-1", "c-2", "d-1", "d-2")
