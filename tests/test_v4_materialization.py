from __future__ import annotations

from pathlib import Path

import pytest

from kds.data.v4_materialization import (
    V4MaterializationError,
    V4RawAsset,
    V4SourceCandidate,
    decide_v4_raw_exact_eligibility,
    load_v4_source_candidates,
)


def _candidate(
    candidate_id: str,
    *,
    language: str,
    label: str,
    rank: int = 1,
) -> V4SourceCandidate:
    source_id = "ruasd_ru_v1_full" if language == "ru" else "ksc2_v1"
    return V4SourceCandidate(
        selection_rank=rank,
        target_state="target",
        language=language,
        label=label,
        candidate_id=candidate_id,
        pair_id=f"pair:{candidate_id}",
        source_id=source_id,
        source_lineage_id=f"{source_id}:train-only",
        source_component="component",
        archive_audio_member=f"{candidate_id}.wav",
        text_hash=("1" if label == "bonafide" else "2") * 64,
        canonical_text_hash=("1" if label == "bonafide" else "2") * 64,
        parent_group_id=f"group:{candidate_id}",
    )


def _asset(candidate: V4SourceCandidate, digest: str) -> V4RawAsset:
    return V4RawAsset(
        candidate=candidate,
        raw_relative_path=f"raw/v4/{candidate.candidate_id}.wav",
        raw_audio_sha256=digest,
        raw_size_bytes=10,
        duration_s=3.0,
        original_sr=16_000,
        codec="wav",
    )


def test_repository_canonical_v4_source_packet_loads() -> None:
    rows = load_v4_source_candidates(
        Path("data/manifests/v4/xlsr_sls_model_v4_train_candidates_v2.csv"),
        Path(
            "docs/artifacts/v4/"
            "xlsr_sls_model_v4_train_candidate_selection_governance_v1.json"
        ),
    )

    assert len(rows) == 21_600
    assert sum(row.source_id == "ruasd_ru_v1_full" for row in rows) == 14_400
    assert sum(row.source_id == "ksc2_v1" for row in rows) == 7_200


def test_raw_exact_gate_accounts_history_and_within_pool_duplicates() -> None:
    ru_bf = _candidate("ru-bf", language="ru", label="bonafide")
    ru_bf_duplicate = _candidate("ru-bf-duplicate", language="ru", label="bonafide", rank=2)
    ru_spoof_history = _candidate("ru-spoof-history", language="ru", label="spoof")
    ru_spoof = _candidate("ru-spoof", language="ru", label="spoof", rank=2)
    kk_bf = _candidate("kk-bf", language="kk", label="bonafide")
    decisions = decide_v4_raw_exact_eligibility(
        (
            _asset(ru_bf, "a" * 64),
            _asset(ru_bf_duplicate, "a" * 64),
            _asset(ru_spoof_history, "b" * 64),
            _asset(ru_spoof, "c" * 64),
            _asset(kk_bf, "d" * 64),
        ),
        frozenset({"b" * 64}),
        target_per_cell=1,
    )

    by_id = {decision.asset.candidate.candidate_id: decision for decision in decisions}
    assert by_id["ru-bf-duplicate"].duplicate_of_candidate_id == "ru-bf"
    assert by_id["ru-spoof-history"].rejection_reason == "historical_exact_raw_audio_hash"
    assert sum(item.eligibility_status == "eligible_for_decode_qa" for item in decisions) == 3


def test_raw_exact_gate_rejects_conflicting_labels_for_identical_audio() -> None:
    with pytest.raises(V4MaterializationError, match="conflicting language/label"):
        decide_v4_raw_exact_eligibility(
            (
                _asset(_candidate("real", language="ru", label="bonafide"), "a" * 64),
                _asset(_candidate("fake", language="ru", label="spoof"), "a" * 64),
                _asset(_candidate("kk", language="kk", label="bonafide"), "b" * 64),
            ),
            frozenset(),
            target_per_cell=1,
        )
