from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from kds.data.kazakhtts import KazakhTtsRuntime, load_kazakhtts_runtime
from kds.data.kazakhtts_candidate import (
    KAZAKHTTS_SOURCE_LICENSE,
    KazakhTtsCandidateError,
    build_kazakhtts_pairs,
    kazakhtts_spoof_row,
)
from kds.data.manifest import load_manifest, validate_manifest
from kds.data.research_tts import ResearchTtsModel, load_research_tts_model_lock


def _route() -> tuple[ResearchTtsModel, KazakhTtsRuntime]:
    lock = load_research_tts_model_lock(
        Path("configs/research/kazakhtts_tacotron2_pwg_v1_models.json")
    )
    model = lock.models[0]
    return model, load_kazakhtts_runtime(model)


def test_spoof_row_preserves_frozen_base_text_and_forbids_reference_audio() -> None:
    base = load_manifest(Path("data/manifests/fresh_suite_stage_c_base_ready_v1.csv"))[0]
    model, runtime = _route()

    row = kazakhtts_spoof_row(
        base_row=base,
        model=model,
        runtime=runtime,
        relative_path="raw/fresh_suite_v1_kazakhtts_tacotron2_pwg/test.wav",
        sha256="a" * 64,
        duration_s=3.0,
        created_at="2026-08-12T00:00:00Z",
        device="cuda",
    )

    validate_manifest([row])
    assert row.text_id == base.text_id
    assert row.text_hash == base.text_hash
    assert row.source_license == KAZAKHTTS_SOURCE_LICENSE
    assert "no_reference_audio" in row.clone_consent_id
    assert row.language == base.language


def test_pair_builder_accounts_rejections_without_backfill() -> None:
    base = load_manifest(Path("data/manifests/fresh_suite_stage_c_base_ready_v1.csv"))[:2]
    model, runtime = _route()
    raw = [
        kazakhtts_spoof_row(
            base_row=row,
            model=model,
            runtime=runtime,
            relative_path=f"raw/fresh/{index}.wav",
            sha256=str(index + 1) * 64,
            duration_s=3.0,
            created_at="2026-08-12T00:00:00Z",
            device="cuda",
        )
        for index, row in enumerate(base)
    ]

    pairs = build_kazakhtts_pairs(
        base_rows=base,
        raw_spoof_rows=raw,
        ready_spoof_rows=[raw[0]],
        text_rejected_base_ids=set(),
        rejected_spoof_ids={raw[1].sample_id},
    )

    assert len(pairs) == 2
    assert {row.label for row in pairs} == {"bonafide", "spoof"}
    with pytest.raises(KazakhTtsCandidateError, match="minus accounted"):
        build_kazakhtts_pairs(
            base_rows=base,
            raw_spoof_rows=raw,
            ready_spoof_rows=[replace(raw[0], text_id="changed")],
            text_rejected_base_ids=set(),
            rejected_spoof_ids={raw[1].sample_id},
        )


def test_normalized_spoof_identity_and_provenance_are_explicit() -> None:
    base = load_manifest(Path("data/manifests/fresh_suite_stage_c_base_ready_v1.csv"))[0]
    model, runtime = _route()
    row = kazakhtts_spoof_row(
        base_row=base,
        model=model,
        runtime=runtime,
        relative_path="raw/fresh/normalized.wav",
        sha256="b" * 64,
        duration_s=3.0,
        created_at="2026-08-12T00:00:00Z",
        device="cuda",
        normalizer_id="stage_c_kazakhtts_text_v1",
        synthesis_text_sha256="c" * 64,
    )

    assert "text_normalizer=stage_c_kazakhtts_text_v1" in row.augmentation_chain
    assert "normalizer=stage_c_kazakhtts_text_v1" in row.rights_basis
