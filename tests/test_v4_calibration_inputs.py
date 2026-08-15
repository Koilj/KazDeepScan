from __future__ import annotations

import hashlib

import pytest

from kds.data.manifest import ManifestRow
from kds.data.v4_calibration import (
    V4CalibrationInputError,
    select_fresh_voxforge_metadata_candidates,
)
from kds.data.voxforge import VOXFORGE_RU_SOURCE_ID, VoxForgeRuRecord
from kds.eval.voxforge_metadata_screen import voxforge_metadata_identity


def _record(submission: str, contributor: str, prompt: str, text: str) -> VoxForgeRuRecord:
    return VoxForgeRuRecord(
        submission_id=submission,
        contributor_alias=contributor,
        prompt_id=prompt,
        prompt_text=text,
        original_prompt_text=text,
    )


def _manifest_row(
    *,
    sample_id: str,
    parent_group_id: str,
    text_hash: str,
    source_name: str,
    generator_family: str = "",
) -> ManifestRow:
    return ManifestRow(
        sample_id=sample_id,
        relative_path=f"processed/{hashlib.sha256(sample_id.encode()).hexdigest()}.wav",
        sha256=hashlib.sha256(f"asset:{sample_id}".encode()).hexdigest(),
        split="test",
        label="bonafide",
        language="ru",
        code_switch="false",
        parent_group_id=parent_group_id,
        source_name=source_name,
        source_license="test",
        rights_basis="test",
        speaker_pseudo_id=parent_group_id,
        text_id=f"text:{text_hash[:12]}",
        text_hash=text_hash,
        duration_s=1.0,
        generator_family=generator_family,
        generator_name="",
        generator_version="",
        voice_id="",
        clone_consent_id="",
        device="test",
        capture_route="test",
        original_sr=16000,
        codec="wav",
        augmentation_chain="",
        augmentation_seed="",
        created_at="2026-08-15T00:00:00Z",
    )


def test_v4_calibration_selection_requires_fresh_source_groups_and_espeak_texts() -> None:
    old = _record("old", "legacy", "p1", "one")
    fresh_p1_group_1 = _record("fresh-1", "group-1", "p1", "one")
    fresh_p1_group_2 = _record("fresh-2", "group-2", "p1", "one")
    fresh_p2_group_2 = _record("fresh-3", "group-2", "p2", "two")
    historical_espeak_text = _record("fresh-4", "group-3", "p3", "three")
    old_identity = voxforge_metadata_identity(old)
    old_row = _manifest_row(
        sample_id=old_identity.sample_id,
        parent_group_id=old_identity.parent_group_id,
        text_hash=old_identity.prompt_text_hash,
        source_name=VOXFORGE_RU_SOURCE_ID,
    )
    blocked_espeak_identity = voxforge_metadata_identity(historical_espeak_text)
    blocked_espeak_row = _manifest_row(
        sample_id="historical-espeak",
        parent_group_id="historical-espeak-group",
        text_hash=blocked_espeak_identity.prompt_text_hash,
        source_name="fleurs_ru_v1_espeakng",
        generator_family="formant_rule_based_tts",
    )

    selected = select_fresh_voxforge_metadata_candidates(
        records=(
            old,
            fresh_p1_group_1,
            fresh_p1_group_2,
            fresh_p2_group_2,
            historical_espeak_text,
        ),
        historical_rows=(old_row, blocked_espeak_row),
        target_text_groups=2,
        selection_seed="v4-calibration-test",
    )

    assert len(selected) == 2
    assert {row.prompt_text_hash for row in selected} == {
        voxforge_metadata_identity(fresh_p1_group_1).prompt_text_hash,
        voxforge_metadata_identity(fresh_p2_group_2).prompt_text_hash,
    }
    assert old_identity.sample_id not in {row.sample_id for row in selected}
    assert old_identity.parent_group_id not in {row.parent_group_id for row in selected}
    assert len({row.parent_group_id for row in selected}) == 2


def test_v4_calibration_selection_fails_when_target_exceeds_fresh_text_capacity() -> None:
    record = _record("fresh", "group", "p1", "one")

    with pytest.raises(V4CalibrationInputError, match="exceeds fresh VoxForge text capacity"):
        select_fresh_voxforge_metadata_candidates(
            records=(record,),
            historical_rows=(),
            target_text_groups=2,
            selection_seed="v4-calibration-test",
        )
