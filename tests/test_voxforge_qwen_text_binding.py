from __future__ import annotations

import hashlib

import pytest

from kds.data.manifest import ManifestRow
from kds.data.voxforge import VoxForgeRuRecord
from kds.eval.voxforge_metadata_screen import voxforge_metadata_identity
from scripts.bind_voxforge_ru_mdc_qwen3_tts_customvoice_pre_qa_text import (
    VoxForgeQwenTextBindingError,
    bind_literal_texts,
)


def _record(text: str = "Точный исходный текст") -> VoxForgeRuRecord:
    return VoxForgeRuRecord(
        submission_id="tester-20260512-abc",
        contributor_alias="tester",
        prompt_id="ru_0001",
        prompt_text=text,
        original_prompt_text=f"{text}.",
    )


def _row(record: VoxForgeRuRecord) -> ManifestRow:
    identity = voxforge_metadata_identity(record)
    return ManifestRow(
        sample_id=identity.sample_id,
        relative_path="processed/example.wav",
        sha256=hashlib.sha256(b"audio").hexdigest(),
        split="test",
        label="bonafide",
        language="ru",
        code_switch="unknown",
        parent_group_id=identity.parent_group_id,
        source_name="voxforge_ru_mdc_2026_05",
        source_license="GPL-3.0-or-later",
        rights_basis="test rights",
        speaker_pseudo_id=identity.speaker_pseudo_id,
        text_id=record.prompt_id,
        text_hash=identity.prompt_text_hash,
        duration_s=1.0,
        generator_family="",
        generator_name="",
        generator_version="",
        voice_id="",
        clone_consent_id="",
        device="unknown",
        capture_route="voxforge_submission_read_speech",
        original_sr=48_000,
        codec="wav",
        augmentation_chain="",
        augmentation_seed="",
        created_at="2026-08-13T18:13:54Z",
    )


def _selection(record: VoxForgeRuRecord) -> dict[str, dict[str, object]]:
    identity = voxforge_metadata_identity(record)
    return {
        identity.sample_id: {
            "selection_rank": 1,
            "prompt_id": record.prompt_id,
            "prompt_text_hash": identity.prompt_text_hash,
            "original_prompt_text_hash": identity.original_prompt_text_hash,
        }
    }


def test_bind_literal_texts_preserves_hash_only_text_binding_and_seed() -> None:
    record = _record()
    row = _row(record)

    bound = bind_literal_texts((row,), (record,), _selection(record))

    assert len(bound) == 1
    assert bound[0]["literal_text_sha256"] == row.text_hash
    assert bound[0]["literal_text_utf8_bytes"] == len(record.prompt_text.encode("utf-8"))
    assert bound[0]["rng_seed"] == int.from_bytes(
        hashlib.sha256(record.prompt_text.encode("utf-8")).digest()[:4], "big"
    )
    assert "literal_text" not in bound[0]


def test_bind_literal_texts_rejects_qwen_unsafe_long_text() -> None:
    record = _record("а" * 4097)

    with pytest.raises(VoxForgeQwenTextBindingError, match="exact Qwen-safe"):
        bind_literal_texts((_row(record),), (record,), _selection(record))
