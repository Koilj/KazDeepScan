from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

import kds.eval.denis_source_screen as denis_screen
from kds.data.denis import DenisRecord
from kds.data.manifest import ManifestRow


def _denis_record(*, sample_id: str, text_hash: str) -> DenisRecord:
    return DenisRecord(
        sample_id=sample_id,
        member_stem=f"ru-RU/fixture/{sample_id}",
        category="fixture",
        literal_text_sha256=text_hash,
        whitespace_canonical_text_sha256=text_hash,
        nfkc_whitespace_canonical_text_sha256=text_hash,
        audio_sha256="a" * 64,
        audio_size_bytes=100,
        decoded_frames=48_000,
        sample_rate_hz=48_000,
        channels=2,
        decoded_container="OGG",
        decoded_subtype="OPUS",
    )


def _prior_row(*, text_hash: str) -> ManifestRow:
    return ManifestRow(
        sample_id="prior:denis",
        relative_path="processed/prior.wav",
        sha256="b" * 64,
        split="train",
        label="spoof",
        language="ru",
        code_switch="false",
        parent_group_id="prior:group",
        source_name="prior",
        source_license="CC-BY-NC-SA-4.0",
        rights_basis="fixture",
        speaker_pseudo_id="prior:speaker",
        text_id="prior:text",
        text_hash=text_hash,
        duration_s=1.0,
        generator_family="tts",
        generator_name="piperTTS",
        generator_version="ru_RU-denis-medium",
        voice_id="prior:unknown",
        clone_consent_id="",
        device="unknown",
        capture_route="fixture",
        original_sr=22_050,
        codec="wav",
        augmentation_chain="",
        augmentation_seed="",
        created_at="2026-08-14T00:00:00Z",
    )


def test_source_screen_taints_single_speaker_source_on_any_direct_text_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_root = tmp_path / "configs"
    manifest_root = tmp_path / "manifests"
    config_root.mkdir()
    manifest_root.mkdir()
    records = (
        _denis_record(sample_id="denis:one", text_hash="c" * 64),
        _denis_record(sample_id="denis:two", text_hash="d" * 64),
    )
    prior = _prior_row(text_hash="c" * 64)
    monkeypatch.setattr(
        denis_screen,
        "configured_role_scope",
        lambda _project_root, _config_root: ([prior], [], []),
    )
    monkeypatch.setattr(
        denis_screen,
        "_load_inventory",
        lambda **_kwargs: ([prior], [], []),
    )

    receipt = denis_screen.screen_denis_source_records(
        records=records,
        project_root=tmp_path,
        config_root=config_root,
        manifest_root=manifest_root,
        created_at="2026-08-14T00:00:00+05:00",
        source_audit_receipt={"path": "receipt.json", "sha256": "e" * 64},
    )

    strict = cast(dict[str, object], receipt["strict_single_speaker_group_exclusion"])
    claims = cast(dict[str, object], receipt["claims"])
    lineage = cast(dict[str, object], receipt["historical_likely_speaker_lineage"])
    assert strict["direct_identity_overlap_found"] is True
    assert strict["surviving_records"] == 0
    assert claims["new_direct_human_source"] is False
    assert claims["speaker_independent"] is False
    configured = cast(dict[str, object], lineage["configured_scope"])
    assert configured["unique_sample_ids"] == 1
