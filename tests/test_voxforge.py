from __future__ import annotations

import hashlib
import io
import tarfile
import wave
from pathlib import Path

import pytest

import kds.data.voxforge as voxforge
import kds.eval.voxforge_metadata_screen as metadata_screen
from kds.data.manifest import ManifestRow


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\0\0" * 160)
    return buffer.getvalue()


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def _add_directory(archive: tarfile.TarFile, name: str) -> None:
    member = tarfile.TarInfo(name)
    member.type = tarfile.DIRTYPE
    archive.addfile(member)


def _fixture_archive(tmp_path: Path, *, include_prompt: bool = True) -> Path:
    path = tmp_path / "voxforge-russian-9a8495d3.tar.gz"
    root = "voxforge-ru"
    submission = "tester-20260512-abc"
    gpl = "GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n"
    with tarfile.open(path, "w:gz") as archive:
        _add_directory(archive, root)
        _add_directory(archive, f"{root}/{submission}")
        _add_directory(archive, f"{root}/{submission}/wav")
        _add_bytes(archive, f"{root}/{submission}/wav/ru_0001.wav", _wav_bytes())
        _add_directory(archive, f"{root}/{submission}/etc")
        _add_bytes(archive, f"{root}/{submission}/etc/GPL_license.txt", gpl.encode())
        _add_bytes(archive, f"{root}/{submission}/etc/README", b"User Name:Tester\n")
        _add_bytes(
            archive,
            f"{root}/{submission}/etc/PROMPTS",
            (f"{submission}/mfc/ru_0001 Test prompt\n" if include_prompt else "").encode(),
        )
        _add_bytes(
            archive,
            f"{root}/{submission}/etc/prompts-original",
            b"ru_0001 Test prompt.\n",
        )
        _add_bytes(archive, f"{root}/{submission}/LICENSE", gpl.encode())
    return path


def _pin_fixture(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(voxforge, "VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES", path.stat().st_size)
    monkeypatch.setattr(
        voxforge,
        "VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def test_audit_voxforge_ru_archive_validates_transcript_bound_wav_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _fixture_archive(tmp_path)
    _pin_fixture(monkeypatch, path)

    audit = voxforge.audit_voxforge_ru_archive(path)

    assert audit.intake_status == "accepted_source_level_only"
    assert audit.submissions == 1
    assert audit.wav_files == audit.prompt_rows == audit.original_prompt_rows == 1
    assert audit.source_provided_contributor_groups == 1
    assert audit.sample_rates_hz == {"16000": 1}
    assert audit.extraction_performed is False


def test_audit_voxforge_ru_archive_rejects_wav_without_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _fixture_archive(tmp_path, include_prompt=False)
    _pin_fixture(monkeypatch, path)

    with pytest.raises(voxforge.VoxForgeRuAuditError, match="no prompt rows"):
        voxforge.audit_voxforge_ru_archive(path)


def test_load_voxforge_ru_metadata_reads_bound_transcripts_without_wav_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _fixture_archive(tmp_path)
    _pin_fixture(monkeypatch, path)

    records = voxforge.load_voxforge_ru_metadata(path)

    assert records == (
        voxforge.VoxForgeRuRecord(
            submission_id="tester-20260512-abc",
            contributor_alias="tester",
            prompt_id="ru_0001",
            prompt_text="Test prompt",
            original_prompt_text="Test prompt.",
        ),
    )


def _prior_manifest_row(*, text_hash: str) -> ManifestRow:
    return ManifestRow(
        sample_id="prior:sample",
        relative_path="processed/prior.wav",
        sha256="a" * 64,
        split="test",
        label="bonafide",
        language="ru",
        code_switch="false",
        parent_group_id="prior:group",
        source_name="prior",
        source_license="CC-BY-4.0",
        rights_basis="fixture",
        speaker_pseudo_id="prior:speaker",
        text_id="prior:text",
        text_hash=text_hash,
        duration_s=1.0,
        generator_family="",
        generator_name="",
        generator_version="",
        voice_id="",
        clone_consent_id="not_applicable",
        device="cpu",
        capture_route="fixture",
        original_sr=16_000,
        codec="wav",
        augmentation_chain="",
        augmentation_seed="",
        created_at="2026-08-13T00:00:00Z",
    )


def test_metadata_screen_taints_every_record_in_contributor_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = (
        voxforge.VoxForgeRuRecord("submission-a", "alias-a", "one", "shared", "shared"),
        voxforge.VoxForgeRuRecord("submission-a", "alias-a", "two", "unique-a", "unique-a"),
        voxforge.VoxForgeRuRecord("submission-b", "alias-b", "three", "unique-b", "unique-b"),
    )
    prior_row = _prior_manifest_row(
        text_hash=hashlib.sha256(b"shared").hexdigest()
    )
    config_root = tmp_path / "configs"
    manifest_root = tmp_path / "manifests"
    config_root.mkdir()
    manifest_root.mkdir()
    monkeypatch.setattr(
        metadata_screen,
        "configured_role_scope",
        lambda _project_root, _config_root: ([prior_row], [], []),
    )
    monkeypatch.setattr(
        metadata_screen,
        "_load_inventory",
        lambda **_kwargs: ([prior_row], []),
    )

    screen = metadata_screen.screen_voxforge_ru_metadata(
        records=records,
        project_root=tmp_path,
        config_root=config_root,
        manifest_root=manifest_root,
        created_at="2026-08-13T00:00:00Z",
    )

    assert [identity.sample_id for identity in screen.surviving] == [
        "voxforge_ru_mdc_2026_05:submission-b:three"
    ]
    strict = screen.receipt["strict_group_exclusion"]
    overlaps = screen.receipt["direct_overlap_record_counts"]
    assert strict["tainted_contributor_groups"] == 1
    assert strict["excluded_records"] == 2
    assert overlaps["configured_roles"]["prompt_text_hash"] == 1
    assert overlaps["manifest_inventory"]["prompt_text_hash"] == 1
