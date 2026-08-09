from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

import kds.data.common_voice as common_voice
from kds.data.manifest import validate_manifest
from kds.data.split import GroupSplitter, SplitConfig


def _metadata_content(split: str) -> bytes:
    header = "client_id\tpath\tsentence_id\tsentence\tlocale\n"
    if split != "train":
        return header.encode()
    return (
        header
        + "hashed-client\tcommon_voice_ru_1.mp3\tsentence-1\tПроверочная фраза\tru\n"
    ).encode()


def _write_small_common_voice_archive(archive: Path) -> None:
    with tarfile.open(archive, mode="w:gz") as tar:
        for name in (
            common_voice.COMMON_VOICE_RU_V24_ARCHIVE_ROOT,
            common_voice.COMMON_VOICE_RU_V24_DIRECTORY,
            common_voice.COMMON_VOICE_RU_V24_CLIPS_DIRECTORY,
        ):
            directory = tarfile.TarInfo(name)
            directory.type = tarfile.DIRTYPE
            tar.addfile(directory)
        for filename in common_voice.COMMON_VOICE_RU_V24_METADATA_FILENAMES:
            payload = _metadata_content(filename.removesuffix(".tsv"))
            info = tarfile.TarInfo(f"{common_voice.COMMON_VOICE_RU_V24_DIRECTORY}/{filename}")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        audio_payload = b"minimal-mp3-bytes"
        audio = tarfile.TarInfo(
            f"{common_voice.COMMON_VOICE_RU_V24_CLIPS_DIRECTORY}/common_voice_ru_1.mp3"
        )
        audio.size = len(audio_payload)
        tar.addfile(audio, io.BytesIO(audio_payload))


def test_common_voice_archive_metadata_and_slice_are_validated_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / common_voice.COMMON_VOICE_RU_V24_ARCHIVE_NAME
    _write_small_common_voice_archive(archive)
    monkeypatch.setattr(
        common_voice, "COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SIZE_BYTES", archive.stat().st_size
    )
    output_parent = tmp_path / "output"
    output_parent.mkdir()

    report = common_voice.inspect_common_voice_archive(archive)
    records = common_voice.load_common_voice_metadata_from_archive(archive, ["train"])
    extracted = common_voice.extract_common_voice_audio_slice(
        archive, ["common_voice_ru_1.mp3"], output_parent / "slice"
    )

    assert report.audio_files == 1
    assert report.metadata_files == len(common_voice.COMMON_VOICE_RU_V24_METADATA_FILENAMES)
    assert records[0].client_id == "hashed-client"
    assert extracted["common_voice_ru_1.mp3"].read_bytes() == b"minimal-mp3-bytes"


def test_common_voice_manifest_uses_client_id_only_as_pseudonymous_group() -> None:
    record = common_voice.CommonVoiceRecord(
        clip_name="common_voice_ru_1.mp3",
        split="train",
        client_id="hashed-client",
        sentence_id="sentence-1",
        sentence="Проверочная фраза",
    )
    asset = common_voice.ExtractedCommonVoiceAsset(
        clip_name=record.clip_name,
        relative_path="raw/common_voice/clips/common_voice_ru_1.mp3",
        sha256=hashlib.sha256(b"audio").hexdigest(),
        duration_s=3.0,
        original_sr=48_000,
    )

    rows = common_voice.common_voice_manifest_rows(
        [record], {record.clip_name: asset}, created_at="2026-08-09T00:00:00Z"
    )
    assigned = GroupSplitter(SplitConfig(seed="fixed")).assign_rows(rows)
    validate_manifest(assigned)

    assert rows[0].speaker_pseudo_id.endswith("hashed-client")
    assert rows[0].parent_group_id == rows[0].speaker_pseudo_id
    assert rows[0].code_switch == "unknown"


def test_common_voice_metadata_rejects_path_traversal() -> None:
    content = (
        "client_id\tpath\tsentence_id\tsentence\tlocale\n"
        "hashed-client\t../outside.mp3\tsentence-1\tПроверочная фраза\tru\n"
    )

    with pytest.raises(common_voice.CommonVoiceIngestionError, match="Invalid Common Voice clip"):
        common_voice._parse_metadata_tsv(content, "train", "train.tsv")
