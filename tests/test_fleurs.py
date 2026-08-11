from __future__ import annotations

import hashlib
import io
import tarfile
import wave
from pathlib import Path

import pytest

from kds.data.fleurs import (
    FleursArtifactSpec,
    FleursExtractedAsset,
    FleursIngestionError,
    FleursLocaleSpec,
    extract_fleurs_audio_slice,
    fleurs_manifest_rows,
    inspect_extracted_fleurs_audio,
    inspect_fleurs_release,
    select_fleurs_records,
)
from kds.data.manifest import validate_manifest


def _wav_bytes(value: int = 0) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(value.to_bytes(2, "little", signed=True) * 160)
    return buffer.getvalue()


def _blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()  # noqa: S324


def _write_archive(path: Path, split: str, files: dict[str, bytes]) -> bytes:
    with tarfile.open(path, "w:gz") as archive:
        directory = tarfile.TarInfo(f"{split}/")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for filename, payload in files.items():
            member = tarfile.TarInfo(f"{split}/{filename}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return path.read_bytes()


def _write_tsv(path: Path, rows: list[tuple[str, str, str]]) -> bytes:
    payload = "".join(
        f'{prompt}\t{filename}\t"{transcript}.\t{transcript}\tт е к с т\t160\tMALE\n'
        for prompt, filename, transcript in rows
    ).encode("utf-8")
    path.write_bytes(payload)
    return payload


def _small_release(root: Path) -> FleursLocaleSpec:
    artifact_specs: list[FleursArtifactSpec] = []
    for split in ("train", "dev", "test"):
        directory = root / "data" / "ru_ru"
        audio_dir = directory / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        filename_by_split = {"train": "1001.wav", "dev": "2001.wav", "test": "3001.wav"}
        rows = [("1", filename_by_split[split], "текст")]
        if split == "test":
            rows.append(("1", "3002.wav", "текст"))
            rows.append(("2", "3003.wav", "другой текст"))
        tsv = directory / f"{split}.tsv"
        tsv_payload = _write_tsv(tsv, rows)
        archive = audio_dir / f"{split}.tar.gz"
        archive_payload = _write_archive(
            archive,
            split,
            {
                filename: _wav_bytes(index + 1)
                for index, (_prompt, filename, _text) in enumerate(rows)
            },
        )
        artifact_specs.extend(
            (
                FleursArtifactSpec(
                    relative_path=f"data/ru_ru/{split}.tsv",
                    expected_size_bytes=len(tsv_payload),
                    git_blob_sha1=_blob_sha1(tsv_payload),
                ),
                FleursArtifactSpec(
                    relative_path=f"data/ru_ru/audio/{split}.tar.gz",
                    expected_size_bytes=len(archive_payload),
                    lfs_sha256=hashlib.sha256(archive_payload).hexdigest(),
                ),
            )
        )
    return FleursLocaleSpec(
        locale="ru_ru",
        language="ru",
        source_id="google_fleurs_ru_v1",
        artifacts=tuple(artifact_specs),
    )


def test_fleurs_release_audit_selection_extraction_and_manifest(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    spec = _small_release(release)

    report, records_by_split = inspect_fleurs_release(release, "ru_ru", spec=spec)
    assert report.source_splits == {"train": 1, "dev": 1, "test": 3}
    assert report.unique_text_groups["test"] == 2
    selected = select_fleurs_records(records_by_split["test"], limit=2, seed="seed")
    assert len(selected) == 2
    assert len({record.text_hash for record in selected}) == 2

    output_parent = tmp_path / "output"
    output_parent.mkdir()
    extracted = extract_fleurs_audio_slice(
        release, "ru_ru", "test", selected, output_parent / "slice"
    )
    assets = {}
    for filename, path in extracted.items():
        duration_s, original_sr, codec = inspect_extracted_fleurs_audio(path)
        assets[filename] = FleursExtractedAsset(
            filename=filename,
            relative_path=f"raw/google_fleurs_ru_v1/slices/test/{filename}",
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            duration_s=duration_s,
            original_sr=original_sr,
            codec=codec,
        )
    rows = fleurs_manifest_rows(
        selected,
        assets,
        manifest_split="test",
        created_at="2026-08-10T00:00:00Z",
    )
    validate_manifest(rows)
    assert {row.speaker_pseudo_id for row in rows} == {"google_fleurs_ru_v1:unknown"}
    assert {row.parent_group_id for row in rows} == {
        "google_fleurs_ru_v1:prompt:1",
        "google_fleurs_ru_v1:prompt:2",
    }


def test_fleurs_release_rejects_tar_member_not_present_in_tsv(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    spec = _small_release(release)
    archive = release / "data" / "ru_ru" / "audio" / "dev.tar.gz"
    payload = _write_archive(archive, "dev", {"2001.wav": _wav_bytes(), "2002.wav": _wav_bytes()})
    artifact_specs = tuple(
        FleursArtifactSpec(
            relative_path=artifact.relative_path,
            expected_size_bytes=len(payload),
            lfs_sha256=hashlib.sha256(payload).hexdigest(),
        )
        if artifact.relative_path == "data/ru_ru/audio/dev.tar.gz"
        else artifact
        for artifact in spec.artifacts
    )
    changed_spec = FleursLocaleSpec(
        locale="ru_ru", language="ru", source_id="google_fleurs_ru_v1", artifacts=artifact_specs
    )

    with pytest.raises(FleursIngestionError, match="membership mismatch"):
        inspect_fleurs_release(release, "ru_ru", spec=changed_spec)


def test_fleurs_selection_refuses_insufficient_text_groups(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    spec = _small_release(release)
    _report, records_by_split = inspect_fleurs_release(release, "ru_ru", spec=spec)

    with pytest.raises(FleursIngestionError, match="eligible text groups"):
        select_fleurs_records(
            records_by_split["test"],
            limit=2,
            seed="seed",
            excluded_text_hashes={record.text_hash for record in records_by_split["test"]},
        )
