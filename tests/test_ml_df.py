from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import py7zr
import pytest

import kds.data.ml_df as ml_df
from kds.data.manifest import validate_manifest


def _metadata_rows() -> list[str]:
    return [
        "dataset_IT/IT_1_2_000001_0_g.wav bonafide F train 1",
        "dataset_IT/IT_2_3_000002_0_1_VITS.wav VITS F train 1",
        "dataset_IT/IT_3_4_000003_0_1_ZMM-TTS.wav ZMM-TTS F train 1",
        "dataset_IT/IT_4_5_000004_0_1_LVC-VC.wav LVC-VC F train 1",
        "dataset_IT/IT_5_6_000005_0_1_DDDM-VC.wav DDDM-VC F train 1",
    ]


def _write_metadata_archive(path: Path) -> None:
    header = "wav_file tool gender group speaker\n"
    with zipfile.ZipFile(path, mode="w") as archive:
        for filename in ml_df.ML_DF_METADATA_FILENAMES:
            content = header
            if filename == ml_df.ML_DF_IT_METADATA_NAME:
                content += "\n".join(_metadata_rows()) + "\n"
            archive.writestr(filename, content)


def _write_audio_archive(path: Path, source_root: Path) -> None:
    for row in _metadata_rows():
        relative_path = row.split()[0]
        audio_path = source_root / relative_path
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(relative_path.encode())
    metadata_path = source_root / "metadata" / ml_df.ML_DF_IT_METADATA_NAME
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text("wav_file tool gender group speaker\n", encoding="utf-8")
    with py7zr.SevenZipFile(path, mode="w") as archive:
        archive.writeall(source_root / ml_df.ML_DF_IT_DIRECTORY, arcname=ml_df.ML_DF_IT_DIRECTORY)
        archive.writeall(source_root / "metadata", arcname="metadata")


def test_ml_df_metadata_archive_and_ood_slice_are_checked_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata_archive = tmp_path / ml_df.ML_DF_METADATA_ARCHIVE_NAME
    audio_archive = tmp_path / ml_df.ML_DF_IT_ARCHIVE_NAME
    source_root = tmp_path / "source"
    _write_metadata_archive(metadata_archive)
    _write_audio_archive(audio_archive, source_root)
    metadata_md5 = hashlib.md5(metadata_archive.read_bytes()).hexdigest()
    archive_md5 = hashlib.md5(audio_archive.read_bytes()).hexdigest()
    with py7zr.SevenZipFile(audio_archive, mode="r") as archive:
        archive_uncompressed_size = sum(
            info.uncompressed for info in archive.list() if info.is_file
        )
    monkeypatch.setattr(ml_df, "ML_DF_METADATA_EXPECTED_MD5", metadata_md5)
    monkeypatch.setattr(ml_df, "ML_DF_IT_ARCHIVE_EXPECTED_SIZE_BYTES", audio_archive.stat().st_size)
    monkeypatch.setattr(ml_df, "ML_DF_IT_ARCHIVE_EXPECTED_MD5", archive_md5)
    monkeypatch.setattr(
        ml_df, "ML_DF_IT_ARCHIVE_EXPECTED_UNCOMPRESSED_BYTES", archive_uncompressed_size
    )
    output_parent = tmp_path / "output"
    output_parent.mkdir()

    records = ml_df.load_ml_df_it_metadata(metadata_archive)
    report = ml_df.inspect_ml_df_archive(audio_archive, records)
    selected = ml_df.select_ml_df_ood_records(records, 1, 1, "seed")
    extracted = ml_df.extract_ml_df_audio_slice(audio_archive, selected, output_parent / "slice")

    assert report.audio_files == 5
    assert len(selected) == 5
    assert set(extracted) == {record.relative_path for record in selected}
    assert all(path.is_file() for path in extracted.values())


def test_ml_df_manifest_is_cross_lingual_ood_with_generator_provenance() -> None:
    records = [
        ml_df.MlDfRecord(
            relative_path=row.split()[0],
            tool=row.split()[1],
            gender=row.split()[2],
            source_group=row.split()[3],
            target_speaker_id=row.split()[4],
        )
        for row in _metadata_rows()
    ]
    assets = {
        record.relative_path: ml_df.ExtractedMlDfAsset(
            relative_path=record.relative_path,
            sha256=hashlib.sha256(record.relative_path.encode()).hexdigest(),
            duration_s=3.0,
            original_sr=16_000,
        )
        for record in records
    }

    rows = ml_df.ml_df_ood_manifest_rows(records, assets, created_at="2026-08-09T00:00:00Z")

    validate_manifest(rows, require_ood_generator=True)
    assert {row.language for row in rows} == {"other"}
    assert {row.generator_name for row in rows if row.label == "spoof"} == {
        "DDDM-VC",
        "LVC-VC",
        "VITS",
        "ZMM-TTS",
    }
