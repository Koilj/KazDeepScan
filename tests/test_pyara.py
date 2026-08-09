from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

import kds.data.pyara as pyara
from kds.data.manifest import validate_manifest
from kds.data.split import GroupSplitter, SplitConfig


def _write_small_pyara_archive(path: Path) -> None:
    rows = (
        ("Real/real_1.wav", "Shared sentence", "0", ""),
        ("Real/real_2.wav", "Second sentence", "0", ""),
        ("Fake/alg_1_1.wav", "Shared sentence", "1", "alg_1"),
        ("Fake/alg_2_1.wav", "Third sentence", "1", "alg_2"),
    )
    header = "path\tsentence\tage\tgender\tfake\talgorithm\tlength\n"
    metadata = header + "".join(
        f"{audio_path}\t{sentence}\tunknown\tunknown\t{fake}\t{algorithm}\t3.0\n"
        for audio_path, sentence, fake, algorithm in rows
    )
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr(f"{pyara.PYARA_ARCHIVE_ROOT}/final_dataset.tsv", metadata)
        for audio_path, _sentence, _fake, _algorithm in rows:
            archive.writestr(f"{pyara.PYARA_ARCHIVE_ROOT}/{audio_path}", audio_path.encode())


def test_pyara_archive_metadata_and_selected_wavs_are_checked_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / pyara.PYARA_ARCHIVE_NAME
    _write_small_pyara_archive(archive)
    monkeypatch.setattr(pyara, "PYARA_ARCHIVE_EXPECTED_SIZE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(
        pyara, "PYARA_ARCHIVE_EXPECTED_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(pyara, "PYARA_EXPECTED_AUDIO_FILES", 4)
    output_parent = tmp_path / "output"
    output_parent.mkdir()

    report, records = pyara.inspect_pyara_archive(archive)
    selected = pyara.select_pyara_records(
        records, real_limit=1, fake_limit_per_algorithm=1, seed="fixed"
    )
    extracted = pyara.extract_pyara_audio_slice(archive, selected, output_parent / "slice")

    assert report.audio_files == 4
    assert report.real_files == 2
    assert report.fake_files == 2
    assert len(selected) == 3
    assert all(path.is_file() for path in extracted.values())


def test_pyara_manifest_records_text_groups_but_never_claims_speaker_identity() -> None:
    records = [
        pyara.PyAraRecord("Real/real_1.wav", "bonafide", "Shared sentence", "", 3.0),
        pyara.PyAraRecord("Fake/alg_1_1.wav", "spoof", "Shared sentence", "alg_1", 3.0),
    ]
    assets = {
        record.relative_path: pyara.ExtractedPyAraAsset(
            relative_path=f"raw/pyara/{record.relative_path}",
            sha256=hashlib.sha256(record.relative_path.encode()).hexdigest(),
            duration_s=3.0,
            original_sr=16_000,
        )
        for record in records
    }

    source_rows = pyara.pyara_manifest_rows(records, assets, created_at="2026-08-09T00:00:00Z")
    rows = GroupSplitter(SplitConfig(seed="fixed")).assign_rows(source_rows)

    validate_manifest(rows)
    assert rows[0].split == rows[1].split
    assert all("source-record" in row.speaker_pseudo_id for row in rows)
    assert {row.generator_name for row in rows if row.label == "spoof"} == {"alg_1"}
