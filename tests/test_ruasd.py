from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

import kds.data.ruasd as ruasd
from kds.data.manifest import validate_manifest


def _write_small_ruasd_archive(path: Path) -> None:
    records = (
        ("raw_fake_11labs_a", "11labs"),
        ("raw_fake_11labs_b", "11labs"),
        ("raw_fake_teratts_a", "TeraTTS"),
        ("raw_fake_teratts_b", "TeraTTS"),
    )
    with tarfile.open(path, mode="w") as archive:
        for sample_id, generator_name in records:
            filename = f"{sample_id}.wav"
            metadata = json.dumps(
                {
                    "sample_id": sample_id,
                    "label": "fake",
                    "group": "raw",
                    "subset": generator_name,
                    "source_type": "tts",
                    "filename": filename,
                    "audio_relpath": f"raw/fake/{generator_name}/audio/{filename}",
                }
            ).encode()
            metadata_info = tarfile.TarInfo(f"{sample_id}.json")
            metadata_info.size = len(metadata)
            archive.addfile(metadata_info, io.BytesIO(metadata))
            audio = sample_id.encode()
            audio_info = tarfile.TarInfo(filename)
            audio_info.size = len(audio)
            archive.addfile(audio_info, io.BytesIO(audio))


def test_ruasd_fake_shard_is_validated_and_extracted_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / ruasd.RUASD_ARCHIVE_NAME
    _write_small_ruasd_archive(archive)
    monkeypatch.setattr(ruasd, "RUASD_ARCHIVE_EXPECTED_SIZE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(
        ruasd, "RUASD_ARCHIVE_EXPECTED_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    output_parent = tmp_path / "output"
    output_parent.mkdir()

    records = ruasd.load_ruasd_fake_records(archive)
    selected = ruasd.select_ruasd_ood_records(records, limit_per_generator=1, seed="fixed")
    extracted = ruasd.extract_ruasd_ood_slice(archive, selected, output_parent / "slice")

    assert len(records) == 4
    assert len(selected) == 2
    assert set(extracted) == {record.sample_id for record in selected}
    assert all(path.is_file() for path in extracted.values())


def test_ruasd_rows_are_fake_only_russian_ood_with_generator_provenance() -> None:
    records = [
        ruasd.RuAsdRecord("raw_fake_11labs_a", "11labs"),
        ruasd.RuAsdRecord("raw_fake_teratts_a", "TeraTTS"),
    ]
    assets = {
        record.sample_id: ruasd.ExtractedRuAsdAsset(
            sample_id=record.sample_id,
            relative_path=f"raw/ruasd/{record.sample_id}.wav",
            sha256=hashlib.sha256(record.sample_id.encode()).hexdigest(),
            duration_s=3.0,
            original_sr=16_000,
        )
        for record in records
    }

    rows = ruasd.ruasd_ood_manifest_rows(records, assets, created_at="2026-08-09T00:00:00Z")

    validate_manifest(rows, require_ood_generator=True)
    assert {row.split for row in rows} == {"ood"}
    assert {row.language for row in rows} == {"ru"}
    assert {row.generator_name for row in rows} == {"11labs", "TeraTTS"}
