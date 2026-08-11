from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from kds.data.manifest import validate_manifest
from kds.data.ruasd_catalog import load_ruasd_artifact_catalog, sha256_file
from kds.data.ruasd_research import (
    ExtractedRuAsdResearchAsset,
    RuAsdResearchError,
    extract_ruasd_research_slice,
    ruasd_research_manifest_rows,
    select_ruasd_research_records,
    write_ruasd_research_selection_receipt,
)


def _write_archive(path: Path, records: list[dict[str, object]]) -> None:
    with tarfile.open(path, mode="w") as archive:
        for record in records:
            sample_id = str(record["sample_id"])
            payload = json.dumps(record).encode()
            metadata = tarfile.TarInfo(f"{sample_id}.json")
            metadata.size = len(payload)
            archive.addfile(metadata, io.BytesIO(payload))
            audio = sample_id.encode()
            audio_info = tarfile.TarInfo(f"{sample_id}.wav")
            audio_info.size = len(audio)
            archive.addfile(audio_info, io.BytesIO(audio))


def _record(sample_id: str, *, label: str, subset: str, model: str = "") -> dict[str, object]:
    filename = f"{sample_id}.wav"
    return {
        "sample_id": sample_id,
        "label": label,
        "group": "raw",
        "subset": subset,
        "source_type": "real_speech" if label == "real" else "tts",
        "model": model,
        "filename": filename,
        "audio_relpath": f"raw/{label}/{subset}/audio/{filename}",
        "transcription": f"Текст {sample_id}",
    }


def _write_catalog(path: Path, archives: list[Path]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=[
                "archive_name",
                "expected_size_bytes",
                "sha256",
                "pinned_revision",
                "source_url",
            ],
        )
        writer.writeheader()
        for archive in archives:
            writer.writerow(
                {
                    "archive_name": archive.name,
                    "expected_size_bytes": archive.stat().st_size,
                    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    "pinned_revision": "revision",
                    "source_url": "https://example.test/ruasd",
                }
            )


def test_full_research_selection_is_balanced_and_extracts_atomically(tmp_path: Path) -> None:
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    first = archive_dir / "ruasd-000000.tar"
    second = archive_dir / "ruasd-000001.tar"
    _write_archive(
        first,
        [
            _record("real_cv_a", label="real", subset="CommonVoice"),
            _record("real_cv_b", label="real", subset="CommonVoice"),
            _record("fake_a_a", label="fake", subset="TTS-A", model="model-a"),
            _record("fake_a_b", label="fake", subset="TTS-A", model="model-a"),
        ],
    )
    _write_archive(
        second,
        [
            _record("real_news_a", label="real", subset="News"),
            _record("real_news_b", label="real", subset="News"),
            _record("fake_b_a", label="fake", subset="TTS-B", model="model-b"),
            _record("fake_b_b", label="fake", subset="TTS-B", model="model-b"),
        ],
    )
    catalog_path = tmp_path / "catalog.csv"
    _write_catalog(catalog_path, [first, second])
    catalog = load_ruasd_artifact_catalog(catalog_path)

    selection = select_ruasd_research_records(
        archive_dir,
        catalog,
        limit_per_label=4,
        min_per_stratum=1,
        seed="fixed",
        verify_sha256=True,
    )
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    extracted = extract_ruasd_research_slice(
        archive_dir, catalog, selection.records, output_parent / "slice"
    )

    assert len(selection.records) == 8
    assert selection.sha256_verified_archives == 2
    assert selection.selected_stratum_counts == {
        "bonafide/CommonVoice": 2,
        "bonafide/News": 2,
        "spoof/TTS-A": 2,
        "spoof/TTS-B": 2,
    }
    assert all(path.is_file() for path in extracted.values())

    assets = {
        record_key: ExtractedRuAsdResearchAsset(
            record_key=record_key,
            relative_path=path.relative_to(tmp_path).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            duration_s=3.0,
            original_sr=16_000,
        )
        for record_key, path in extracted.items()
    }
    rows = ruasd_research_manifest_rows(
        selection.records, assets, created_at="2026-08-09T00:00:00Z"
    )
    validate_manifest(rows)
    assert {row.label for row in rows} == {"bonafide", "spoof"}
    assert all(row.text_id.startswith("ruasd_ru_v1_full:text:") for row in rows)


def test_shifted_transcript_model_is_normalized_before_stratification(tmp_path: Path) -> None:
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    archive = archive_dir / "ruasd-000000.tar"
    _write_archive(
        archive,
        [
            _record("real_a", label="real", subset="CommonVoice"),
            _record("real_b", label="real", subset="CommonVoice"),
            _record("fake_valid", label="fake", subset="TeraTTS", model="TeraTTS"),
            _record(
                "fake_shifted",
                label="fake",
                subset="TeraTTS",
                model="собранная международными организациями.",
            ),
        ],
    )
    catalog_path = tmp_path / "catalog.csv"
    _write_catalog(catalog_path, [archive])

    selection = select_ruasd_research_records(
        archive_dir,
        load_ruasd_artifact_catalog(catalog_path),
        limit_per_label=2,
        min_per_stratum=1,
        seed="fixed-v2",
        verify_sha256=True,
    )

    assert selection.selected_stratum_counts == {
        "bonafide/CommonVoice": 2,
        "spoof/TeraTTS": 2,
    }
    shifted = next(record for record in selection.records if record.sample_id == "fake_shifted")
    assert shifted.model == "unspecified_by_source"
    assert shifted.model_status == "invalid_source_metadata"
    assert shifted.source_model_sha256 == hashlib.sha256(
        "собранная международными организациями.".encode()
    ).hexdigest()
    assert selection.selected_model_status_counts == {
        "invalid_source_metadata": 1,
        "not_applicable": 2,
        "source_identifier": 1,
    }

    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("header\n", encoding="utf-8")
    ledger_path = tmp_path / "ledger.csv"
    ledger_path.write_text("header\n", encoding="utf-8")
    receipt_path = tmp_path / "selection.json"
    write_ruasd_research_selection_receipt(
        receipt_path,
        selection,
        seed="fixed-v2",
        limit_per_label=2,
        min_per_stratum=1,
        slice_name="test-v2",
        catalog_path=catalog_path,
        catalog_sha256=sha256_file(catalog_path),
        license_ledger_path=ledger_path,
        license_ledger_sha256=sha256_file(ledger_path),
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        created_at="2026-08-12T00:00:00Z",
    )
    receipt_text = receipt_path.read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    assert "собранная" not in receipt_text
    assert receipt["selected_model_status_counts"]["invalid_source_metadata"] == 1
    assert receipt["invalid_source_model_records"] == [
        {
            "record_key": shifted.record_key,
            "source_model_sha256": shifted.source_model_sha256,
        }
    ]
    with pytest.raises(RuAsdResearchError, match="Unsafe RuASD selection receipt"):
        write_ruasd_research_selection_receipt(
            receipt_path,
            selection,
            seed="fixed-v2",
            limit_per_label=2,
            min_per_stratum=1,
            slice_name="test-v2",
            catalog_path=catalog_path,
            catalog_sha256=sha256_file(catalog_path),
            license_ledger_path=ledger_path,
            license_ledger_sha256=sha256_file(ledger_path),
            manifest_path=manifest_path,
            manifest_sha256=sha256_file(manifest_path),
            created_at="2026-08-12T00:00:00Z",
        )
