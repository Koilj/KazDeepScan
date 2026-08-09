from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

from kds.data.manifest import validate_manifest
from kds.data.ruasd_catalog import load_ruasd_artifact_catalog
from kds.data.ruasd_research import (
    ExtractedRuAsdResearchAsset,
    extract_ruasd_research_slice,
    ruasd_research_manifest_rows,
    select_ruasd_research_records,
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
        "spoof/TTS-A/model-a": 2,
        "spoof/TTS-B/model-b": 2,
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
