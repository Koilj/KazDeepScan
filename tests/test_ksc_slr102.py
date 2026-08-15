from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

import kds.data.ksc_slr102 as ksc


def _write_metadata_root(root: Path) -> None:
    (root / "Meta").mkdir(parents=True)
    (root / "Transcriptions").mkdir()
    (root / "Meta" / "train.csv").write_text("uttID deviceID\nu1 device-1\n", encoding="utf-8")
    (root / "Meta" / "dev.csv").write_text("uttID deviceID\nu2 device-2\n", encoding="utf-8")
    (root / "Meta" / "test.csv").write_text("uttID deviceID\nu3 device-3\n", encoding="utf-8")
    rows = (("u1", "бірінші  мәтін\n"), ("u2", "екінші"), ("u3", "үшінші"))
    for utterance_id, transcript in rows:
        (root / "Transcriptions" / f"{utterance_id}.txt").write_text(transcript, encoding="utf-8")


def _write_small_ksc_archive(
    archive: Path,
    records: tuple[tuple[str, str], ...] = (("u1", "сөйлем"),),
) -> None:
    with tarfile.open(archive, mode="w:gz") as tar:
        for name in (
            ksc.KSC_ARCHIVE_ROOT,
            ksc.KSC_AUDIO_DIRECTORY,
            ksc.KSC_TRANSCRIPT_DIRECTORY,
            ksc.KSC_METADATA_DIRECTORY,
        ):
            directory = tarfile.TarInfo(name)
            directory.type = tarfile.DIRTYPE
            tar.addfile(directory)
        for utterance_id, transcript_text in records:
            payload = f"minimal-flac-bytes-{utterance_id}".encode()
            audio = tarfile.TarInfo(f"{ksc.KSC_AUDIO_DIRECTORY}/{utterance_id}.flac")
            audio.size = len(payload)
            tar.addfile(audio, io.BytesIO(payload))
            transcript = f"{transcript_text}\n".encode()
            transcript_info = tarfile.TarInfo(
                f"{ksc.KSC_TRANSCRIPT_DIRECTORY}/{utterance_id}.txt"
            )
            transcript_info.size = len(transcript)
            tar.addfile(transcript_info, io.BytesIO(transcript))
        for name in ksc.KSC_METADATA_SPLITS.values():
            metadata = (
                "uttID deviceID\n"
                + "".join(
                    f"{utterance_id} device-{index}\n"
                    for index, (utterance_id, _text) in enumerate(records, start=1)
                )
            ).encode()
            metadata_info = tarfile.TarInfo(f"{ksc.KSC_METADATA_DIRECTORY}/{name}")
            metadata_info.size = len(metadata)
            tar.addfile(metadata_info, io.BytesIO(metadata))


def test_ksc_metadata_keeps_unknown_labels_and_never_uses_device_as_speaker(tmp_path: Path) -> None:
    _write_metadata_root(tmp_path)

    records = ksc.load_ksc_metadata(tmp_path, ["train", "dev", "test"])
    assets = {
        record.utterance_id: ksc.ExtractedKscAsset(
            utterance_id=record.utterance_id,
            relative_path=f"raw/ksc/{record.utterance_id}.flac",
            sha256=hashlib.sha256(record.utterance_id.encode()).hexdigest(),
            duration_s=3.0,
            original_sr=16_000,
            codec="flac",
        )
        for record in records
    }

    rows = ksc.ksc_manifest_rows(records, assets, created_at="2026-08-08T00:00:00Z")

    assert [row.split for row in rows] == ["train", "dev", "test"]
    assert all(row.code_switch == "unknown" for row in rows)
    assert all("device-" not in row.speaker_pseudo_id for row in rows)
    assert all("device-" not in row.parent_group_id for row in rows)
    assert rows[0].text_hash == hashlib.sha256("бірінші мәтін".encode()).hexdigest()


def test_ksc_record_selection_is_deterministic_and_limited_per_source_split(tmp_path: Path) -> None:
    _write_metadata_root(tmp_path)
    (tmp_path / "Meta" / "train.csv").write_text(
        "uttID deviceID\nu1 device-1\nu4 device-4\n", encoding="utf-8"
    )
    (tmp_path / "Transcriptions" / "u4.txt").write_text("төртінші", encoding="utf-8")
    records = ksc.load_ksc_metadata(tmp_path, ["train", "dev"])

    first = ksc.select_ksc_records(records, limit=1, seed="seed")
    second = ksc.select_ksc_records(records, limit=1, seed="seed")

    assert first == second
    assert {record.split for record in first} == {"train", "dev"}
    assert len(first) == 2


def test_ksc_record_selection_excludes_prior_utterances_before_ranking(tmp_path: Path) -> None:
    _write_metadata_root(tmp_path)
    (tmp_path / "Meta" / "train.csv").write_text(
        "uttID deviceID\nu1 device-1\nu4 device-4\n", encoding="utf-8"
    )
    (tmp_path / "Transcriptions" / "u4.txt").write_text("төртінші", encoding="utf-8")
    records = ksc.load_ksc_metadata(tmp_path, ["train"])

    selected = ksc.select_ksc_records(records, limit=1, seed="seed", excluded_utterance_ids={"u1"})

    assert [record.utterance_id for record in selected] == ["u4"]
    with pytest.raises(ksc.KscIngestionError, match="after exclusions"):
        ksc.select_ksc_records(records, limit=2, seed="seed", excluded_utterance_ids={"u1"})


def test_ksc_archive_selection_filters_frozen_text_before_audio_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / ksc.KSC_ARCHIVE_NAME
    _write_small_ksc_archive(archive)
    monkeypatch.setattr(ksc, "KSC_ARCHIVE_EXPECTED_SIZE_BYTES", archive.stat().st_size)
    records = ksc.load_ksc_metadata_from_archive(archive, ["train"])

    selected, report = ksc.select_ksc_records_from_archive_excluding_texts(
        archive, records, limit=1, seed="seed"
    )

    assert [record.utterance_id for record in selected] == ["u1"]
    assert report.audio_files == 1
    with pytest.raises(ksc.KscIngestionError, match="text-disjoint"):
        ksc.select_ksc_records_from_archive_excluding_texts(
            archive,
            records,
            limit=1,
            seed="seed",
            excluded_text_hashes={hashlib.sha256("сөйлем".encode()).hexdigest()},
        )


def test_ksc_archive_is_extracted_atomically_after_layout_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / ksc.KSC_ARCHIVE_NAME
    _write_small_ksc_archive(archive)
    monkeypatch.setattr(ksc, "KSC_ARCHIVE_EXPECTED_SIZE_BYTES", archive.stat().st_size)
    output_parent = tmp_path / "output"
    output_parent.mkdir()

    report = ksc.inspect_ksc_archive(archive)
    metadata = ksc.load_ksc_metadata_from_archive(archive, ["train"])
    extracted = ksc.extract_ksc_audio_slice(archive, ["u1"], output_parent / "slice")

    assert report.audio_files == 1
    assert report.transcript_files == 1
    assert report.metadata_files == 3
    assert metadata[0].utterance_id == "u1"
    assert extracted["u1"].read_bytes() == b"minimal-flac-bytes-u1"
    assert (output_parent / "slice" / "Transcriptions" / "u1.txt").read_text() == "сөйлем\n"


def test_ksc_extraction_rejects_frozen_text_before_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / ksc.KSC_ARCHIVE_NAME
    _write_small_ksc_archive(archive)
    monkeypatch.setattr(ksc, "KSC_ARCHIVE_EXPECTED_SIZE_BYTES", archive.stat().st_size)
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    frozen_hash = hashlib.sha256("сөйлем".encode()).hexdigest()

    with pytest.raises(ksc.KscIngestionError, match="transcript text hash"):
        ksc.extract_ksc_audio_slice(
            archive,
            ["u1"],
            output_parent / "slice",
            excluded_text_hashes={frozen_hash},
        )

    assert not (output_parent / "slice").exists()


def test_ksc_archive_selection_keeps_only_one_unseen_transcript_per_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / ksc.KSC_ARCHIVE_NAME
    _write_small_ksc_archive(
        archive,
        (("u1", "бірдей"), ("u2", "бірдей"), ("u3", "бірегей")),
    )
    monkeypatch.setattr(ksc, "KSC_ARCHIVE_EXPECTED_SIZE_BYTES", archive.stat().st_size)
    records = ksc.load_ksc_metadata_from_archive(archive, ["train"])

    selected, _report = ksc.select_ksc_records_from_archive_excluding_texts(
        archive,
        records,
        limit=2,
        seed="seed",
    )

    assert len(selected) == 2
    assert "u3" in {item.utterance_id for item in selected}


def test_ksc_metadata_rejects_path_traversal_in_utterance_id(tmp_path: Path) -> None:
    _write_metadata_root(tmp_path)
    (tmp_path / "Meta" / "train.csv").write_text(
        "uttID deviceID\n../outside device-1\n", encoding="utf-8"
    )

    with pytest.raises(ksc.KscIngestionError, match="Invalid KSC utterance id"):
        ksc.load_ksc_metadata(tmp_path, ["train"])
