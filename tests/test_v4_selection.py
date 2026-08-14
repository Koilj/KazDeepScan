from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from kds.data.ksc2 import Ksc2TextCandidate
from kds.data.ruasd_catalog import load_ruasd_artifact_catalog
from kds.data.v4_selection import (
    V4ExposureInventory,
    V4SelectionConfig,
    V4SelectionError,
    load_v4_selection_config,
    publish_v4_train_candidate_selection,
    select_v4_ksc2_candidates,
    select_v4_ruasd_candidates,
)


def _config() -> V4SelectionConfig:
    return V4SelectionConfig(
        protocol_id="test-v4-selection",
        capacity_receipt_path="capacity.json",
        capacity_receipt_sha256="a" * 64,
        selection_seed="fixed-test-seed",
        target_rows_per_cell=1,
        candidate_rows_per_cell=2,
        ruasd_excluded_subsets=frozenset({"CommonVoice"}),
        ruasd_require_source_text=True,
        ruasd_min_per_stratum=1,
        ruasd_max_per_stratum=1,
        ksc2_component_quotas={"Train/radio": 1, "Train/tv_news": 1},
        kk_generator_quotas={"route-a": 1, "route-b": 1},
        kk_generator_families={"route-a": "family-a", "route-b": "family-b"},
        roles={"train": {}, "dev": {}, "calibration": {}, "final": {}},
        source_lineage_roots={
            "train": ("source-train",),
            "dev": ("source-dev",),
            "calibration": ("source-calibration",),
            "final": ("source-final",),
        },
        tts_family_roots={
            "train": ("family-a", "family-b"),
            "dev": ("family-dev",),
            "calibration": ("family-calibration",),
            "final": ("family-final",),
        },
    )


def _exposure(*, sample_ids: frozenset[str] = frozenset()) -> V4ExposureInventory:
    return V4ExposureInventory(
        manifest_bindings=({"path": "history.csv", "sha256": "b" * 64, "rows": 1},),
        rows=1,
        sample_ids=sample_ids,
        audio_sha256=frozenset(),
        text_hashes=frozenset(),
        parent_group_ids=frozenset(),
        speaker_group_ids=frozenset(),
    )


def _ruasd_record(sample_id: str, *, label: str, subset: str) -> dict[str, object]:
    filename = f"{sample_id}.wav"
    return {
        "sample_id": sample_id,
        "label": label,
        "group": "raw",
        "subset": subset,
        "source_type": "real_speech" if label == "real" else "tts",
        "model": f"model-{subset}" if label == "fake" else "",
        "filename": filename,
        "audio_relpath": f"raw/{label}/{subset}/audio/{filename}",
        "transcription": f"Текст {sample_id}",
    }


def _write_ruasd_archive(path: Path, records: list[dict[str, object]]) -> None:
    with tarfile.open(path, mode="w") as archive:
        for record in records:
            sample_id = str(record["sample_id"])
            payload = json.dumps(record, ensure_ascii=False).encode()
            metadata = tarfile.TarInfo(f"{sample_id}.json")
            metadata.size = len(payload)
            archive.addfile(metadata, io.BytesIO(payload))
            audio_payload = sample_id.encode()
            audio = tarfile.TarInfo(f"{sample_id}.wav")
            audio.size = len(audio_payload)
            archive.addfile(audio, io.BytesIO(audio_payload))


def _write_ruasd_catalog(path: Path, archive: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "archive_name",
                "expected_size_bytes",
                "sha256",
                "pinned_revision",
                "source_url",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "archive_name": archive.name,
                "expected_size_bytes": archive.stat().st_size,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "pinned_revision": "test",
                "source_url": "https://example.test/ruasd",
            }
        )


def test_repository_v4_roles_and_selection_config_loads() -> None:
    config = load_v4_selection_config(
        Path("configs/research/v4/xlsr_sls_model_v4_roles_and_selection_v2.json")
    )

    assert config.candidate_rows_per_cell == 7200
    assert sum(config.kk_generator_quotas.values()) == 7200
    assert len(set(config.kk_generator_families.values())) == 4
    assert "google_fleurs_ru_v1" not in {
        root for roots in config.source_lineage_roots.values() for root in roots
    }


def test_ruasd_selection_excludes_common_voice_and_freezes_strata(tmp_path: Path) -> None:
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    archive = archive_dir / "ruasd-000000.tar"
    _write_ruasd_archive(
        archive,
        [
            _ruasd_record("real-a", label="real", subset="News-A"),
            _ruasd_record("real-b", label="real", subset="News-B"),
            _ruasd_record("real-cv", label="real", subset="CommonVoice"),
            _ruasd_record("fake-a", label="fake", subset="TTS-A"),
            _ruasd_record("fake-b", label="fake", subset="TTS-B"),
        ],
    )
    catalog_path = tmp_path / "catalog.csv"
    _write_ruasd_catalog(catalog_path, archive)

    selection = select_v4_ruasd_candidates(
        archive_dir,
        load_ruasd_artifact_catalog(catalog_path),
        config=_config(),
        exposure=_exposure(),
    )

    assert len(selection.rows) == 4
    assert {row.source_component for row in selection.rows if row.label == "bonafide"} == {
        "News-A",
        "News-B",
    }
    assert selection.rejection_counts == {"excluded_source_subset": 1}
    assert all(row.raw_audio_sha256 == "" for row in selection.rows)


def test_ksc2_selection_excludes_history_and_assigns_exact_generator_quotas() -> None:
    candidates = (
        Ksc2TextCandidate(
            candidate_id="Train/radio/history",
            component="Train/radio",
            archive_audio_member="ISSAI_KSC2/Train/radio/history.flac",
            archive_transcript_member="ISSAI_KSC2/Train/radio/history.txt",
            transcript_sha256="1" * 64,
            canonical_text_sha256="1" * 64,
        ),
        Ksc2TextCandidate(
            candidate_id="Train/radio/a",
            component="Train/radio",
            archive_audio_member="ISSAI_KSC2/Train/radio/a.flac",
            archive_transcript_member="ISSAI_KSC2/Train/radio/a.txt",
            transcript_sha256="2" * 64,
            canonical_text_sha256="2" * 64,
        ),
        Ksc2TextCandidate(
            candidate_id="Train/tv_news/b",
            component="Train/tv_news",
            archive_audio_member="ISSAI_KSC2/Train/tv_news/b.flac",
            archive_transcript_member="ISSAI_KSC2/Train/tv_news/b.txt",
            transcript_sha256="3" * 64,
            canonical_text_sha256="3" * 64,
        ),
    )

    selection = select_v4_ksc2_candidates(
        candidates,
        config=_config(),
        exposure=_exposure(sample_ids=frozenset({"ksc2_v1:Train/radio/history"})),
    )

    assert len(selection.rows) == 4
    assert selection.rejection_counts == {"historical_sample_id": 1}
    spoof = [row for row in selection.rows if row.label == "spoof"]
    assert {row.generator_route_id for row in spoof} == {"route-a", "route-b"}
    assert {row.target_state for row in spoof} == {"target", "reserve"}


def test_v4_selection_publication_is_atomic_and_never_overwrites(tmp_path: Path) -> None:
    config = _config()
    candidates = (
        Ksc2TextCandidate(
            candidate_id="Train/radio/a",
            component="Train/radio",
            archive_audio_member="ISSAI_KSC2/Train/radio/a.flac",
            archive_transcript_member="ISSAI_KSC2/Train/radio/a.txt",
            transcript_sha256="2" * 64,
            canonical_text_sha256="2" * 64,
        ),
        Ksc2TextCandidate(
            candidate_id="Train/tv_news/b",
            component="Train/tv_news",
            archive_audio_member="ISSAI_KSC2/Train/tv_news/b.flac",
            archive_transcript_member="ISSAI_KSC2/Train/tv_news/b.txt",
            transcript_sha256="3" * 64,
            canonical_text_sha256="3" * 64,
        ),
    )
    ksc2 = select_v4_ksc2_candidates(candidates, config=config, exposure=_exposure())
    ru_rows = tuple(
        replace(
            row,
            language="ru",
            candidate_id=f"ru:{row.candidate_id}",
            pair_id=f"ru:{row.pair_id}",
            source_id="ru-test-source",
            source_lineage_id="ru-test-source:train-only",
            parent_group_id=f"ru:{row.parent_group_id}",
            text_hash=("4" if row.label == "bonafide" else "5") * 64
            if row.selection_rank == 1
            else ("6" if row.label == "bonafide" else "7") * 64,
            canonical_text_hash=("4" if row.label == "bonafide" else "5") * 64
            if row.selection_rank == 1
            else ("6" if row.label == "bonafide" else "7") * 64,
        )
        for row in ksc2.rows
    )
    # Slots dataclasses intentionally have no writable attributes; build a synthetic source receipt.
    ruasd = ksc2.__class__(
        rows=ru_rows,
        available_by_stratum={"test": 2},
        selected_by_stratum={"test": 2},
        rejection_counts={},
    )
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    manifest_dir = tmp_path / "manifests"
    receipt_dir = tmp_path / "receipts"
    manifest_dir.mkdir()
    receipt_dir.mkdir()
    output_csv = manifest_dir / "selection.csv"
    output_receipt = receipt_dir / "selection.json"

    publish_v4_train_candidate_selection(
        output_csv=output_csv,
        output_receipt=output_receipt,
        rows=(*ru_rows, *ksc2.rows),
        config_path=config_path,
        config=config,
        exposure=_exposure(),
        ruasd_selection=ruasd,
        ksc2_selection=ksc2,
        created_at="2026-08-14T12:00:00+06:00",
        source_bindings={},
    )

    assert output_csv.is_file() and output_receipt.is_file()
    receipt = json.loads(output_receipt.read_text(encoding="utf-8"))
    assert receipt["claims"]["historical_sample_or_text_collisions_selected"] is False
    assert (
        receipt["claims"]["historical_exact_audio_collision_status"]
        == "not_checked_audio_unmaterialized"
    )
    with pytest.raises(V4SelectionError, match="Unsafe v4 selection output"):
        publish_v4_train_candidate_selection(
            output_csv=output_csv,
            output_receipt=output_receipt,
            rows=(*ru_rows, *ksc2.rows),
            config_path=config_path,
            config=config,
            exposure=_exposure(),
            ruasd_selection=ruasd,
            ksc2_selection=ksc2,
            created_at="2026-08-14T12:00:00+06:00",
            source_bindings={},
        )
