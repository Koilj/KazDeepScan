from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from kds.data.assets import sha256_file
from kds.data.denis import DENIS_SOURCE_ID, DenisRecord
from kds.eval.denis_selection import DENIS_SINGLE_SPEAKER_GROUP
from scripts.materialize_denis_mdc_pre_qa import (
    FrozenDenisSelectionRow,
    bind_denis_selection,
    load_frozen_denis_selection,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_selection(project: Path) -> tuple[Path, Path, tuple[DenisRecord, ...]]:
    manifests = project / "data" / "manifests"
    licenses = project / "data" / "licenses"
    manifests.mkdir(parents=True)
    licenses.mkdir(parents=True)
    source_audit = licenses / "source.json"
    source_screen = manifests / "screen.json"
    source_audit.write_text("{}\n", encoding="utf-8")
    source_screen.write_text("{}\n", encoding="utf-8")
    selection = manifests / "selection.csv"
    records: list[DenisRecord] = []
    with selection.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "selection_rank",
                "sample_id",
                "member_stem",
                "category",
                "parent_group_id",
                "speaker_pseudo_id",
                "text_id",
                "literal_text_sha256",
                "whitespace_canonical_text_sha256",
                "nfkc_whitespace_canonical_text_sha256",
                "source_audio_sha256",
                "source_audio_size_bytes",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        categories = ("General", "Chat", "CustomerService")
        for rank in range(1, 80):
            category = categories[(rank - 1) % 3]
            identity = f"{category}:{rank}"
            literal = _hash(f"literal:{identity}")
            canonical = _hash(f"canonical:{identity}")
            nfkc = _hash(f"nfkc:{identity}")
            audio_hash = _hash(f"audio:{identity}")
            sample_id = f"{DENIS_SOURCE_ID}:{identity}"
            member_stem = f"ru-RU/{category}/{rank:010d}"
            writer.writerow(
                {
                    "selection_rank": rank,
                    "sample_id": sample_id,
                    "member_stem": member_stem,
                    "category": category,
                    "parent_group_id": DENIS_SINGLE_SPEAKER_GROUP,
                    "speaker_pseudo_id": DENIS_SINGLE_SPEAKER_GROUP,
                    "text_id": f"{DENIS_SOURCE_ID}:text:{canonical}",
                    "literal_text_sha256": literal,
                    "whitespace_canonical_text_sha256": canonical,
                    "nfkc_whitespace_canonical_text_sha256": nfkc,
                    "source_audio_sha256": audio_hash,
                    "source_audio_size_bytes": 1000 + rank,
                }
            )
            records.append(
                DenisRecord(
                    sample_id=sample_id,
                    member_stem=member_stem,
                    category=category,
                    literal_text_sha256=literal,
                    whitespace_canonical_text_sha256=canonical,
                    nfkc_whitespace_canonical_text_sha256=nfkc,
                    audio_sha256=audio_hash,
                    audio_size_bytes=1000 + rank,
                    decoded_frames=144_000 + rank,
                    sample_rate_hz=48_000,
                    channels=2,
                    decoded_container="OGG",
                    decoded_subtype="OPUS",
                )
            )
    receipt = manifests / "selection.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": "denis-1-0-mdc-pre-qa-selection-v1",
                "archive": {
                    "expected_size_bytes": 109_594_943,
                    "expected_sha256": (
                        "75e2c63c5082df7623c6a98c529718b22015dfbd2d38a1ea328635f4dd4ccf9b"
                    ),
                },
                "output_selection": {
                    "path": "data/manifests/selection.csv",
                    "sha256": sha256_file(selection),
                    "rows": 79,
                },
                "selection_policy": {
                    "target_pairs": 79,
                    "requested_records": 79,
                    "selected_records": 79,
                    "selected_speaker_groups": 1,
                    "literal_and_canonical_text_hashes_bound_before_materialization": True,
                    "single_source_speaker_group_retained_for_every_row": True,
                    "post_selection_replacement_or_backfill": False,
                    "selection_uses_audio_or_duration": False,
                    "selection_uses_audio_quality_or_vad": False,
                    "selection_uses_detector_or_model_output": False,
                },
                "claims": {
                    "selection_frozen": True,
                    "audio_extraction_performed": False,
                    "technical_decode_qa_vad_performed": False,
                    "qa_rejects_must_not_trigger_replacement_or_backfill": True,
                    "training_data_overlap_unverified": True,
                    "single_speaker": True,
                    "speaker_independent": False,
                },
                "inputs": {
                    "source_audit_receipt": {
                        "path": "data/licenses/source.json",
                        "sha256": sha256_file(source_audit),
                    },
                    "source_exposure_screen": {
                        "path": "data/manifests/screen.json",
                        "sha256": sha256_file(source_screen),
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return selection, receipt, tuple(records)


def test_frozen_denis_selection_binds_all_exact_archive_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    selection_csv, selection_receipt, records = _write_selection(project)
    monkeypatch.chdir(project)

    selection = load_frozen_denis_selection(
        selection_csv.relative_to(project), selection_receipt.relative_to(project), Path(".")
    )

    assert len(selection) == 79
    assert isinstance(selection[0], FrozenDenisSelectionRow)
    assert len(bind_denis_selection(selection, records)) == 79


def test_frozen_denis_selection_rejects_changed_audio_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    selection_csv, selection_receipt, records = _write_selection(project)
    monkeypatch.chdir(project)
    selection = load_frozen_denis_selection(
        selection_csv.relative_to(project), selection_receipt.relative_to(project), Path(".")
    )
    first = records[0]
    changed = DenisRecord(
        sample_id=first.sample_id,
        member_stem=first.member_stem,
        category=first.category,
        literal_text_sha256=first.literal_text_sha256,
        whitespace_canonical_text_sha256=first.whitespace_canonical_text_sha256,
        nfkc_whitespace_canonical_text_sha256=first.nfkc_whitespace_canonical_text_sha256,
        audio_sha256="0" * 64,
        audio_size_bytes=first.audio_size_bytes,
        decoded_frames=first.decoded_frames,
        sample_rate_hz=first.sample_rate_hz,
        channels=first.channels,
        decoded_container=first.decoded_container,
        decoded_subtype=first.decoded_subtype,
    )

    with pytest.raises(ValueError, match="binding changed"):
        bind_denis_selection(selection, (changed, *records[1:]))
