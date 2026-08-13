from __future__ import annotations

import csv
from pathlib import Path

from kds.data.manifest import MANIFEST_FIELD_ORDER
from kds.eval.voxcpm2_route_screen import screen_voxcpm2_project_history


def _write_manifest(path: Path, *, family: str, name: str) -> None:
    row = {
        "sample_id": "sample-1",
        "relative_path": "sample.wav",
        "sha256": "a" * 64,
        "split": "test",
        "label": "spoof",
        "language": "ru",
        "code_switch": "false",
        "parent_group_id": "group-1",
        "source_name": "source",
        "source_license": "Apache-2.0",
        "rights_basis": "fixture",
        "speaker_pseudo_id": "speaker-1",
        "text_id": "text-1",
        "text_hash": "b" * 64,
        "duration_s": "3.0",
        "generator_family": family,
        "generator_name": name,
        "generator_version": "revision-1",
        "voice_id": "voice-1",
        "clone_consent_id": "not_applicable",
        "device": "cpu",
        "capture_route": "fixture",
        "original_sr": "16000",
        "codec": "wav",
        "augmentation_chain": "",
        "augmentation_seed": "",
        "created_at": "2026-08-14T00:00:00Z",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELD_ORDER)
        writer.writeheader()
        writer.writerow(row)


def test_route_screen_accepts_history_without_voxcpm(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    receipt = tmp_path / "artifact.json"
    receipt.write_text("{}\n", encoding="utf-8")
    _write_manifest(manifests / "prior.csv", family="other_tts", name="Other TTS")

    result = screen_voxcpm2_project_history(
        project_root=tmp_path,
        manifest_root=manifests,
        created_at="2026-08-14T00:00:00Z",
        artifact_receipt_path=receipt,
    )

    assert result["claims"]["new_project_generator_family"] is True  # type: ignore[index]


def test_route_screen_fails_novelty_claim_on_voxcpm_history(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    receipt = tmp_path / "artifact.json"
    receipt.write_text("{}\n", encoding="utf-8")
    _write_manifest(manifests / "prior.csv", family="OpenBMB_VoxCPM", name="VoxCPM")

    result = screen_voxcpm2_project_history(
        project_root=tmp_path,
        manifest_root=manifests,
        created_at="2026-08-14T00:00:00Z",
        artifact_receipt_path=receipt,
    )

    assert result["claims"]["new_project_generator_family"] is False  # type: ignore[index]
    assert result["voxcpm_history"]["matching_manifest_rows"] == 1  # type: ignore[index]
