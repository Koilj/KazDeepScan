from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from kds.data.manifest import ManifestRow, write_manifest
from kds.eval.candidate_exposure import CandidateExposureError, audit_candidate_project_exposure


def _row(*, sample_id: str, split: str, sha256: str, text_hash: str) -> ManifestRow:
    return ManifestRow(
        sample_id=sample_id,
        relative_path=f"processed/{sample_id}.wav",
        sha256=sha256,
        split=split,
        label="bonafide",
        language="ru",
        code_switch="false",
        parent_group_id=f"group-{sample_id}",
        source_name="candidate-source",
        source_license="CC0-1.0",
        rights_basis="personal research",
        speaker_pseudo_id=f"speaker-{sample_id}",
        text_id=f"text-{sample_id}",
        text_hash=text_hash,
        duration_s=1.0,
        generator_family="",
        generator_name="",
        generator_version="",
        voice_id="",
        clone_consent_id="",
        device="unknown",
        capture_route="source",
        original_sr=16_000,
        codec="wav",
        augmentation_chain="",
        augmentation_seed="",
        created_at="2026-08-13T00:00:00Z",
    )


def _write_config(config_root: Path, manifest: Path) -> None:
    config_root.mkdir()
    (config_root / "run.json").write_text(
        json.dumps({"roles": {"train": {"manifest": "../manifests/prior.csv"}}}),
        encoding="utf-8",
    )


def test_candidate_exposure_excludes_only_selected_rows_from_its_source_manifest(
    tmp_path: Path,
) -> None:
    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()
    source = manifest_root / "source.csv"
    candidate = _row(sample_id="candidate", split="test", sha256="a" * 64, text_hash="b" * 64)
    sibling = _row(sample_id="sibling", split="train", sha256="c" * 64, text_hash="d" * 64)
    write_manifest(source, [candidate, sibling])
    raw_source = manifest_root / "raw-source.csv"
    write_manifest(
        raw_source,
        [
            replace(
                candidate,
                relative_path="raw/candidate.mp3",
                sha256="1" * 64,
                original_sr=48_000,
                codec="mp3",
            ),
            replace(
                candidate,
                sample_id="raw-sibling",
                relative_path="raw/sibling.mp3",
                sha256="2" * 64,
                text_id="text-raw-sibling",
                text_hash="3" * 64,
                original_sr=48_000,
                codec="mp3",
            ),
        ],
    )
    prior = manifest_root / "prior.csv"
    write_manifest(
        prior,
        [_row(sample_id="prior", split="train", sha256="e" * 64, text_hash="f" * 64)],
    )
    config_root = tmp_path / "configs"
    _write_config(config_root, prior)

    receipt = audit_candidate_project_exposure(
        candidate_manifest=source,
        candidate_split="test",
        source_manifest=source,
        related_source_manifests=[raw_source],
        project_root=tmp_path,
        config_root=config_root,
        manifest_root=manifest_root,
        created_at="2026-08-13T00:00:00Z",
    )

    candidate_record = cast(dict[str, object], receipt["candidate"])
    source_records = cast(list[object], receipt["candidate_source_manifests"])
    source_record = cast(dict[str, object], source_records[0])
    raw_source_record = cast(dict[str, object], source_records[1])
    assert candidate_record["rows"] == 1
    assert source_record["selected_candidate_rows_excluded_from_inventory"] == 1
    assert raw_source_record["binding"] == "same_sample_text_group_lineage"
    assert receipt["configured_role_overlap_counts"] == {
        "sample_id": 0,
        "sha256": 0,
        "text_hash": 0,
        "parent_group_id": 0,
        "speaker_pseudo_id": 0,
    }
    assert receipt["inventory_overlap_counts"] == {
        "sample_id": 0,
        "sha256": 0,
        "text_hash": 0,
        "parent_group_id": 1,
        "speaker_pseudo_id": 1,
    }


def test_candidate_exposure_fails_when_candidate_text_was_used_by_a_prior_config(
    tmp_path: Path,
) -> None:
    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()
    source = manifest_root / "source.csv"
    candidate = _row(sample_id="candidate", split="test", sha256="a" * 64, text_hash="b" * 64)
    write_manifest(source, [candidate])
    prior = manifest_root / "prior.csv"
    write_manifest(
        prior,
        [_row(sample_id="prior", split="train", sha256="c" * 64, text_hash="b" * 64)],
    )
    config_root = tmp_path / "configs"
    _write_config(config_root, prior)

    with pytest.raises(CandidateExposureError, match="configured.text_hash=1"):
        audit_candidate_project_exposure(
            candidate_manifest=source,
            candidate_split="test",
            source_manifest=source,
            project_root=tmp_path,
            config_root=config_root,
            manifest_root=manifest_root,
            created_at="2026-08-13T00:00:00Z",
        )
