from __future__ import annotations

import json
from pathlib import Path

import pytest

from kds.data.licenses import load_license_ledger
from kds.data.manifest import ManifestRow, write_manifest
from kds.data.source_matrix import (
    SourceMatrixError,
    load_source_mixed_research_matrix,
    validate_source_mixed_research_matrix,
)
from tests.factories import manifest_mapping


def _write_ledger(
    path: Path, source_ids: tuple[str, ...], *, ood_use: str = "research_only"
) -> None:
    rows = [
        "source_id,usage_scope,train_dev_test_use,ood_evaluation_use,"
        "bonafide_group_provenance,spoof_voice_group_provenance,license,source_url,"
        "artifact_name,expected_size_bytes,last_modified_utc,sha256,rights_basis,status,notes"
    ]
    for source_id in source_ids:
        rows.append(
            f"{source_id},personal_research,research_only,{ood_use},unknown,unknown,"
            "CC-BY-4.0,https://example.test/source,source.tar,1024,2026-08-10T00:00:00Z,"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,"
            "research,verified,test"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_binary_manifest(path: Path, source_id: str, split: str, offset: int) -> None:
    rows: list[ManifestRow] = []
    for index, label in enumerate(("bonafide", "spoof")):
        spoof = label == "spoof"
        rows.append(
            ManifestRow.from_mapping(
                manifest_mapping(
                    sample_id=f"{source_id}-{label}",
                    sha256=f"{offset + index:064x}",
                    source_name=source_id,
                    split=split,
                    label=label,
                    parent_group_id=f"{source_id}-parent-{label}",
                    speaker_pseudo_id=f"{source_id}-speaker-{label}",
                    text_id=f"{source_id}-text-{label}",
                    text_hash=f"{source_id}-text-hash-{label}",
                    generator_family="generator" if spoof else "",
                    generator_name="generator-v1" if spoof else "",
                    generator_version="1" if spoof else "",
                    voice_id=f"{source_id}-voice" if spoof else "",
                ),
                row_number=index + 2,
            )
        )
    write_manifest(path, rows)


def _write_matrix(path: Path, *, dev_source: str = "dev-source") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": "matrix-test",
                "purpose": "research",
                "roles": [
                    {
                        "name": "train",
                        "manifest": "train.csv",
                        "source_split": "train",
                        "expected_source_ids": ["train-source"],
                    },
                    {
                        "name": "dev",
                        "manifest": "dev.csv",
                        "source_split": "dev",
                        "expected_source_ids": [dev_source],
                    },
                    {
                        "name": "test",
                        "manifest": "test.csv",
                        "source_split": "ood",
                        "expected_source_ids": ["test-source"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_source_mixed_matrix_accepts_disjoint_binary_sources(tmp_path: Path) -> None:
    _write_binary_manifest(tmp_path / "train.csv", "train-source", "train", 1)
    _write_binary_manifest(tmp_path / "dev.csv", "dev-source", "dev", 10)
    _write_binary_manifest(tmp_path / "test.csv", "test-source", "ood", 20)
    _write_matrix(tmp_path / "matrix.json")
    ledger_path = tmp_path / "ledger.csv"
    _write_ledger(ledger_path, ("train-source", "dev-source", "test-source"))

    report = validate_source_mixed_research_matrix(
        load_source_mixed_research_matrix(tmp_path / "matrix.json"),
        load_license_ledger(ledger_path),
    )

    assert report.source_ids == ("dev-source", "test-source", "train-source")
    assert [role.rows for role in report.roles] == [2, 2, 2]
    assert report.roles[2].source_split == "ood"


def test_source_mixed_matrix_rejects_source_reused_across_roles(tmp_path: Path) -> None:
    _write_binary_manifest(tmp_path / "train.csv", "train-source", "train", 1)
    _write_binary_manifest(tmp_path / "dev.csv", "train-source", "dev", 10)
    _write_binary_manifest(tmp_path / "test.csv", "test-source", "ood", 20)
    _write_matrix(tmp_path / "matrix.json", dev_source="train-source")
    ledger_path = tmp_path / "ledger.csv"
    _write_ledger(ledger_path, ("train-source", "test-source"))

    with pytest.raises(SourceMatrixError, match="Source leakage"):
        validate_source_mixed_research_matrix(
            load_source_mixed_research_matrix(tmp_path / "matrix.json"),
            load_license_ledger(ledger_path),
        )


def test_source_mixed_matrix_rejects_asset_reused_across_distinct_sources(tmp_path: Path) -> None:
    _write_binary_manifest(tmp_path / "train.csv", "train-source", "train", 1)
    _write_binary_manifest(tmp_path / "dev.csv", "dev-source", "dev", 1)
    _write_binary_manifest(tmp_path / "test.csv", "test-source", "ood", 20)
    _write_matrix(tmp_path / "matrix.json")
    ledger_path = tmp_path / "ledger.csv"
    _write_ledger(ledger_path, ("train-source", "dev-source", "test-source"))

    with pytest.raises(SourceMatrixError, match="Duplicate sha256"):
        validate_source_mixed_research_matrix(
            load_source_mixed_research_matrix(tmp_path / "matrix.json"),
            load_license_ledger(ledger_path),
        )


def test_source_mixed_matrix_respects_ood_rights(tmp_path: Path) -> None:
    _write_binary_manifest(tmp_path / "train.csv", "train-source", "train", 1)
    _write_binary_manifest(tmp_path / "dev.csv", "dev-source", "dev", 10)
    _write_binary_manifest(tmp_path / "test.csv", "test-source", "ood", 20)
    _write_matrix(tmp_path / "matrix.json")
    ledger_path = tmp_path / "ledger.csv"
    _write_ledger(
        ledger_path,
        ("train-source", "dev-source", "test-source"),
        ood_use="prohibited",
    )

    with pytest.raises(SourceMatrixError, match="ood_evaluation_use"):
        validate_source_mixed_research_matrix(
            load_source_mixed_research_matrix(tmp_path / "matrix.json"),
            load_license_ledger(ledger_path),
        )
