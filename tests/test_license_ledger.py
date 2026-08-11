from __future__ import annotations

from pathlib import Path

import pytest

from kds.data.licenses import (
    LicenseLedgerError,
    TrainingProtocolError,
    load_license_ledger,
    validate_manifest_licenses,
    validate_training_protocol,
    write_license_ledger_snapshot,
)
from kds.data.manifest import ManifestRow
from tests.factories import manifest_mapping


def _write_ledger(
    path: Path,
    status: str = "verified",
    train_dev_test_use: str = "product_allowed",
    ood_evaluation_use: str = "product_allowed",
    bonafide_group_provenance: str = "verified",
    spoof_voice_group_provenance: str = "verified",
) -> None:
    path.write_text(
        "source_id,usage_scope,train_dev_test_use,ood_evaluation_use,"
        "bonafide_group_provenance,spoof_voice_group_provenance,license,source_url,artifact_name,expected_size_bytes,"
        "last_modified_utc,sha256,rights_basis,status,notes\n"
        "approved-source,commercial_clean,"
        f"{train_dev_test_use},{ood_evaluation_use},{bonafide_group_provenance},{spoof_voice_group_provenance},"
        "license-v1,https://example.test/source,source.tar,"
        "1024,2026-08-09T00:00:00Z,"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,"
        f"consent-001,{status},local test record\n",
        encoding="utf-8",
    )


def _row(source_name: str = "approved-source") -> ManifestRow:
    return ManifestRow.from_mapping(manifest_mapping(source_name=source_name), row_number=2)


def test_verified_ledger_source_allows_manifest(tmp_path: Path) -> None:
    ledger_path = tmp_path / "license_ledger.csv"
    _write_ledger(ledger_path)

    validate_manifest_licenses([_row()], load_license_ledger(ledger_path))


def test_pending_ledger_source_cannot_be_used_in_manifest(tmp_path: Path) -> None:
    ledger_path = tmp_path / "license_ledger.csv"
    _write_ledger(ledger_path, status="scope_confirmation_required")

    with pytest.raises(LicenseLedgerError, match="not approved"):
        validate_manifest_licenses([_row()], load_license_ledger(ledger_path))


def test_owner_authorized_personal_research_source_allows_manifest(tmp_path: Path) -> None:
    ledger_path = tmp_path / "license_ledger.csv"
    _write_ledger(ledger_path, status="owner_authorized_personal_research")

    validate_manifest_licenses([_row()], load_license_ledger(ledger_path))


def test_manifest_source_must_be_listed_in_ledger(tmp_path: Path) -> None:
    ledger_path = tmp_path / "license_ledger.csv"
    _write_ledger(ledger_path)

    with pytest.raises(LicenseLedgerError, match="missing from the license ledger"):
        validate_manifest_licenses([_row("unlisted-source")], load_license_ledger(ledger_path))


def test_verified_ledger_entry_requires_archive_digest(tmp_path: Path) -> None:
    ledger_path = tmp_path / "license_ledger.csv"
    _write_ledger(ledger_path)
    content = ledger_path.read_text(encoding="utf-8")
    ledger_path.write_text(content.replace("a" * 64, ""), encoding="utf-8")

    with pytest.raises(LicenseLedgerError, match="requires an archive SHA-256"):
        load_license_ledger(ledger_path)


def test_license_ledger_snapshot_is_minimal_deterministic_and_write_once(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "license_ledger.csv"
    _write_ledger(ledger_path)
    snapshot_path = tmp_path / "frozen" / "plan_v1.csv"
    snapshot_path.parent.mkdir()
    ledger = load_license_ledger(ledger_path)

    assert write_license_ledger_snapshot(
        snapshot_path, ledger, source_ids=["approved-source", "approved-source"]
    ) == ("approved-source",)
    assert load_license_ledger(snapshot_path) == ledger
    with pytest.raises(LicenseLedgerError, match="Unsafe license ledger snapshot"):
        write_license_ledger_snapshot(
            snapshot_path, ledger, source_ids=["approved-source"]
        )


def test_license_ledger_snapshot_rejects_unknown_source(tmp_path: Path) -> None:
    ledger_path = tmp_path / "license_ledger.csv"
    _write_ledger(ledger_path)

    with pytest.raises(LicenseLedgerError, match="sources are missing"):
        write_license_ledger_snapshot(
            tmp_path / "snapshot.csv",
            load_license_ledger(ledger_path),
            source_ids=["missing-source"],
        )


def test_training_protocol_requires_explicit_source_policy(tmp_path: Path) -> None:
    ledger_path = tmp_path / "license_ledger.csv"
    _write_ledger(ledger_path, train_dev_test_use="research_only")
    ledger = load_license_ledger(ledger_path)
    rows = [
        ManifestRow.from_mapping(
            manifest_mapping(
                sample_id=f"{split}-{label}",
                source_name="approved-source",
                sha256=("a" if index % 2 else "b") * 63 + str(index % 10),
                split=split,
                label=label,
                parent_group_id=f"parent-{split}-{label}",
                speaker_pseudo_id=f"speaker-{split}-{label}",
                text_id=f"text-{split}-{label}",
                text_hash=f"text-hash-{split}-{label}",
                generator_family=("ood-generator" if split == "ood" else "generator")
                if label == "spoof"
                else "",
                generator_name=("ood-generator-v1" if split == "ood" else "generator-v1")
                if label == "spoof"
                else "",
                generator_version="1" if label == "spoof" else "",
                voice_id=f"voice-{split}" if label == "spoof" else "",
            ),
            row_number=index + 2,
        )
        for index, (split, label) in enumerate(
            (split, label)
            for split in ("train", "dev", "test", "ood")
            for label in ("bonafide", "spoof")
        )
    ]

    report = validate_training_protocol(rows, ledger, purpose="research")
    assert report.split_counts == {"train": 2, "dev": 2, "test": 2}

    with pytest.raises(TrainingProtocolError, match="not product-allowed"):
        validate_training_protocol(rows, ledger, purpose="product")
