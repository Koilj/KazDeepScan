from __future__ import annotations

from pathlib import Path

import pytest

from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestRow
from tests.factories import manifest_mapping


def _write_ledger(path: Path, status: str = "verified") -> None:
    path.write_text(
        "source_id,usage_scope,license,source_url,artifact_name,expected_size_bytes,"
        "last_modified_utc,sha256,rights_basis,status,notes\n"
        "approved-source,commercial_clean,license-v1,https://example.test/source,source.tar,"
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
