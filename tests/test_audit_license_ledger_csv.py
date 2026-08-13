from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest


def _script() -> Any:
    spec = spec_from_file_location(
        "kds_audit_license_ledger_csv",
        "scripts/audit_license_ledger_csv.py",
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_ledger() -> str:
    return (
        "source_id,usage_scope,train_dev_test_use,ood_evaluation_use,"
        "bonafide_group_provenance,spoof_voice_group_provenance,license,source_url,"
        "artifact_name,expected_size_bytes,last_modified_utc,sha256,rights_basis,status,notes\n"
        "source,personal_research,research_only,research_only,verified,not_applicable,"
        "CC-BY-4.0,https://example.test/source,source.tar,1,2026-08-14T00:00:00Z,"
        f"{'a' * 64},rights,verified,\"complete note, with comma\"\n"
    )


def test_current_mutable_license_ledger_has_strict_csv_shape() -> None:
    assert _script().audit_license_ledger_csv(
        Path("data/licenses/license_ledger.csv")
    ) == 23


def test_audit_rejects_extra_unquoted_row_field(tmp_path: Path) -> None:
    script = _script()
    path = tmp_path / "ledger.csv"
    path.write_text(
        _valid_ledger().replace('"complete note, with comma"', "complete note, with comma"),
        encoding="utf-8",
    )

    with pytest.raises(script.LicenseLedgerCsvShapeError, match="expected 15 fields, got 16"):
        script.audit_license_ledger_csv(path)


def test_audit_rejects_non_exact_header(tmp_path: Path) -> None:
    script = _script()
    path = tmp_path / "ledger.csv"
    path.write_text(
        _valid_ledger().replace("source_id,usage_scope", "source_id,source_id,usage_scope"),
        encoding="utf-8",
    )

    with pytest.raises(script.LicenseLedgerCsvShapeError, match="header must exactly match"):
        script.audit_license_ledger_csv(path)
