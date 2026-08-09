from __future__ import annotations

from pathlib import Path

import pytest

from kds.cli import main
from kds.data.consents import (
    ConsentRegistryError,
    load_consent_registry,
    product_eligible_speaker_ids,
)

HEADER = (
    "consent_record_id,speaker_pseudo_id,language,collection_version,"
    "product_training_authorized,synthetic_derivatives_authorized,"
    "commercial_deployment_authorized,status,signed_at,revoked_at\n"
)


def _row(
    *,
    consent_record_id: str = "consent-001",
    speaker_pseudo_id: str = "speaker-001",
    product_training_authorized: str = "true",
    synthetic_derivatives_authorized: str = "true",
    commercial_deployment_authorized: str = "true",
    status: str = "active",
    revoked_at: str = "",
) -> str:
    return (
        f"{consent_record_id},{speaker_pseudo_id},kk,consented-product-v1,"
        f"{product_training_authorized},{synthetic_derivatives_authorized},"
        f"{commercial_deployment_authorized},{status},2026-08-09T00:00:00Z,{revoked_at}\n"
    )


def test_product_consent_registry_returns_active_eligible_speakers(tmp_path: Path) -> None:
    path = tmp_path / "consents.csv"
    path.write_text(HEADER + _row(), encoding="utf-8")

    entries = load_consent_registry(path)

    assert product_eligible_speaker_ids(entries) == frozenset({"speaker-001"})


def test_product_consent_registry_rejects_missing_scope(tmp_path: Path) -> None:
    path = tmp_path / "consents.csv"
    path.write_text(HEADER + _row(synthetic_derivatives_authorized="false"), encoding="utf-8")

    with pytest.raises(ConsentRegistryError, match="synthetic_derivatives_authorized"):
        product_eligible_speaker_ids(load_consent_registry(path))


def test_consent_registry_allows_revoked_history_but_not_duplicate_active_speaker(
    tmp_path: Path,
) -> None:
    path = tmp_path / "consents.csv"
    path.write_text(
        HEADER
        + _row(status="revoked", revoked_at="2026-08-10T00:00:00Z")
        + _row(consent_record_id="consent-002"),
        encoding="utf-8",
    )

    assert product_eligible_speaker_ids(load_consent_registry(path)) == frozenset({"speaker-001"})


def test_consent_registry_rejects_duplicate_active_speaker(tmp_path: Path) -> None:
    path = tmp_path / "consents.csv"
    path.write_text(HEADER + _row() + _row(consent_record_id="consent-002"), encoding="utf-8")

    with pytest.raises(ConsentRegistryError, match="Duplicate active consent"):
        load_consent_registry(path)


def test_consent_registry_cli_reports_eligible_speakers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "consents.csv"
    path.write_text(HEADER + _row(), encoding="utf-8")

    assert main(["validate-consent-registry", str(path)]) == 0

    assert '"product_eligible_speakers": 1' in capsys.readouterr().out
