from __future__ import annotations

from pathlib import Path

import pytest

from kds.data.licenses import TrainingProtocolError, load_license_ledger, validate_training_protocol
from kds.data.manifest import ManifestRow
from tests.factories import manifest_mapping


def _write_product_ledger(path: Path, *, bonafide_group: str = "verified") -> None:
    path.write_text(
        "source_id,usage_scope,train_dev_test_use,ood_evaluation_use,"
        "bonafide_group_provenance,spoof_voice_group_provenance,license,source_url,"
        "artifact_name,expected_size_bytes,last_modified_utc,sha256,rights_basis,status,notes\n"
        "candidate,commercial_clean,product_allowed,product_allowed,"
        f"{bonafide_group},verified,license-v1,https://example.test/source,source.tar,1024,"
        "2026-08-09T00:00:00Z,"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,"
        "consent-001,verified,local test record\n",
        encoding="utf-8",
    )


def _binary_product_rows(*, leak_voice: bool = False) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for index, (split, label) in enumerate(
        (split, label)
        for split in ("train", "dev", "test", "ood")
        for label in ("bonafide", "spoof")
    ):
        spoof = label == "spoof"
        generator_family = "ood-generator" if split == "ood" and spoof else "train-generator"
        voice_id = "voice-train" if leak_voice and split == "dev" and spoof else f"voice-{split}"
        rows.append(
            ManifestRow.from_mapping(
                manifest_mapping(
                    sample_id=f"{split}-{label}",
                    source_name="candidate",
                    sha256=f"{index:064x}",
                    split=split,
                    label=label,
                    parent_group_id=f"parent-{split}-{label}",
                    speaker_pseudo_id=f"speaker-{split}-{label}",
                    text_id=f"text-{split}-{label}",
                    text_hash=f"text-hash-{split}-{label}",
                    generator_family=generator_family if spoof else "",
                    generator_name=f"{generator_family}-v1" if spoof else "",
                    generator_version="1" if spoof else "",
                    voice_id=voice_id if spoof else "",
                ),
                row_number=index + 2,
            )
        )
    return rows


def test_product_protocol_accepts_explicitly_eligible_binary_manifest(tmp_path: Path) -> None:
    ledger_path = tmp_path / "license_ledger.csv"
    _write_product_ledger(ledger_path)

    report = validate_training_protocol(
        _binary_product_rows(), load_license_ledger(ledger_path), purpose="product"
    )

    assert report.purpose == "product"
    assert report.split_counts == {"train": 2, "dev": 2, "test": 2, "ood": 2}


def test_product_protocol_rejects_reused_verified_spoof_voice(tmp_path: Path) -> None:
    ledger_path = tmp_path / "license_ledger.csv"
    _write_product_ledger(ledger_path)

    with pytest.raises(TrainingProtocolError, match="Leakage: spoof voice_id"):
        validate_training_protocol(
            _binary_product_rows(leak_voice=True),
            load_license_ledger(ledger_path),
            purpose="product",
        )


def test_product_protocol_rejects_source_provided_group_as_unverified(tmp_path: Path) -> None:
    ledger_path = tmp_path / "license_ledger.csv"
    _write_product_ledger(ledger_path, bonafide_group="source_provided")

    with pytest.raises(TrainingProtocolError, match="no verified bonafide_group_provenance"):
        validate_training_protocol(
            _binary_product_rows(), load_license_ledger(ledger_path), purpose="product"
        )
