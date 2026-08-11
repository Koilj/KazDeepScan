from __future__ import annotations

import json
from pathlib import Path

import pytest

from kds.data.licenses import load_license_ledger
from kds.data.manifest import ManifestRow, write_manifest
from kds.data.unseen_generator_ood import (
    UnseenGeneratorSuiteError,
    load_unseen_generator_suite,
    validate_and_select_unseen_generator_suite,
    validate_unseen_generator_suite,
)
from tests.factories import manifest_mapping


def _write_ledger(path: Path, source_ids: set[str]) -> None:
    lines = [
        "source_id,usage_scope,train_dev_test_use,ood_evaluation_use,"
        "bonafide_group_provenance,spoof_voice_group_provenance,license,source_url,"
        "artifact_name,expected_size_bytes,last_modified_utc,sha256,rights_basis,status,notes"
    ]
    for source_id in sorted(source_ids):
        lines.append(
            f"{source_id},personal_research,research_only,research_only,unknown,unknown,"
            "CC-BY-4.0,https://example.test/source,source.tar,1024,2026-08-10T00:00:00Z,"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,"
            "research,verified,test"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row(
    *,
    sample_id: str,
    sha256: str,
    source_id: str,
    split: str,
    label: str,
    text_hash: str,
    family: str = "",
) -> ManifestRow:
    spoof = label == "spoof"
    return ManifestRow.from_mapping(
        manifest_mapping(
            sample_id=sample_id,
            sha256=sha256,
            source_name=source_id,
            split=split,
            label=label,
            parent_group_id=f"{source_id}-{sample_id}",
            speaker_pseudo_id=f"{source_id}-{sample_id}",
            text_id=f"{source_id}-{text_hash}",
            text_hash=text_hash,
            generator_family=family if spoof else "",
            generator_name="test-generator" if spoof else "",
            generator_version="1" if spoof else "",
            voice_id="control" if spoof else "",
        ),
        row_number=2,
    )


def _write_binary_role(path: Path, source_id: str, split: str, offset: int, family: str) -> None:
    write_manifest(
        path,
        [
            _row(
                sample_id=f"{source_id}-b",
                sha256=f"{offset:064x}",
                source_id=source_id,
                split=split,
                label="bonafide",
                text_hash=f"{offset + 1:064x}",
            ),
            _row(
                sample_id=f"{source_id}-s",
                sha256=f"{offset + 2:064x}",
                source_id=source_id,
                split=split,
                label="spoof",
                text_hash=f"{offset + 3:064x}",
                family=family,
            ),
        ],
    )


def _write_final(
    path: Path, *, test_id: str, family: str, offset: int, text_hash: str | None = None
) -> None:
    pair_text_hash = text_hash or f"{offset + 1:064x}"
    write_manifest(
        path,
        [
            _row(
                sample_id=f"{test_id}-b",
                sha256=f"{offset:064x}",
                source_id="base-source",
                split="test",
                label="bonafide",
                text_hash=pair_text_hash,
            ),
            _row(
                sample_id=f"{test_id}-s",
                sha256=f"{offset + 2:064x}",
                source_id=f"{test_id}-source",
                split="test",
                label="spoof",
                text_hash=pair_text_hash,
                family=family,
            ),
        ],
    )


def _write_suite(path: Path, families: tuple[str, str, str]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": "suite-test",
                "purpose": "research",
                "train": {
                    "manifest": "train.csv",
                    "source_split": "train",
                    "expected_source_ids": ["train-source"],
                },
                "dev": {
                    "manifest": "dev.csv",
                    "source_split": "dev",
                    "expected_source_ids": ["dev-source"],
                },
                "shared_final_source_ids": ["base-source"],
                "final_tests": [
                    {
                        "id": "first",
                        "manifest": "first.csv",
                        "source_split": "test",
                        "expected_source_ids": ["base-source", "first-source"],
                        "expected_generator_families": [families[0]],
                    },
                    {
                        "id": "second",
                        "manifest": "second.csv",
                        "source_split": "test",
                        "expected_source_ids": ["base-source", "second-source"],
                        "expected_generator_families": [families[1]],
                    },
                    {
                        "id": "third",
                        "manifest": "third.csv",
                        "source_split": "test",
                        "expected_source_ids": ["base-source", "third-source"],
                        "expected_generator_families": [families[2]],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _prepare_suite(tmp_path: Path, *, repeated_final_text: bool = False) -> Path:
    _write_binary_role(tmp_path / "train.csv", "train-source", "train", 10, "seen-train")
    _write_binary_role(tmp_path / "dev.csv", "dev-source", "dev", 20, "seen-dev")
    _write_final(tmp_path / "first.csv", test_id="first", family="family-one", offset=30)
    first_text = f"{31:064x}" if repeated_final_text else None
    _write_final(
        tmp_path / "second.csv",
        test_id="second",
        family="family-two",
        offset=40,
        text_hash=first_text,
    )
    _write_final(tmp_path / "third.csv", test_id="third", family="family-three", offset=50)
    _write_suite(tmp_path / "suite.json", ("family-one", "family-two", "family-three"))
    _write_ledger(
        tmp_path / "ledger.csv",
        {
            "train-source",
            "dev-source",
            "base-source",
            "first-source",
            "second-source",
            "third-source",
        },
    )
    return tmp_path / "ledger.csv"


def test_unseen_generator_suite_accepts_three_disjoint_frozen_final_families(
    tmp_path: Path,
) -> None:
    ledger_path = _prepare_suite(tmp_path)

    report, selected = validate_and_select_unseen_generator_suite(
        load_unseen_generator_suite(tmp_path / "suite.json"), load_license_ledger(ledger_path)
    )

    assert report.train_dev_generator_families == ("seen-dev", "seen-train")
    assert [test.generator_families for test in report.final_tests] == [
        ("family-one",),
        ("family-two",),
        ("family-three",),
    ]
    assert len(selected.train) == 2
    assert len(selected.dev) == 2
    assert [(test.test_id, len(test.rows)) for test in selected.final_tests] == [
        ("first", 2),
        ("second", 2),
        ("third", 2),
    ]


def test_unseen_generator_suite_rejects_text_reused_by_final_tests(tmp_path: Path) -> None:
    ledger_path = _prepare_suite(tmp_path, repeated_final_text=True)

    with pytest.raises(UnseenGeneratorSuiteError, match="overlap by text_hash"):
        validate_unseen_generator_suite(
            load_unseen_generator_suite(tmp_path / "suite.json"), load_license_ledger(ledger_path)
        )
