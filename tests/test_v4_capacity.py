from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from kds.data.licenses import LICENSE_LEDGER_FIELD_ORDER, LicenseLedgerEntry
from kds.data.manifest import ManifestRow, write_manifest
from kds.data.v4_capacity import (
    V4CapacityError,
    audit_v4_local_capacity,
    load_v4_gate_a_config,
    write_v4_capacity_receipt,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _entry(source_id: str, artifact: Path, *, archive_identity: bool = False) -> LicenseLedgerEntry:
    payload_size = 123 if archive_identity else artifact.stat().st_size
    payload_hash = "a" * 64 if archive_identity else _sha256(artifact.read_bytes())
    return LicenseLedgerEntry(
        source_id=source_id,
        usage_scope="personal_research",
        train_dev_test_use="research_only",
        ood_evaluation_use="research_only",
        bonafide_group_provenance="unknown",
        spoof_voice_group_provenance="unknown",
        license="test-license",
        source_url="https://example.test/source",
        artifact_name=artifact.name,
        expected_size_bytes=payload_size,
        last_modified_utc="2026-08-14T00:00:00Z",
        sha256=payload_hash,
        rights_basis="test-only fixture",
        status="verified",
        notes="unit test",
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _manifest_row() -> ManifestRow:
    return ManifestRow(
        sample_id="historical-ksc2",
        relative_path="history.wav",
        sha256="b" * 64,
        split="test",
        label="bonafide",
        language="kk",
        code_switch="unknown",
        parent_group_id="history-group",
        source_name="ksc2_v1",
        source_license="CC-BY-4.0",
        rights_basis="unit test",
        speaker_pseudo_id="unknown:ksc2",
        text_id="history-text",
        text_hash="c" * 64,
        duration_s=1.0,
        generator_family="",
        generator_name="",
        generator_version="",
        voice_id="",
        clone_consent_id="",
        device="unknown",
        capture_route="archive",
        original_sr=16000,
        codec="pcm_s16le",
        augmentation_chain="",
        augmentation_seed="",
        created_at="2026-08-14T00:00:00Z",
    )


def _build_gate_fixture(root: Path) -> dict[str, Path]:
    config_path = root / "configs/research/v4/gate.json"
    manifest_path = root / "data/manifests/history.csv"
    ledger_path = root / "data/licenses/license_ledger.csv"
    ruasd_catalog = root / "data/licenses/ruasd_catalog.csv"
    ksc2_lock = root / "data/licenses/ksc2_lock.json"
    for directory in (
        config_path.parent,
        manifest_path.parent,
        ledger_path.parent,
        root / "models/research",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    with ruasd_catalog.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "archive_name",
                "expected_size_bytes",
                "sha256",
                "pinned_revision",
                "source_url",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for index in range(250):
            writer.writerow(
                {
                    "archive_name": f"ruasd-{index:06d}.tar",
                    "expected_size_bytes": 4,
                    "sha256": f"{index:064x}",
                    "pinned_revision": "test",
                    "source_url": "https://example.test/ruasd",
                }
            )
    _write_json(
        ksc2_lock,
        {
            "multipart_archive": {
                "compressed_bytes": 1000,
                "compressed_sha256": "d" * 64,
                "parts": [{"filename": "part", "size_bytes": 1000, "sha256": "e" * 64}],
            }
        },
    )
    write_manifest(manifest_path, [_manifest_row()])

    entries = [
        replace(_entry("ruasd", ruasd_catalog), expected_size_bytes=1000),
        _entry("ksc2", ksc2_lock),
        _entry("common_voice", root / "common-voice-archive", archive_identity=True),
    ]
    routes: list[dict[str, str]] = []
    for index in range(4):
        lock_path = root / f"configs/research/tts_{index}.json"
        model_root = root / f"models/research/tts_{index}"
        bundle = model_root / f"bundle_{index}"
        bundle.mkdir(parents=True)
        artifact = bundle / "model.bin"
        artifact.write_bytes(f"model-{index}".encode())
        _write_json(
            lock_path,
            {
                "schema_version": 1,
                "protocol_id": f"test-lock-{index}",
                "models": [
                    {
                        "model_id": f"model_{index}",
                        "destination": f"bundle_{index}",
                        "generator_family": f"family_{index}",
                        "generator_name": f"generator {index}",
                        "generator_version": "1",
                        "license": "test-license",
                        "source_url": "https://example.test/model",
                        "runtime": {"kind": "test"},
                        "artifacts": [
                            {
                                "relative_path": "model.bin",
                                "url": "https://example.test/model.bin",
                                "expected_size_bytes": artifact.stat().st_size,
                                "sha256": _sha256(artifact.read_bytes()),
                            }
                        ],
                    }
                ],
            },
        )
        entries.append(_entry(f"tts_{index}", lock_path))
        routes.append(
            {
                "route_id": f"route-{index}",
                "ledger_source_id": f"tts_{index}",
                "lock_path": lock_path.relative_to(root).as_posix(),
                "model_root": model_root.relative_to(root).as_posix(),
            }
        )

    with ledger_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LICENSE_LEDGER_FIELD_ORDER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(entry) for entry in entries)

    _write_json(
        config_path,
        {
            "schema_version": 1,
            "protocol_id": "test-v4-gate-a",
            "purpose": "personal_research",
            "inventory": {
                "file_globs": [
                    "configs/research/**/*.json",
                    "data/licenses/**/*.csv",
                    "data/licenses/**/*.json",
                    "data/manifests/**/*.csv",
                ]
            },
            "targets": {
                "preferred_train_rows": 24000,
                "minimum_train_rows": 20000,
                "maximum_train_rows": 30000,
                "preferred_rows_per_cell": 6000,
                "minimum_rows_per_cell": 5000,
                "final_rows_per_cell": 500,
                "minimum_train_tts_families": 4,
            },
            "sources": {
                "ruasd": {
                    "ledger_source_id": "ruasd",
                    "catalog_path": ruasd_catalog.relative_to(root).as_posix(),
                    "raw_bonafide_record_key": "real/raw/real_speech",
                    "raw_spoof_record_key": "fake/raw/tts",
                    "excluded_subset_keys": ["real/raw/CommonVoice"],
                    "history_source_ids": ["ruasd"],
                },
                "common_voice": {"ledger_source_id": "common_voice"},
                "ksc2": {
                    "ledger_source_id": "ksc2",
                    "artifact_lock_path": ksc2_lock.relative_to(root).as_posix(),
                    "allowed_train_components": ["Train/radio"],
                    "history_source_ids": ["ksc2_v1"],
                },
                "tts_routes": routes,
            },
            "claims": {
                "local_sources_only": True,
                "speaker_independence": "not_verified_speaker_independent",
                "capacity_is_pre_qa": True,
                "training_authorized_by_gate_a": False,
                "synthesis_authorized_by_gate_a": False,
            },
        },
    )

    ruasd_audit = root / "ruasd-audit.json"
    _write_json(
        ruasd_audit,
        {
            "archive_count": 250,
            "sha256_verified_archives": 250,
            "record_counts": {"real/raw/real_speech": 7000, "fake/raw/tts": 7000},
            "subset_counts": {"real/raw/CommonVoice": 100},
        },
    )
    ksc2_audit = root / "ksc2-audit.json"
    _write_json(
        ksc2_audit,
        {
            "compressed_bytes": 1000,
            "compressed_sha256": "d" * 64,
            "parts": [{"filename": "part", "size_bytes": 1000, "sha256": "e" * 64}],
            "files_by_component": {"Train/radio": 14002},
            "unpaired_audio_files": 0,
        },
    )
    cv_screen = root / "cv-screen.json"
    configuration_files = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path.read_bytes()),
        }
        for path in sorted((root / "configs/research").glob("*.json"))
    ]
    inventory_manifests = [
        {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": _sha256(manifest_path.read_bytes()),
        }
    ]
    _write_json(
        cv_screen,
        {
            "archive": {
                "expected_size_bytes": 123,
                "expected_sha256": "a" * 64,
                "identity_verified_before_metadata_read": True,
            },
            "scope": {
                "configuration_directory": "configs/research",
                "configuration_files": configuration_files,
                "manifest_inventory_directory": "data/manifests",
                "inventory_manifests": inventory_manifests,
            },
            "strict_group_exclusion": {"surviving_records": 600},
        },
    )
    return {
        "config": config_path,
        "ledger": ledger_path,
        "ruasd": ruasd_audit,
        "ksc2": ksc2_audit,
        "cv": cv_screen,
    }


def test_repository_v4_gate_a_config_loads_strictly() -> None:
    config = load_v4_gate_a_config(
        Path("configs/research/v4/xlsr_sls_model_v4_gate_a_v1.json")
    )

    assert config.targets.preferred_train_rows == 24_000
    assert len(config.tts_routes) == 4


def test_v4_gate_a_verifies_inventory_models_and_conservative_capacity(tmp_path: Path) -> None:
    paths = _build_gate_fixture(tmp_path)

    receipt = audit_v4_local_capacity(
        config_path=paths["config"],
        project_root=tmp_path,
        license_ledger_path=paths["ledger"],
        ruasd_audit_path=paths["ruasd"],
        ksc2_audit_path=paths["ksc2"],
        common_voice_screen_path=paths["cv"],
        audited_at="2026-08-14T12:00:00+06:00",
    )

    assert receipt["decision"] == "proceed_24k"
    capacity = receipt["capacity"]
    assert isinstance(capacity, dict)
    cells = capacity["cells"]
    assert isinstance(cells, dict)
    kk_bonafide = cells["kk/bonafide"]
    assert isinstance(kk_bonafide, dict)
    assert kk_bonafide["pre_qa_candidate_upper_bound"] == 7000
    assert receipt["claims"]["ready_train_rows_certified"] is False  # type: ignore[index]


def test_v4_gate_a_receipt_never_overwrites(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    write_v4_capacity_receipt(output, {"decision": "stop_local_capacity_exhausted"})

    with pytest.raises(V4CapacityError, match="Unsafe v4 Gate A receipt destination"):
        write_v4_capacity_receipt(output, {"decision": "proceed_24k"})
