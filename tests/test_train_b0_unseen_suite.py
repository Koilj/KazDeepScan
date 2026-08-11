from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import soundfile as sf  # type: ignore[import-untyped]
import torch

from kds.data.manifest import ManifestRow, write_manifest
from scripts.train_b0_unseen_suite import main
from tests.factories import manifest_mapping


def _audio_row(
    root: Path,
    *,
    sample_id: str,
    source_id: str,
    split: str,
    label: str,
    text_hash: str,
    value: float,
    family: str = "",
) -> ManifestRow:
    path = root / f"{sample_id}.wav"
    samples = [value] * 1_600
    for index, byte in enumerate(sample_id.encode("utf-8")):
        samples[index] = value + byte / 100_000
    sf.write(path, samples, 16_000, subtype="FLOAT")
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    spoof = label == "spoof"
    return ManifestRow.from_mapping(
        manifest_mapping(
            sample_id=sample_id,
            relative_path=path.name,
            sha256=sha256,
            source_name=source_id,
            split=split,
            label=label,
            parent_group_id=f"parent-{sample_id}",
            speaker_pseudo_id=f"speaker-{sample_id}",
            text_id=f"text-{sample_id}",
            text_hash=text_hash,
            generator_family=family if spoof else "",
            generator_name=f"generator-{family}" if spoof else "",
            generator_version="1" if spoof else "",
            voice_id=f"voice-{family}" if spoof else "",
            original_sr="16000",
        ),
        row_number=2,
    )


def _write_binary_role(root: Path, name: str, source: str, split: str, value: float) -> None:
    write_manifest(
        root / name,
        [
            _audio_row(
                root,
                sample_id=f"{source}-bonafide",
                source_id=source,
                split=split,
                label="bonafide",
                text_hash=f"hash-{source}-bonafide",
                value=value,
            ),
            _audio_row(
                root,
                sample_id=f"{source}-spoof",
                source_id=source,
                split=split,
                label="spoof",
                text_hash=f"hash-{source}-spoof",
                value=-value,
                family=f"seen-{source}",
            ),
        ],
    )


def _write_final(root: Path, index: int) -> tuple[str, str, str]:
    test_id = f"final-{index}"
    source_id = f"synthetic-{index}"
    family = f"unseen-family-{index}"
    text_hash = f"final-text-hash-{index}"
    write_manifest(
        root / f"{test_id}.csv",
        [
            _audio_row(
                root,
                sample_id=f"{test_id}-bonafide",
                source_id="shared-base",
                split="test",
                label="bonafide",
                text_hash=text_hash,
                value=0.01 * index,
            ),
            _audio_row(
                root,
                sample_id=f"{test_id}-spoof",
                source_id=source_id,
                split="test",
                label="spoof",
                text_hash=text_hash,
                value=-0.01 * index,
                family=family,
            ),
        ],
    )
    return test_id, source_id, family


def _prepare_run(root: Path) -> Path:
    _write_binary_role(root, "train.csv", "train-source", "train", 0.02)
    _write_binary_role(root, "dev.csv", "dev-source", "dev", 0.03)
    finals = [_write_final(root, index) for index in range(1, 4)]
    suite_path = root / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": "runner-test-suite",
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
                "shared_final_source_ids": ["shared-base"],
                "final_tests": [
                    {
                        "id": test_id,
                        "manifest": f"{test_id}.csv",
                        "source_split": "test",
                        "expected_source_ids": ["shared-base", source_id],
                        "expected_generator_families": [family],
                    }
                    for test_id, source_id, family in finals
                ],
            }
        ),
        encoding="utf-8",
    )
    sources = [
        "train-source",
        "dev-source",
        "shared-base",
        *(source_id for _test_id, source_id, _family in finals),
    ]
    ledger_lines = [
        "source_id,usage_scope,train_dev_test_use,ood_evaluation_use,"
        "bonafide_group_provenance,spoof_voice_group_provenance,license,source_url,"
        "artifact_name,expected_size_bytes,last_modified_utc,sha256,rights_basis,status,notes"
    ]
    for source_id in sources:
        ledger_lines.append(
            f"{source_id},personal_research,research_only,research_only,unknown,unknown,"
            "CC-BY-4.0,https://example.test/source,source.tar,1024,2026-08-10T00:00:00Z,"
            + "a" * 64
            + ",research,verified,test"
        )
    (root / "ledger.csv").write_text("\n".join(ledger_lines) + "\n", encoding="utf-8")

    plan_path = root / "plan.json"
    pinned_manifests = [
        root / "train.csv",
        root / "dev.csv",
        *(root / f"{test_id}.csv" for test_id, _source_id, _family in finals),
    ]
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "runner-e2e-test",
                "purpose": "research",
                "suite": {
                    "path": suite_path.name,
                    "sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
                },
                "license_ledger": {
                    "path": "ledger.csv",
                    "sha256": hashlib.sha256((root / "ledger.csv").read_bytes()).hexdigest(),
                },
                "manifests": [
                    {
                        "path": manifest_path.name,
                        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    }
                    for manifest_path in pinned_manifests
                ],
                "model": {
                    "name": "b0_logmel_cnn",
                    "config": {
                        "sample_rate": 16000,
                        "n_fft": 128,
                        "hop_length": 64,
                        "n_mels": 16,
                        "dropout": 0.0,
                    },
                },
                "training": {
                    "seed": 20260818,
                    "epochs": 1,
                    "batch_size": 2,
                    "window_samples": 1600,
                    "learning_rate": 0.0001,
                    "weight_decay": 0.0001,
                    "num_workers": 0,
                    "device": "cpu",
                },
                "outputs": {"checkpoint": "checkpoint.pt", "report": "report.json"},
            }
        ),
        encoding="utf-8",
    )
    return plan_path


def test_runner_publishes_one_checkpoint_and_complete_multi_final_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = _prepare_run(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_b0_unseen_suite.py",
            "--plan",
            str(plan_path),
            "--audio-root",
            str(tmp_path),
        ],
    )

    assert main() == 0

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert report["best_epoch"] == 1
    assert [item["test_id"] for item in report["final_tests"]] == [
        "final-1",
        "final-2",
        "final-3",
    ]
    assert all(item["stratified_metrics"] for item in report["final_tests"])
    assert checkpoint["frozen_run"]["selected_state_sha256"] == report["selected_state_sha256"]

    with pytest.raises(ValueError, match="Refusing to repeat or overwrite"):
        main()
